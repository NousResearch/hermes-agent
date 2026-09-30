import { Button } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

import { CanonicalGroupAttachments } from './canonical-group-attachments'
import { type CanonicalGroupEvent, CanonicalGroupHistory } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { prepareCanonicalGroupSend, readCanonicalGroupSend, retireCanonicalGroupSend } from './canonical-group-send'
import type { PreparedCanonicalGroupSend } from './canonical-group-send'
import { actCanonicalGroup, canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalPendingAction } from './canonical-groups'

type RoomEvent = CanonicalGroupEvent
interface Attachment { attachment_id?: string; event_id?: string; kind: string; name: string; mime: string; size?: number }
interface DriverStatus {
  running?: boolean
  working?: boolean
  blocked?: boolean
  pending_actions?: CanonicalPendingAction[]
}
interface RoomState { room: { name: string; authority_epoch?: number }; driver_status?: DriverStatus }
type Labels = ReturnType<typeof useCanonicalGroupLabels>

// A 4001 with one of these reasons means nothing was accepted; any other failure may have been.
const TERMINAL_SEND_REFUSALS = new Set(['invalid_params', 'permission_denied', 'unknown_execution', 'stale_generation'])

function sendOutcome(error: unknown): 'refused' | 'retryable' | 'unknown' {
  const failure = error as { code?: unknown; data?: { reason?: unknown } } | null

  if (failure?.code !== 4001) {return 'unknown'}

  return TERMINAL_SEND_REFUSALS.has(String(failure.data?.reason)) ? 'refused' : 'retryable'
}

/** Live work comes only from the driver; unresolved members are listed beside it, never instead. */
function roomStatus(status: DriverStatus, labels: Labels) {
  const actions = status.pending_actions || []
  const approvals = actions.filter(action => action.kind === 'approval').length
  const parts = [status.working ? labels.statusWorking : status.running === false ? labels.statusStopped : labels.statusIdle]

  if (status.blocked) {parts.push(labels.statusBlocked)}

  if (approvals) {parts.push(labels.statusApprovals.replace('{count}', String(approvals)))}

  if (actions.length > approvals) {parts.push(labels.statusAttention.replace('{count}', String(actions.length - approvals)))}

  return parts.join(' · ')
}

/** Room controls rendered by the owner of the binding (rename, disband). */
export type CanonicalRoomActions = (room: { name: string; refresh: () => void }) => ReactNode

export function CanonicalGroupWorkspace({ binding, visible = true, onBack, actions }: {
  binding: CanonicalGroupBinding; visible?: boolean; onBack?: () => void; actions?: CanonicalRoomActions
}) {
  // Remount on identity changes: old polls and pending confirmations never cross rooms.
  return <CanonicalRoomView actions={actions} binding={binding} key={JSON.stringify(binding)} onBack={onBack} visible={visible} />
}

function CanonicalRoomView({ binding: initialBinding, visible, onBack, actions }: {
  binding: CanonicalGroupBinding; visible: boolean; onBack?: () => void; actions?: CanonicalRoomActions
}) {
  const [binding] = useState(() => ({ ...initialBinding }))
  const labels = useCanonicalGroupLabels()
  const [state, setState] = useState<RoomState | null>(null)
  const [events, setEvents] = useState<RoomEvent[]>([])
  const [error, setError] = useState('')
  const [readError, setReadError] = useState('')
  const [draft, setDraft] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [restored, setRestored] = useState(false)
  const [pending, setPending] = useState<PreparedCanonicalGroupSend | null>(null)
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)
  const [stopping, setStopping] = useState(false)
  const [notice, setNotice] = useState('')
  const [sendHint, setSendHint] = useState('')
  const alive = useRef(true)
  const [discard, setDiscard] = useState<CanonicalPendingAction | null>(null)
  const revision = useRef(0)
  // The log is append-only within one authority epoch: read only what is new.
  const seen = useRef<{ epoch?: number; seq: number }>({ seq: 0 })

  // eslint-disable-next-line no-restricted-syntax -- journal hydration and mounted lifetime, not a reactive store mirror
  useEffect(() => {
    alive.current = true
    let cancelled = false
    void readCanonicalGroupSend(binding).then(entry => {
      if (cancelled) {return}

      if (entry) {
        setPending(entry)
        setDraft(String(entry.params.payload.text ?? ''))
        setAttachments((entry.params.payload.attachments as Attachment[] | undefined) ?? [])
      }

      setRestored(true)
    }).catch(e => { if (!cancelled) {setError(e instanceof Error ? e.message : String(e))} })

    return () => { cancelled = true; alive.current = false; revision.current++ }
  }, [binding])

  const refresh = async () => {
    const version = ++revision.current
    const snapshot = await canonicalGroupRequest<RoomState>(binding, 'groups.state', { room_id: binding.roomId })
    const epoch = snapshot.room?.authority_epoch
    const fresh = epoch !== seen.current.epoch
    const log: RoomEvent[] = []
    let cursor = fresh ? 0 : seen.current.seq

    for (;;) {
      const page = await canonicalGroupRequest<{ events: RoomEvent[]; has_more?: boolean; next_seq?: number }>(binding, 'groups.log', { room_id: binding.roomId, since_seq: cursor, limit: 100 })
      log.push(...page.events)

      if (!page.has_more) {break}
      const next = page.events.at(-1)?.seq

      if (!next || next <= cursor) {throw new Error(labels.invalidLogCursor)}
      cursor = next
    }

    if (alive.current && version === revision.current) {
      const last = seen.current.seq
      seen.current = { epoch, seq: log.at(-1)?.seq ?? cursor }
      setState(snapshot)
      setEvents(current => fresh ? log : [...current, ...log.filter(event => event.seq > last)])
      setReadError('')
    }
  }

  useEffect(() => {
    if (!visible) {return}
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const poll = async () => {
      try { await refresh() } catch (e) { if (!cancelled) {setReadError(String(e instanceof Error ? e.message : e))} }

      if (!cancelled) {timer = setTimeout(() => void poll(), 2000)}
    }

    void poll()

    return () => { cancelled = true; revision.current++; clearTimeout(timer) }
    // The keyed parent freezes the authority binding for this lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible])

  const mutate = async (operation: () => Promise<unknown>) => {
    if (busyRef.current) {return}
    busyRef.current = true
    setBusy(true)
    setError('')

    try { await operation();

 if (alive.current) {await refresh()} }
    catch (e) { if (alive.current) {setError(e instanceof Error ? e.message : String(e))} }
    finally {
      busyRef.current = false

      if (alive.current) {setBusy(false)}
    }
  }

  const send = () => {
    if (!restored || busyRef.current || !state?.driver_status || (!pending && !draft.trim() && !attachments.length)) {return}
    setSendHint('')
    void mutate(async () => {
      const exact = pending ?? await prepareCanonicalGroupSend(binding, { text: draft, attachments })

      if (!alive.current) {return}
      setPending(exact)
      setDraft(String(exact.params.payload.text ?? ''))
      setAttachments((exact.params.payload.attachments as Attachment[] | undefined) ?? [])

      try {
        const result = await canonicalGroupRequest<{ accepted?: boolean; event_id?: unknown } | undefined>(binding, 'groups.send', exact.params)

        if (result?.accepted === false || (typeof result?.event_id === 'string' && result.event_id !== exact.params.event_id)) {
          throw new Error(labels.unconfirmedSend)
        }
      } catch (error) {
        const outcome = sendOutcome(error)

        // Only the room that sent it gets the text back; a view that moved on keeps the journal entry.
        if (outcome === 'refused' && alive.current) {
          await retireCanonicalGroupSend(binding, exact.params.event_id)

          if (alive.current) {setPending(null)}
        }

        if (alive.current) {setSendHint(outcome === 'refused' ? labels.sendRefused : outcome === 'retryable' ? labels.sendNotYet : labels.sendMaybe)}
        throw error
      }

      await retireCanonicalGroupSend(binding, exact.params.event_id)

      if (alive.current) {setPending(null); setDraft(''); setAttachments([])}
    })
  }

  const act = (action: CanonicalPendingAction, choice?: 'once' | 'deny') =>
    mutate(() => actCanonicalGroup(binding, action, choice))

  // Stop has its own busy state: it must stay available while a Send is in flight.
  const stop = async () => {
    setStopping(true)
    setNotice('')
    setError('')

    try {
      const result = await canonicalGroupRequest<{ cancelled?: number }>(binding, 'groups.stop', { room_id: binding.roomId, cancel_id: crypto.randomUUID() })
      const cancelled = Number(result?.cancelled) || 0

      if (alive.current) {setNotice(cancelled ? labels.stopped.replace('{count}', String(cancelled)) : labels.nothingRunning)}

      if (alive.current) {await refresh()}
    } catch (e) {
      if (alive.current) {setError(e instanceof Error ? e.message : String(e))}
    } finally {
      if (alive.current) {setStopping(false)}
    }
  }

  return <section className="flex h-full min-h-0 flex-col gap-3 p-3">
    <header className="flex items-center gap-2">
      {onBack && <Button onClick={onBack}>{labels.back}</Button>}
      <h2>{state?.room.name || labels.loadingGroup}</h2>
      <Button disabled={stopping || !state?.driver_status} onClick={() => void stop()}>{labels.stop}</Button>
      {state && actions?.({ name: state.room.name, refresh: () => void refresh().catch(e => setReadError(String(e))) })}
    </header>
    {state?.driver_status && <p aria-live="polite">{roomStatus(state.driver_status, labels)}</p>}
    {notice && <p aria-live="polite">{notice}</p>}
    {readError && <div role="alert">{readError}<Button onClick={() => void refresh().catch(e => setReadError(String(e)))}>{labels.refresh}</Button></div>}
    {error && <div role="alert">{error}</div>}
    {state && !state.driver_status && <p>{labels.driverUnavailable}</p>}
    <div className="min-h-0 flex-1 overflow-auto" role="log">
      <CanonicalGroupHistory binding={binding} disabled={!visible} events={events} />
    </div>
    {(state?.driver_status?.pending_actions || []).map(action => <div className="flex items-center gap-2" key={`${action.kind}:${action.task_id}:${action.execution_generation}`}>
      <span>{action.member_id}</span>
      {action.kind === 'discard' && <Button disabled={busy} onClick={() => setDiscard({ ...action })}>{labels.discard}</Button>}
      {action.kind === 'retry' && <Button disabled={busy} onClick={() => void act({ ...action })}>{labels.retry}</Button>}
      {action.kind === 'approval' && <><Button disabled={busy} onClick={() => void act({ ...action }, 'once')}>{labels.allowOnce}</Button><Button disabled={busy} onClick={() => void act({ ...action }, 'deny')}>{labels.deny}</Button></>}
    </div>)}
    {discard && <div aria-label={labels.discardUnknown} role="alertdialog">
      <p>{labels.discardWarning}</p>
      <Button disabled={busy} onClick={() => { const exact = discard; setDiscard(null); void act(exact) }}>{labels.confirmDiscard}</Button>
      <Button onClick={() => setDiscard(null)}>{labels.cancel}</Button>
    </div>}
    {pending && <p role="status">{labels.restoredPendingSend}</p>}
    {sendHint && <p aria-live="polite">{sendHint}</p>}
    <form className="flex gap-2" onSubmit={event => { event.preventDefault(); send() }}>
      <CanonicalGroupAttachments attachments={attachments} binding={binding} disabled={!restored || busy || !!pending} onChange={setAttachments} />
      <textarea aria-label={labels.groupMessage} className="min-w-0 flex-1" disabled={!restored || busy || !!pending} onChange={e => setDraft(e.target.value)} value={draft} />
      <Button disabled={!restored || busy || (!pending && !draft.trim() && !attachments.length) || !state?.driver_status} type="submit">{pending ? labels.retry : labels.send}</Button>
    </form>
  </section>
}
