import { Button, Codicon, composerInputSurface, PRIMARY_ICON_BTN, Tip } from '@hermes/plugin-sdk'
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

import { CanonicalGroupAttachments } from './canonical-group-attachments'
import { CanonicalGroupComposerInput } from './canonical-group-composer'
import { CanonicalGroupHeader } from './canonical-group-header'
import { type CanonicalGroupEvent, CanonicalGroupHistory } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { CanonicalGroupPendingActions } from './canonical-group-pending-actions'
import { updateCanonicalGroupName } from './canonical-group-registry'
import { attemptCanonicalGroupSend, claimCanonicalGroupSend, listCanonicalGroupSends, prepareCanonicalGroupSend, readCanonicalGroupSend, retireCanonicalGroupSend, settleCanonicalGroupSend } from './canonical-group-send'
import type { PreparedCanonicalGroupSend, RecoverableCanonicalGroupSend } from './canonical-group-send'
import { actCanonicalGroup, canonicalGroupRequest, isPendingFileAction } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalPendingAction, CanonicalRoomMember } from './canonical-groups'

type RoomEvent = CanonicalGroupEvent
interface Attachment { attachment_id?: string; event_id?: string; kind: string; name: string; mime: string; size?: number }
interface DriverStatus {
  running?: boolean
  working?: boolean
  blocked?: boolean
  counts?: Record<string, number>
  pending_actions?: CanonicalPendingAction[]
}
interface RoomState { room: { name: string; authority_epoch?: number; members?: CanonicalRoomMember[] }; driver_status?: DriverStatus }
type Labels = ReturnType<typeof useCanonicalGroupLabels>

// These reasons prove only this attempt had no effect, not any earlier attempt.
const TERMINAL_SEND_REFUSALS = new Set(['invalid_params', 'permission_denied', 'unknown_execution', 'stale_generation'])

function sendOutcome(error: unknown): 'refused' | 'retryable' | 'unknown' {
  const failure = error as { code?: unknown; data?: { reason?: unknown } } | null

  if (failure?.code !== 4001) {return 'unknown'}

  return TERMINAL_SEND_REFUSALS.has(String(failure.data?.reason)) ? 'refused' : 'retryable'
}

function onlyPendingFiles(status: DriverStatus): boolean {
  const actions = status.pending_actions ?? []

  return actions.length > 0 && actions.every(isPendingFileAction) && !(status.counts?.unknown || status.counts?.stopping)
}

function needsAttention(status: DriverStatus): boolean {
  return Boolean(status.blocked && !onlyPendingFiles(status) ||
    status.pending_actions?.some(action => action.kind !== 'stopping' && !isPendingFileAction(action)))
}

/** Live work comes only from the driver; unresolved members are listed beside it, never instead. */
function roomStatus(status: DriverStatus, labels: Labels) {
  const actions = status.pending_actions || []
  const approvals = actions.filter(action => action.kind === 'approval').length
  const stopping = (status.counts?.stopping ?? 0) > 0 || actions.some(action => action.kind === 'stopping')
  const attention = actions.filter(action => action.kind !== 'approval' && action.kind !== 'stopping' && !isPendingFileAction(action)).length
  const parts = [stopping ? labels.statusStopping : status.working || actions.some(isPendingFileAction) ? labels.statusWorking : status.running === false ? labels.statusStopped : labels.statusIdle]

  if (status.blocked && !onlyPendingFiles(status)) {parts.push(labels.statusBlocked)}

  if (approvals) {parts.push(labels.statusApprovals.replace('{count}', String(approvals)))}

  if (attention) {parts.push(labels.statusAttention.replace('{count}', String(attention)))}

  return parts.join(' · ')
}

/** Room controls rendered by the owner of the binding (rename, disband). */
export type CanonicalRoomActions = (room: { name: string; refresh: () => void; latestFileSeq: number; visible: boolean }) => ReactNode

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
  const [uploading, setUploading] = useState(false)
  const uploadingRef = useRef(false)
  const [restored, setRestored] = useState(false)
  const [pending, setPending] = useState<PreparedCanonicalGroupSend | null>(null)
  const [recoveries, setRecoveries] = useState<RecoverableCanonicalGroupSend[]>([])
  const inputRevision = useRef(0)
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)
  const [stopping, setStopping] = useState(false)
  const stopPending = useRef(false)
  const stopIntent = useRef<string | null>(null)
  const [notice, setNotice] = useState('')
  const [sendHint, setSendHint] = useState('')
  const alive = useRef(true)
  const transcript = useRef<HTMLDivElement>(null)
  const following = useRef(true)
  const revision = useRef(0)
  // The log is append-only within one authority epoch: read only what is new.
  const seen = useRef<{ epoch?: number; seq: number }>({ seq: 0 })

  // eslint-disable-next-line no-restricted-syntax -- journal hydration and mounted lifetime, not a reactive store mirror
  useEffect(() => {
    alive.current = true
    let cancelled = false
    void Promise.all([readCanonicalGroupSend(binding), listCanonicalGroupSends(binding)]).then(([entry, recoverable]) => {
      if (cancelled) {return}

      if (entry) {
        setPending(entry)
        setDraft(String(entry.params.payload.text ?? ''))
        setAttachments((entry.params.payload.attachments as Attachment[] | undefined) ?? [])
      }

      setRecoveries(recoverable)
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
      updateCanonicalGroupName(binding, snapshot.room.name)
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

  // Follow new messages only while the viewer remains at the end of the chat.
  useLayoutEffect(() => {
    if (visible && following.current && transcript.current) {transcript.current.scrollTop = transcript.current.scrollHeight}
  }, [events, visible])

  const mutate = async (operation: () => Promise<unknown>, propagate = false) => {
    if (busyRef.current) {
      if (propagate) {throw new Error(labels.actionInFlight)}

      return
    }

    busyRef.current = true
    setBusy(true)
    setError('')

    try { await operation();

 if (alive.current) {await refresh()} }
    catch (e) {
      if (propagate) {throw e}

      if (alive.current) {setError(e instanceof Error ? e.message : String(e))}
    }
    finally {
      busyRef.current = false

      if (alive.current) {setBusy(false)}
    }
  }

  const send = () => {
    if (!restored || busyRef.current || uploadingRef.current || !state?.driver_status || (!pending && !draft.trim() && !attachments.length)) {return}
    setSendHint('')
    const editing = inputRevision.current
    void mutate(async () => {
      const exact = pending ?? await prepareCanonicalGroupSend(binding, { text: draft, attachments })

      if (!alive.current) {return}
      if (inputRevision.current === editing) {
        setPending(exact)
        setDraft(String(exact.params.payload.text ?? ''))
        setAttachments((exact.params.payload.attachments as Attachment[] | undefined) ?? [])
      }

      const freshAttempt = await attemptCanonicalGroupSend(binding, exact)

      if (!alive.current) {return}
      try {
        const result = await canonicalGroupRequest<{ accepted?: unknown; client_event_id?: unknown } | undefined>(binding, 'groups.send', exact.params)

        if (!result || typeof result !== 'object' || Array.isArray(result) ||
            (Object.hasOwn(result, 'accepted') && result.accepted !== true) || result.client_event_id !== exact.params.event_id) {
          throw new Error(labels.unconfirmedSend)
        }
      } catch (error) {
        const outcome = freshAttempt ? sendOutcome(error) : 'unknown'

        // Only the room that sent it gets the text back; a view that moved on keeps the journal entry.
        if (outcome === 'refused' && alive.current) {
          await retireCanonicalGroupSend(binding, exact.params.event_id, exact)

          if (alive.current) {setPending(current => current?.params.event_id === exact.params.event_id ? null : current)}
        }

        if (alive.current) {setSendHint(outcome === 'refused' ? labels.sendRefused : outcome === 'retryable' ? labels.sendNotYet : labels.sendMaybe)}
        throw error
      }

      await settleCanonicalGroupSend(binding, exact)

      if (alive.current) {
        setPending(current => current?.params.event_id === exact.params.event_id ? null : current)
        if (inputRevision.current === editing) {setDraft(''); setAttachments([])}
        try {
          const recoverable = await listCanonicalGroupSends(binding)
          if (alive.current) {setRecoveries(recoverable)}
        } catch (error) {console.warn('Accepted group Send recovery journal could not be read', error)}
      }
    })
  }

  const restore = (recovery: RecoverableCanonicalGroupSend) => {
    if (pending || draft.trim() || attachments.length || uploadingRef.current) {return}
    const editing = inputRevision.current
    void mutate(async () => {
      const exact = await claimCanonicalGroupSend(binding, recovery)

      if (!alive.current) {return}
      if (inputRevision.current === editing) {
        setPending(exact)
        setDraft(String(exact.params.payload.text ?? ''))
        setAttachments((exact.params.payload.attachments as Attachment[] | undefined) ?? [])
      }
      const recoverable = await listCanonicalGroupSends(binding)
      if (alive.current) {setRecoveries(recoverable)}
    })
  }

  const act = (action: CanonicalPendingAction, choice?: 'once' | 'deny') =>
    mutate(() => actCanonicalGroup(binding, action, choice), true)

  // Stop has its own busy state: it must stay available while a Send is in flight.
  const stop = async () => {
    if (stopPending.current) {return}
    stopPending.current = true
    stopIntent.current ??= crypto.randomUUID()
    setStopping(true)
    setNotice('')
    setError('')

    try {
      const result = await canonicalGroupRequest<{ cancelled?: number }>(binding, 'groups.stop', { room_id: binding.roomId, cancel_id: stopIntent.current })
      const cancelled = result?.cancelled
      if (typeof cancelled !== 'number' || !Number.isSafeInteger(cancelled) || cancelled < 0) {throw new Error(labels.pendingActionUnconfirmed)}
      stopIntent.current = null

      if (alive.current) {setNotice(cancelled ? labels.stopped.replace('{count}', String(cancelled)) : labels.nothingRunning)}

      if (alive.current) {await refresh()}
    } catch (e) {
      if (alive.current) {setError(e instanceof Error ? e.message : String(e))}
    } finally {
      stopPending.current = false
      if (alive.current) {setStopping(false)}
    }
  }

  const members = state?.room.members ?? []
  const name = state?.room.name || labels.loadingGroup
  const pendingActions = state?.driver_status?.pending_actions ?? []
  const inputDisabled = !restored || busy || !!pending || !state?.driver_status

  // `running` reports gateway-worker health, including while this chat is idle.
  const canStop = Boolean(pending || busy || stopping || state?.driver_status && (state.driver_status.working ||
    ['queued', 'running', 'stopping'].some(status => (state.driver_status?.counts?.[status] ?? 0) > 0) ||
    pendingActions.some(action => action.kind !== 'output_retry')))

  return <section className="flex h-full min-h-0 flex-col" data-slot="canonical-group-chat">
    <CanonicalGroupHeader attention={state?.driver_status && needsAttention(state.driver_status)}
      members={members} name={name} onBack={onBack} status={state?.driver_status && roomStatus(state.driver_status, labels)} visible={visible} working={state?.driver_status?.working || pendingActions.some(isPendingFileAction)}>
      {visible && state && actions?.({ name: state.room.name, refresh: () => void refresh().catch(e => setReadError(String(e))),
        latestFileSeq: events.reduce((latest, event) => event.payload.attachments?.length ? Math.max(latest, event.seq) : latest, 0), visible })}
    </CanonicalGroupHeader>
    <div aria-label={labels.conversationHistory} className="min-h-0 flex-1 overflow-y-auto overscroll-y-contain px-2"
      onScroll={event => { const node = event.currentTarget; following.current = node.scrollHeight - node.scrollTop - node.clientHeight < 48 }} ref={transcript} role="log">
      <div className="mx-auto w-full max-w-3xl pb-4">
        <CanonicalGroupHistory binding={binding} disabled={!visible} events={events} members={members} />
        {state && !events.length && <div className="grid gap-1 px-3 py-10 text-center">
          <p className="text-sm text-(--ui-text-secondary)">{labels.emptyHistory}</p>
          <p className="text-xs text-(--ui-text-quaternary)">{labels.emptyHistoryHint}</p>
        </div>}
      </div>
    </div>
    <div className="mx-auto w-full max-w-3xl shrink-0 px-4 pb-4">
      <div className="max-h-[min(40vh,24rem)] overflow-y-auto">
        {visible && <CanonicalGroupPendingActions actions={pendingActions} busy={busy} members={members} onAction={act}
          onDiscard={action => act(action)} onRefresh={refresh} />}
      </div>
      <div className="grid gap-2 pb-2 text-xs text-(--ui-text-secondary)">
        {notice && <p aria-live="polite">{notice}</p>}
        {readError && <div className="grid gap-1" role="alert"><div className="flex items-center gap-2"><span>{labels.driverUnavailable}</span><Button onClick={() => void refresh().catch(e => setReadError(String(e)))} size="inline" variant="text">{labels.refresh}</Button></div>
          <details className="text-(--ui-text-quaternary)"><summary className="cursor-pointer">{labels.setupDetails}</summary><p className="mt-1 whitespace-pre-wrap break-words">{readError}</p></details>
        </div>}
        {error && <div className="grid gap-1 text-destructive" role="alert"><p>{labels.pendingActionUnconfirmed}</p>
          <details className="text-(--ui-text-quaternary)"><summary className="cursor-pointer">{labels.setupDetails}</summary><p className="mt-1 whitespace-pre-wrap break-words">{error}</p></details>
        </div>}
        {state && !state.driver_status && <p>{labels.driverUnavailable}</p>}
        {pending && <p role="status">{labels.restoredPendingSend}</p>}
        {sendHint && <p aria-live="polite">{sendHint}</p>}
        {recoveries.filter(recovery => recovery.entry.params.event_id !== pending?.params.event_id).map(recovery =>
          <div className="flex items-center gap-2" key={recovery.storageKey}>
            <span className="min-w-0 flex-1 truncate">{String(recovery.entry.params.payload.text || labels.groupMessage)}</span>
            <Button disabled={!visible || busy || uploading || !!pending || !!draft.trim() || !!attachments.length} onClick={() => restore(recovery)} size="inline" variant="text">{labels.restorePendingSend}</Button>
          </div>)}
      </div>
      <form className={`${composerInputSurface} rounded-2xl border border-(--ui-stroke-tertiary) p-2`} data-slot="composer-root"
        onSubmit={event => { event.preventDefault(); send() }}>
        <CanonicalGroupComposerInput disabled={inputDisabled} members={members} name={name} onChange={value => {inputRevision.current++; setDraft(value)}} onSubmit={send} value={draft} />
        <div className="mt-1 flex items-end gap-2">
          <CanonicalGroupAttachments attachments={attachments} binding={binding} disabled={inputDisabled} onChange={value => {inputRevision.current++; setAttachments(value)}}
            onUploadingChange={uploading => { uploadingRef.current = uploading;

 if (alive.current) {setUploading(uploading)} }} />
          <div className="ml-auto flex shrink-0 items-center gap-2">
            {canStop && <Button disabled={stopping} loading={stopping} onClick={() => void stop()} size="xs" type="button" variant="ghost"><Codicon name="debug-stop" />{labels.stop}</Button>}
            <Tip label={pending ? labels.retry : labels.send}><Button aria-label={pending ? labels.retry : labels.send} className={PRIMARY_ICON_BTN}
              disabled={!restored || busy || uploading || (!pending && !draft.trim() && !attachments.length) || !state?.driver_status} loading={busy}
              size="icon-xs" type="submit" variant="ghost"><Codicon name={pending ? 'refresh' : 'arrow-up'} /></Button></Tip>
          </div>
        </div>
      </form>
    </div>
  </section>
}
