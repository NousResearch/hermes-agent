import { Button } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { actCanonicalGroup, canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalPendingAction } from './canonical-groups'

interface RoomEvent { seq: number; kind: string; payload: { text?: string; content?: string }; actor?: { member_id?: string } }
interface RoomState { room: { name: string }; driver_status?: { pending_actions?: CanonicalPendingAction[] } }

export function CanonicalGroupWorkspace({ binding, visible = true, onBack }: {
  binding: CanonicalGroupBinding; visible?: boolean; onBack?: () => void
}) {
  // Remount on identity changes: old polls and pending confirmations never cross rooms.
  return <CanonicalRoomView binding={binding} key={JSON.stringify(binding)} onBack={onBack} visible={visible} />
}

function CanonicalRoomView({ binding, visible, onBack }: {
  binding: CanonicalGroupBinding; visible: boolean; onBack?: () => void
}) {
  const [state, setState] = useState<RoomState | null>(null)
  const [events, setEvents] = useState<RoomEvent[]>([])
  const [error, setError] = useState('')
  const [readError, setReadError] = useState('')
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)
  const alive = useRef(true)
  const [discard, setDiscard] = useState<CanonicalPendingAction | null>(null)
  const revision = useRef(0)
  const prepared = useRef<{ text: string; eventId: string } | null>(null)

  const refresh = async () => {
    const version = ++revision.current
    const snapshot = await canonicalGroupRequest<RoomState>(binding, 'groups.state', { room_id: binding.roomId })
    const log: RoomEvent[] = []
    let cursor = 0

    for (;;) {
      const page = await canonicalGroupRequest<{ events: RoomEvent[]; has_more?: boolean; next_seq?: number }>(binding, 'groups.log', { room_id: binding.roomId, since_seq: cursor, limit: 100 })
      log.push(...page.events)

      if (!page.has_more) {break}
      const next = page.events.at(-1)?.seq

      if (!next || next <= cursor) {throw new Error('Invalid room log cursor')}
      cursor = next
    }

    if (alive.current && version === revision.current) {
      setState(snapshot)
      setEvents(log)
      setReadError('')
    }
  }

  // eslint-disable-next-line no-restricted-syntax -- mounted lifetime guard, not a reactive store mirror
  useEffect(() => {
    alive.current = true

    if (!visible) {return}
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const poll = async () => {
      try { await refresh() } catch (e) { if (!cancelled) {setReadError(String(e instanceof Error ? e.message : e))} }

      if (!cancelled) {timer = setTimeout(() => void poll(), 2000)}
    }

    void poll()

    return () => { cancelled = true; alive.current = false; revision.current++; clearTimeout(timer) }
    // The keyed parent freezes the authority binding for this lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible])

  const mutate = async (operation: () => Promise<unknown>) => {
    if (busyRef.current) {return}
    busyRef.current = true
    setBusy(true)
    setError('')

    try { await operation(); await refresh() }
    catch (e) { if (alive.current) {setError(e instanceof Error ? e.message : String(e))} }
    finally { busyRef.current = false;

 if (alive.current) {setBusy(false)} }
  }

  const act = (action: CanonicalPendingAction, choice?: 'once' | 'deny') =>
    mutate(() => actCanonicalGroup(binding, action, choice))

  return <section className="flex h-full min-h-0 flex-col gap-3 p-3">
    <header className="flex items-center gap-2">
      {onBack && <Button onClick={onBack}>Back</Button>}
      <h2>{state?.room.name || 'Loading group…'}</h2>
      <Button disabled={busy || !state?.driver_status} onClick={() => void mutate(() => canonicalGroupRequest(binding, 'groups.stop', { room_id: binding.roomId, cancel_id: crypto.randomUUID() }))}>Stop</Button>
    </header>
    {readError && <div role="alert">{readError}<Button onClick={() => void refresh().catch(e => setReadError(String(e)))}>Refresh</Button></div>}
    {error && <div role="alert">{error}</div>}
    {state && !state.driver_status && <p>Group driver unavailable. Update or reconnect the owning gateway.</p>}
    <div className="min-h-0 flex-1 overflow-auto" role="log">
      {events.map(event => <div className="whitespace-pre-wrap py-2" key={event.seq}>{event.actor?.member_id && <strong>{event.actor.member_id}: </strong>}{event.payload.text || event.payload.content || event.kind}</div>)}
    </div>
    {(state?.driver_status?.pending_actions || []).map(action => <div className="flex items-center gap-2" key={`${action.kind}:${action.task_id}:${action.execution_generation}`}>
      <span>{action.member_id}</span>
      {action.kind === 'discard' && <Button disabled={busy} onClick={() => setDiscard({ ...action })}>Discard</Button>}
      {action.kind === 'retry' && <Button disabled={busy} onClick={() => void act({ ...action })}>Retry</Button>}
      {action.kind === 'approval' && <><Button disabled={busy} onClick={() => void act({ ...action }, 'once')}>Allow once</Button><Button disabled={busy} onClick={() => void act({ ...action }, 'deny')}>Deny</Button></>}
    </div>)}
    {discard && <div aria-label="Discard unknown work" role="alertdialog">
      <p>Side effects may already have occurred. Discarding does not undo them.</p>
      <Button disabled={busy} onClick={() => { const exact = discard; setDiscard(null); void act(exact) }}>Confirm discard</Button>
      <Button onClick={() => setDiscard(null)}>Cancel</Button>
    </div>}
    <form className="flex gap-2" onSubmit={event => {
      event.preventDefault()

      if (!draft.trim() || busyRef.current || !state?.driver_status) {return}
      const text = draft

      if (prepared.current && prepared.current.text !== text) {
        setError('The previous send is unconfirmed. Retry its original text before sending another message.')

        return
      }

      prepared.current ||= { text, eventId: crypto.randomUUID() }
      const eventId = prepared.current.eventId
      void mutate(async () => {
        await canonicalGroupRequest(binding, 'groups.send', { room_id: binding.roomId, event_id: eventId, payload: { text, thread_id: eventId } })
        prepared.current = null

        if (alive.current) {setDraft(current => current === text ? '' : current)}
      })
    }}>
      <textarea aria-label="Group message" className="min-w-0 flex-1" onChange={e => setDraft(e.target.value)} value={draft} />
      <Button disabled={busy || !draft.trim() || !state?.driver_status} type="submit">Send</Button>
    </form>
  </section>
}
