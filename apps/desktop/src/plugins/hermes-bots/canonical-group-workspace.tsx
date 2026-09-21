import { Button } from '@hermes/plugin-sdk'
import { useCallback, useEffect, useRef, useState } from 'react'

import { CanonicalGroupAttachments } from './canonical-group-attachments'
import { type CanonicalGroupEvent, CanonicalGroupHistory } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { prepareCanonicalGroupSend, readCanonicalGroupSend, retireCanonicalGroupSend } from './canonical-group-send'
import type { PreparedCanonicalGroupSend } from './canonical-group-send'
import { actCanonicalGroup, canonicalGroupRequest, resolveCanonicalGroupMember } from './canonical-groups'
import type {
  CanonicalGroupBinding,
  CanonicalPendingAction,
  CanonicalRoom,
  CanonicalRoomMember
} from './canonical-groups'
import { checkHostedRoomGateway } from './hosted-room-runtime'
import type { GroupChat } from './types'

type RoomEvent = CanonicalGroupEvent
interface Attachment {
  attachment_id?: string
  event_id?: string
  kind: string
  name: string
  mime: string
  size?: number
}
interface RoomState {
  room: CanonicalRoom
  driver_status?: { pending_actions?: CanonicalPendingAction[] }
}
interface CanonicalContinuity {
  group: string
  room: GroupChat
}

function canonicalMemberStatus(
  member: CanonicalRoomMember,
  labels: ReturnType<typeof useCanonicalGroupLabels>
): string {
  const membership = member.membership?.state
  const availability = member.availability?.state
  const reason = member.availability?.reason

  if (membership === 'retiring' || reason === 'member_retirement_pending') {
    return labels.memberRemovalPending
  }

  if (membership === 'former' || availability === 'retired') {
    return labels.memberFormer
  }

  if (availability === 'ready') {
    return labels.memberReady
  }

  if (reason === 'local_profile_unavailable') {
    return labels.memberLocalUnavailable
  }

  if (reason === 'ambiguous_local_profile') {
    return labels.memberAmbiguous
  }

  if (availability === 'authorization_required') {
    return labels.memberAuthorizationRequired
  }

  return labels.memberUnknown
}




export function CanonicalGroupWorkspace({
  binding,
  continuity,
  visible = true,
  onBack
}: {
  binding: CanonicalGroupBinding
  continuity?: CanonicalContinuity
  visible?: boolean
  onBack?: () => void
}) {
  // Remount on identity changes: old polls and pending confirmations never cross rooms.
  return (
    <CanonicalRoomView
      binding={binding}
      continuity={continuity}
      key={JSON.stringify(binding)}
      onBack={onBack}
      visible={visible}
    />
  )
}

function CanonicalRoomView({
  binding: initialBinding,
  continuity,
  visible,
  onBack
}: {
  binding: CanonicalGroupBinding
  continuity?: CanonicalContinuity
  visible: boolean
  onBack?: () => void
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
  const [checkingGateway, setCheckingGateway] = useState(false)
  const busyRef = useRef(false)
  const alive = useRef(true)
  const current = () => alive.current && binding.isCurrent?.() !== false
  const [discard, setDiscard] = useState<CanonicalPendingAction | null>(null)
  const revision = useRef(0)


  // eslint-disable-next-line no-restricted-syntax -- journal hydration and mounted lifetime, not a reactive store mirror
  useEffect(() => {
    alive.current = true
    let cancelled = false
    void readCanonicalGroupSend(binding)
      .then(entry => {
        if (cancelled || !current()) {
          return
        }

        if (entry) {
          setPending(entry)
          setDraft(String(entry.params.payload.text ?? ''))
          setAttachments((entry.params.payload.attachments as Attachment[] | undefined) ?? [])
        }

        setRestored(true)
      })
      .catch(e => {
        if (!cancelled && current()) {
          setError(e instanceof Error ? e.message : String(e))
        }
      })

    return () => {
      cancelled = true
      alive.current = false
      revision.current++
    }
  }, [binding, current])

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

      if (!next || next <= cursor) {throw new Error(labels.invalidLogCursor)}
      cursor = next
    }

    if (alive.current && version === revision.current) {
      setState(snapshot)
      setEvents(log)
      setReadError('')


    }
  }

  useEffect(() => {
    if (!visible) {
      return
    }

    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const poll = async () => {
      try {
        await refresh()
      } catch (e) {
        if (!cancelled && current()) {
          setReadError(String(e instanceof Error ? e.message : e))
        }
      }

      if (!cancelled && current()) {
        timer = setTimeout(() => void poll(), 2000)
      }
    }

    void poll()

    return () => {
      cancelled = true
      revision.current++
      clearTimeout(timer)
    }
    // The keyed parent freezes the authority binding for this lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, visible])

  const mutate = async (operation: () => Promise<unknown>) => {
    if (busyRef.current || !current()) {
      return
    }

    busyRef.current = true
    setBusy(true)
    setError('')

    try {
      await operation()

      if (current()) {
        await refresh()
      }
    } catch (e) {
      if (current()) {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      busyRef.current = false

      if (current()) {
        setBusy(false)
      }
    }
  }

  const send = () => {
    if (
      !current() ||
      !restored ||
      busyRef.current ||
      !state?.driver_status ||
      (!pending && !draft.trim() && !attachments.length)
    ) {
      return
    }

    void mutate(async () => {
      const exact = pending ?? (await prepareCanonicalGroupSend(binding, { text: draft, attachments }))

      if (!current()) {
        return
      }

      setPending(exact)
      setDraft(String(exact.params.payload.text ?? ''))
      setAttachments((exact.params.payload.attachments as Attachment[] | undefined) ?? [])
      // The journal owns data, never a socket or a deserialized authority.
      await canonicalGroupRequest(binding, 'groups.send', exact.params)

      if (!current()) { return }
      await retireCanonicalGroupSend(binding, exact.params.event_id)

      if (current()) {
        setPending(null)
        setDraft('')
        setAttachments([])
      }
    })
  }

  const act = (action: CanonicalPendingAction, choice?: 'once' | 'deny') =>
    mutate(() => actCanonicalGroup(binding, action, choice))

  const checkGatewayAgain = async () => {
    if (!continuity?.room.hostedStatus?.checkConnectionId || checkingGateway) {
      return
    }

    setCheckingGateway(true)

    try {
      await checkHostedRoomGateway(continuity.group)
    } catch (e) {
      if (current()) {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      if (current()) {
        setCheckingGateway(false)
      }
    }
  }

  return <section className="flex h-full min-h-0 flex-col gap-3 p-3">
    <header className="flex items-center gap-2">
      {onBack && <Button onClick={onBack}>{labels.back}</Button>}
      <h2>{state?.room.name || labels.loadingGroup}</h2>
      <Button disabled={busy || !state?.driver_status} onClick={() => void mutate(() => canonicalGroupRequest(binding, 'groups.stop', { room_id: binding.roomId, cancel_id: crypto.randomUUID() }))}>{labels.stop}</Button>
    </header>
    {readError && <div role="alert">{readError}<Button onClick={() => void refresh().catch(e => setReadError(String(e)))}>{labels.refresh}</Button></div>}
    {error && <div role="alert">{error}</div>}
    {continuity?.room.hostedStatus && <div className="flex flex-wrap items-center gap-2" role="status">
      <span>{continuity.room.hostedStatus.label}</span>
      {continuity.room.continuityIssue && <span>{continuity.room.continuityIssue}</span>}
      {continuity.room.hostedStatus.checkConnectionId && <Button
        aria-busy={checkingGateway}
        disabled={checkingGateway}
        onClick={() => void checkGatewayAgain()}
      >{labels.checkAgain}</Button>}
    </div>}
    {state && !state.driver_status && <p>{labels.driverUnavailable}</p>}
    {!!state?.room.members?.some(member => member.membership || member.availability) &&
      <section aria-label={labels.membersHeading} className="grid gap-1">
      <h3>{labels.membersHeading}</h3>
      {state.room.members.filter(member => member.membership || member.availability).map(member => {
        const former = member.membership?.state === 'former' || member.availability?.state === 'retired'
        const ready = member.availability?.state === 'ready'
        const label = member.display_name || member.profile || member.handle || member.member_id

        return <div className="flex flex-wrap items-center gap-2" key={member.member_id}>
          <strong><bdi>{label}</bdi></strong>
          <span>{canonicalMemberStatus(member, labels)}</span>
          {former
            ? <Button disabled={busy} onClick={() => void mutate(() =>
                resolveCanonicalGroupMember(binding, member.member_id, 'activate'))}>{labels.memberActivate}</Button>
            : ready
              ? <Button disabled={busy} onClick={() => void mutate(() =>
                  resolveCanonicalGroupMember(binding, member.member_id, 'retire'))}>{labels.memberRetire}</Button>
              : <Button disabled={busy} onClick={() => void mutate(() =>
                  resolveCanonicalGroupMember(binding, member.member_id, 'refresh'))}>{labels.checkAgain}</Button>}
        </div>
      })}
    </section>}
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
    <form className="flex gap-2" onSubmit={event => { event.preventDefault(); send() }}>
      <CanonicalGroupAttachments attachments={attachments} binding={binding} disabled={!restored || busy || !!pending} onChange={setAttachments} />
      <textarea aria-label={labels.groupMessage} className="min-w-0 flex-1" disabled={!restored || busy || !!pending} onChange={e => setDraft(e.target.value)} value={draft} />
      <Button disabled={!restored || busy || (!pending && !draft.trim() && !attachments.length) || !state?.driver_status} type="submit">{pending ? labels.retry : labels.send}</Button>
    </form>
  </section>

}
