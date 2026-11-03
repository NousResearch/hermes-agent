import { Button, host, randomUuid } from '@hermes/plugin-sdk'
import { type ClipboardEvent, type DragEvent, useEffect, useRef, useState } from 'react'

import { type CanonicalDriverActivity, CanonicalGroupCycle, CanonicalGroupTyping, canonicalMemberName, useCanonicalRoomLive } from './canonical-group-activity'
import { CanonicalGroupAttachments, type CanonicalGroupUploader } from './canonical-group-attachments'
import { CanonicalGroupDetails, CanonicalPendingCard, holdCanonicalRoomLive, publishCanonicalRoomLive, useStickToBottom } from './canonical-group-details'
import { CanonicalGroupHeader } from './canonical-group-header'
import { type CanonicalGroupEvent, CanonicalGroupHistory } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { forgetCanonicalGroup } from './canonical-group-registry'
import { prepareCanonicalGroupSend, readCanonicalGroupSend, retireCanonicalGroupSend } from './canonical-group-send'
import type { PreparedCanonicalGroupSend } from './canonical-group-send'
import { actCanonicalGroup, canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalPendingAction, CanonicalRoom, CanonicalRoomMember } from './canonical-groups'
import { GroupMentionInput } from './group-chat-parts'
import { RoomBotAvatar } from './group-room-avatar'
import { fill, ROOM_COMPOSER_INPUT_CLASS, RoomComposer, RoomReplyChip } from './group-room-ui'

type RoomEvent = CanonicalGroupEvent
interface Attachment { attachment_id?: string; event_id?: string; kind: string; name: string; mime: string; size?: number }
interface RoomReply { member?: CanonicalRoomMember; name: string; quote: string; thread?: string }
interface RoomState { room: CanonicalRoom; driver_status?: CanonicalDriverActivity & { pending_actions?: CanonicalPendingAction[] } }

const sameGroupPayload = (payload: Record<string, unknown>, text: string, attachments: Attachment[], thread?: string) =>
  payload.text === text && JSON.stringify(payload.attachments ?? []) === JSON.stringify(attachments) &&
  // Preparing mints a thread for a new conversation, so only a reply's thread must match.
  (thread === undefined || payload.thread_id === thread)

/** The durable log only grows: an unchanged length and last seq is the same log, so keep its identity
 *  and the derived history (handoffs, outcomes) is not recomputed on every poll. */
const sameLog = (current: readonly { seq: number }[], next: readonly { seq: number }[]) =>
  current.length === next.length && current.at(-1)?.seq === next.at(-1)?.seq

const canSendNow = (gate: { restored: boolean; busy: boolean; uploading: boolean; pending: boolean; draft: string; attachments: number; driver: boolean }) =>
  gate.restored && !gate.busy && !gate.uploading && gate.driver && (gate.pending || !!gate.draft.trim() || gate.attachments > 0)

export function CanonicalGroupWorkspace({ binding, visible = true, onBack }: {
  binding: CanonicalGroupBinding; visible?: boolean; onBack?: () => void
}) {
  // Remount on identity changes: old polls and pending confirmations never cross rooms.
  return <CanonicalRoomView binding={binding} key={JSON.stringify([binding.connectionId, binding.profile, binding.roomId])} onBack={onBack} visible={visible} />
}

function CanonicalRoomView({ binding: initialBinding, visible, onBack }: {
  binding: CanonicalGroupBinding; visible: boolean; onBack?: () => void
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
  const [restored, setRestored] = useState(false)
  const [pending, setPending] = useState<PreparedCanonicalGroupSend | null>(null)
  // Only a journal entry found on mount is "restored"; this window's own in-flight send also sets `pending`.
  const [restoredPending, setRestoredPending] = useState(false)
  const uploader = useRef<CanonicalGroupUploader>(null)
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)
  const alive = useRef(true)
  const [discard, setDiscard] = useState<CanonicalPendingAction | null>(null)
  const revision = useRef(0)
  // This window's typed draft while it sends the room's one pending intent another window prepared.
  const heldDraft = useRef<{ text: string; attachments: Attachment[]; reply: RoomReply | null } | null>(null)
  const [reply, setReply] = useState<RoomReply | null>(null)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const scroller = useStickToBottom()

  // eslint-disable-next-line no-restricted-syntax -- journal hydration and mounted lifetime, not a reactive store mirror
  useEffect(() => {
    alive.current = true
    let cancelled = false
    void readCanonicalGroupSend(binding).then(entry => {
      if (cancelled) {return}

      if (entry) {
        setPending(entry)
        setRestoredPending(true)
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
      setEvents(current => sameLog(current, log) ? current : log)
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

  const mutate = async (operation: () => Promise<unknown>, settle: () => unknown = refresh) => {
    if (busyRef.current) {return}
    busyRef.current = true
    setBusy(true)
    setError('')

    try { await operation();

 if (alive.current) {await settle()} }
    catch (e) { if (alive.current) {setError(e instanceof Error ? e.message : String(e))} }
    finally {
      busyRef.current = false

      if (alive.current) {setBusy(false)}
    }
  }

  const send = () => {
    if (!restored || busyRef.current || uploading || !state?.driver_status || (!pending && !draft.trim() && !attachments.length)) {return}
    void mutate(async () => {
      const exact = pending ?? await prepareCanonicalGroupSend(binding, { text: draft, attachments, ...(reply?.thread ? { thread_id: reply.thread } : {}) })

      if (!alive.current) {return}

      // Another window won the room's slot: send its intent first, keep this window's words.
      if (!pending && !sameGroupPayload(exact.params.payload, draft, attachments, reply?.thread)) {
        heldDraft.current = { text: draft, attachments, reply }
      }

      setPending(exact)
      setDraft(String(exact.params.payload.text ?? ''))
      setAttachments((exact.params.payload.attachments as Attachment[] | undefined) ?? [])
      await canonicalGroupRequest(exact.binding, 'groups.send', exact.params)
      await retireCanonicalGroupSend(exact.binding, exact.params.event_id)

      if (alive.current) {
        const held = heldDraft.current
        heldDraft.current = null
        setPending(null)
        setReply(held?.reply ?? null)
        setRestoredPending(false)
        setDraft(held?.text ?? '')
        setAttachments(held?.attachments ?? [])
      }
    })
  }

  const act = (action: CanonicalPendingAction, choice?: 'once' | 'deny') =>
    mutate(() => actCanonicalGroup(binding, action, choice))

  const rename = (name: string) =>
    void mutate(() => canonicalGroupRequest(binding, 'groups.rename', { room_id: binding.roomId, event_id: randomUuid(), name }))

  // Rejects so ConfirmDialog stays open with the error; a disbanded room no longer reads back, so leave it.
  const end = async () => {
    await canonicalGroupRequest(binding, 'groups.disband', { room_id: binding.roomId, cancel_id: randomUuid() })

    if (alive.current) {onBack?.()}
    forgetCanonicalGroup(binding)
    host.notify({ kind: 'success', message: labels.disbanded(state?.room.name || binding.roomName || '') })
  }

  const composerLocked = !restored || busy || !!pending
  const driver = state?.driver_status
  const members = state?.room.members ?? []
  const { cycle, statuses, working } = useCanonicalRoomLive(events, driver, members, reply?.thread)
  const mentionMembers = members.map(member => ({ name: member.handle, handle: member.handle, display_name: member.display_name }))
  const memberById = new Map(members.map(member => [member.member_id, member]))
  const avatarFor = (name: string, size: number) => <RoomBotAvatar name={name} size={size} />

  // The Bots list row reads this room's live state while it is open, and forgets it once it closes.
  useEffect(() => holdCanonicalRoomLive(binding), [binding])

  useEffect(() => {
    publishCanonicalRoomLive(binding, {
      working,
      failed: [...statuses.values()].some(status => status.kind === 'failed' || status.kind === 'unavailable'),
      needsYou: !!driver?.pending_actions?.length
    })
  })

  // Hover Reply: continue that message's thread, addressed to the member who wrote it.
  const replyTo = (member: CanonicalRoomMember | undefined, text: string, thread: string | undefined, author: string) => {
    setReply({ member, name: member ? canonicalMemberName(member) : author, quote: text.split('\n').map(line => line.trim()).find(Boolean) ?? '', thread })

    if (member) {
      const tag = `@${member.handle}`
      setDraft(current => current.trimStart().startsWith(tag) ? current : `${tag} ${current}`)
    }
  }

  // Text pastes stay in the composer; only files go through the picker's upload.
  const attachTransfer = (event: ClipboardEvent | DragEvent, data: DataTransfer | null) => {
    const files = [...(data?.files ?? [])]

    if (!files.length) {return}
    event.preventDefault()

    if (!composerLocked) {uploader.current?.upload(files)}
  }

  const canSend = canSendNow({ restored, busy, uploading, pending: !!pending, draft, attachments: attachments.length, driver: !!driver })

  return <section className="flex h-full min-h-0">
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <CanonicalGroupHeader busy={busy} canStop={!!driver} detailsOpen={detailsOpen} onBack={onBack} onDetails={() => setDetailsOpen(open => !open)}
        onDisband={end} onRename={rename} onStop={() => void mutate(() => canonicalGroupRequest(binding, 'groups.stop', { room_id: binding.roomId, cancel_id: randomUuid() }))}
        room={state?.room} statuses={statuses} working={working} />
      {readError && <div className="flex items-center gap-2 px-3 py-1.5 text-xs text-destructive" role="alert">{readError}
        <Button onClick={() => void refresh().catch(e => setReadError(String(e)))} size="xs" variant="ghost">{labels.refresh}</Button></div>}
      {error && <div className="px-3 py-1.5 text-xs text-destructive" role="alert">{error}</div>}
      {state && !driver && <p className="px-3 py-1.5 text-xs text-(--ui-text-tertiary)">{labels.driverUnavailable}</p>}
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4" ref={scroller} role="log">
        {/* Anchored to the composer like a chat thread: a short room sits at the bottom, not the top. */}
        <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col justify-end gap-0.5 py-4">
          <CanonicalGroupHistory avatarFor={avatarFor} binding={binding} disabled={!visible} events={events} members={members} onReply={replyTo} />
          <CanonicalGroupTyping labels={labels} members={members} statuses={statuses} />
          {(driver?.pending_actions || []).map(action => {
            const member = memberById.get(action.member_id)
            const name = member ? canonicalMemberName(member) : action.member_id

            return <CanonicalPendingCard action={action} avatar={avatarFor(member?.profile || action.member_id, 24)} busy={busy}
              key={`${action.kind}:${action.task_id}:${action.execution_generation}`} labels={labels} name={name}
              onAct={act} onDiscard={() => setDiscard({ ...action })} />
          })}
          {discard && <div aria-label={labels.discardUnknown} className="mx-3 grid gap-2 rounded-xl bg-(--chrome-action-hover) px-3 py-2 text-xs" role="alertdialog">
            <p>{labels.discardWarning}</p>
            <div className="flex gap-1.5">
              <Button disabled={busy} onClick={() => { const exact = discard; setDiscard(null); void act(exact) }} size="sm" variant="destructive">{labels.confirmDiscard}</Button>
              <Button onClick={() => setDiscard(null)} size="sm" variant="text">{labels.cancel}</Button>
            </div>
          </div>}
        </div>
      </div>
      {restoredPending && <p className="px-4 pb-1 text-[0.6875rem] text-(--ui-text-tertiary)" role="status">{labels.restoredPendingSend}</p>}
      <div className="px-4">
      <RoomComposer
        above={reply && <RoomReplyChip cancelLabel={labels.cancelReply} label={fill(labels.replyingTo, { name: reply.name })}
          onCancel={() => setReply(null)} quote={reply.quote} />}
        attach={<CanonicalGroupAttachments attachments={attachments} binding={binding} disabled={composerLocked} onChange={setAttachments} onUploadingChange={setUploading} ref={uploader} />}
        canSend={canSend}
        className="mx-auto w-full max-w-3xl"
        onDragOver={event => { if (event.dataTransfer.types.includes('Files')) {event.preventDefault()} }}
        onDrop={event => attachTransfer(event, event.dataTransfer)}
        onPaste={event => attachTransfer(event, event.clipboardData)}
        onSubmit={send}
        sendLabel={pending ? labels.retry : labels.send}
        status={<CanonicalGroupCycle cycle={cycle} driver={driver} labels={labels} members={members} statuses={statuses} />}>
        <GroupMentionInput aria-label={labels.groupMessage} className={ROOM_COMPOSER_INPUT_CLASS} disabled={composerLocked} handleFor={member => member.handle}
          members={mentionMembers} onChange={setDraft} onSubmitDraft={send} placeholder={labels.composerPlaceholder} value={draft} />
      </RoomComposer>
      </div>
    </div>
    {detailsOpen && state?.room && <CanonicalGroupDetails avatarFor={avatarFor} labels={labels} onClose={() => setDetailsOpen(false)} room={state.room} statuses={statuses} />}
  </section>
}
