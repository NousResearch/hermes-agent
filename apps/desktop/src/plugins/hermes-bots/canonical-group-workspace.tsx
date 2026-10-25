import { Button, host, randomUuid } from '@hermes/plugin-sdk'
import { type ClipboardEvent, type DragEvent, useEffect, useRef, useState } from 'react'

import { type CanonicalDriverStatus, CanonicalGroupCycle, CanonicalGroupNotices, CanonicalGroupTyping, canonicalMemberName, useCanonicalRoomLive } from './canonical-group-activity'
import { CanonicalGroupAttachments, type CanonicalGroupUploader } from './canonical-group-attachments'
import { offersGroupMethod, useGroupMethods } from './canonical-group-capabilities'
import { CanonicalGroupDetails, CanonicalPendingCard, holdCanonicalRoomLive, publishCanonicalRoomLive, useStickToBottom } from './canonical-group-details'
import { CanonicalGroupHeader } from './canonical-group-header'
import { type CanonicalGroupEvent, CanonicalGroupHistory } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { markRoomLead } from './canonical-group-lead'
import { forgetCanonicalGroup } from './canonical-group-registry'
import { CanonicalGroupRoomSettings } from './canonical-group-room-settings'
import { attemptCanonicalGroupSend, prepareCanonicalGroupSend, readCanonicalGroupSend, replaceCanonicalGroupSend, retireCanonicalGroupSend, withdrawCanonicalGroupSend } from './canonical-group-send'
import type { PreparedCanonicalGroupSend } from './canonical-group-send'
import { actCanonicalGroup, canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalPendingAction, CanonicalRoom, CanonicalRoomContext, CanonicalRoomMember } from './canonical-groups'
import { GroupMentionInput } from './group-chat-parts'
import { RoomBotAvatar } from './group-room-avatar'
import { fill, ROOM_COMPOSER_INPUT_CLASS, RoomComposer, RoomReplyChip } from './group-room-ui'

type RoomEvent = CanonicalGroupEvent
interface Attachment { attachment_id?: string; event_id?: string; kind: string; name: string; mime: string; size?: number }
interface RoomReply { member?: CanonicalRoomMember; name: string; quote: string; thread?: string }
interface Words { text: string; attachments: Attachment[] }
interface SendRun { stopped: boolean; posted: boolean; settled: Promise<boolean> }
interface RoomState {
  room: CanonicalRoom & { authority_epoch?: number; latest_seq?: number }
  context?: CanonicalRoomContext | null
  driver_status?: CanonicalDriverStatus<CanonicalPendingAction>
}

const sameGroupPayload = (payload: Record<string, unknown>, text: string, attachments: Attachment[], thread?: string) =>
  payload.text === text && JSON.stringify(payload.attachments ?? []) === JSON.stringify(attachments) &&
  // Preparing mints a thread for a new conversation, so only a reply's thread must match.
  (thread === undefined || payload.thread_id === thread)

/** The durable log only grows: an unchanged length and last seq is the same log, so keep its identity
 *  and the derived history (handoffs, outcomes) is not recomputed on every poll. */
const sameLog = (current: readonly { seq: number }[], next: readonly { seq: number }[]) =>
  current.length === next.length && current.at(-1)?.seq === next.at(-1)?.seq

const canSendNow = (gate: { restored: boolean; busy: boolean; stopping: boolean; uploading: boolean; pending: boolean; draft: string; attachments: number; driver: boolean }) =>
  gate.restored && !gate.busy && !gate.stopping && !gate.uploading && gate.driver && (gate.pending || !!gate.draft.trim() || gate.attachments > 0)

// Both groups.send dispatchers mark a refusal raised before the user event committed (and only
// that) `admitted: false`: proof that this request posted nothing. A reason or code alone is not:
// the same `invalid_params` or 4111 can follow a committed event.
const admittedNothing = (error: unknown) =>
  (error as { data?: { admitted?: unknown } } | null)?.data?.admitted === false

/** A Send is delivered only by a receipt for this exact intent. */
function confirmSend(result: unknown, eventId: string, unconfirmed: string) {
  const receipt = result as { accepted?: unknown; client_event_id?: unknown } | null

  if (!receipt || typeof receipt !== 'object' || Array.isArray(receipt) ||
    (Object.hasOwn(receipt, 'accepted') && receipt.accepted !== true) || receipt.client_event_id !== eventId) {
    throw new Error(unconfirmed)
  }
}

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
  const methods = useGroupMethods(binding, visible)
  const [state, setState] = useState<RoomState | null>(null)
  const [events, setEvents] = useState<RoomEvent[]>([])
  const [error, setError] = useState('')
  const [readError, setReadError] = useState('')
  const [draft, setDraft] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [uploading, setUploading] = useState(false)
  const [restored, setRestored] = useState(false)
  const [journalError, setJournalError] = useState('')
  const [journalAttempt, setJournalAttempt] = useState(0)
  const [pending, setPending] = useState<PreparedCanonicalGroupSend | null>(null)
  // Only a journal entry found on mount is "restored"; this window's own in-flight send also sets `pending`.
  const [restoredPending, setRestoredPending] = useState(false)
  const uploader = useRef<CanonicalGroupUploader>(null)
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)
  const alive = useRef(true)
  const [discard, setDiscard] = useState<CanonicalPendingAction | null>(null)
  const [stopping, setStopping] = useState(false)
  const stopPending = useRef(false)
  // One cancellation intent survives uncertain replies until new work starts or Stop is confirmed.
  const stopIntent = useRef<string | null>(null)
  const [notice, setNotice] = useState('')
  const [sendHint, setSendHint] = useState('')
  const revision = useRef(0)
  // This window's typed draft while it sends the room's one pending intent another window prepared.
  const heldDraft = useRef<{ text: string; attachments: Attachment[]; reply: RoomReply | null } | null>(null)
  const [reply, setReply] = useState<RoomReply | null>(null)
  // A handed-back journal entry keeps its thread even without a visible reply chip.
  const draftThread = useRef<string | undefined>(undefined)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const scroller = useStickToBottom()
  // An intent handed back for editing (nothing of it admitted): the next Send replaces it.
  const editing = useRef<string | null>(null)
  // A delivered intent whose journal cleanup failed: retired before the next Send may prepare.
  const delivered = useRef<string | null>(null)
  // The Send in progress, which a Stop pressed now must cover; `settled` says whether it may have posted.
  const sending = useRef<SendRun | null>(null)
  // The log is append-only within one authority epoch: each poll reads only what is new.
  const seen = useRef<{ epoch?: number; seq: number }>({ seq: 0 })

  // eslint-disable-next-line no-restricted-syntax -- mounted lifetime, not a reactive store mirror
  useEffect(() => {
    alive.current = true

    return () => { alive.current = false; revision.current++ }
  }, [])

  // Send stays locked until the journal is read: a pending intent must never be sent twice. A
  // failed read only reads again; it never writes, so a journal that recovers keeps its intent.
  // eslint-disable-next-line no-restricted-syntax -- journal hydration, not a reactive store mirror
  useEffect(() => {
    let cancelled = false
    setJournalError('')
    void readCanonicalGroupSend(binding).then(entry => {
      if (cancelled) {return}

      if (entry) {
        setPending(entry)
        setRestoredPending(true)
        draftThread.current = entry.params.payload.thread_id as string | undefined
        setDraft(String(entry.params.payload.text ?? ''))
        setAttachments((entry.params.payload.attachments as Attachment[] | undefined) ?? [])
      }

      setRestored(true)
    }).catch(e => { if (!cancelled) {setJournalError(e instanceof Error ? e.message : String(e))} })

    return () => { cancelled = true }
  }, [binding, journalAttempt])

  const refresh = async () => {
    const version = ++revision.current
    const snapshot = await canonicalGroupRequest<RoomState>(binding, 'groups.state', { room_id: binding.roomId })
    const { authority_epoch: epoch, latest_seq: latest } = snapshot.room ?? {}
    // A new authority epoch (or a log behind what was read) starts the log again.
    const fresh = epoch !== seen.current.epoch || (typeof latest === 'number' && latest < seen.current.seq)
    const since = fresh ? 0 : seen.current.seq
    const log: RoomEvent[] = []
    let cursor = since

    for (;;) {
      const page = await canonicalGroupRequest<{ events: RoomEvent[]; has_more?: boolean; next_seq?: number }>(binding, 'groups.log', { room_id: binding.roomId, since_seq: cursor, limit: 100 })
      log.push(...page.events)

      if (!page.has_more) {break}
      const next = page.events.at(-1)?.seq

      if (!next || next <= cursor) {throw new Error(labels.invalidLogCursor)}
      cursor = next
    }

    if (alive.current && version === revision.current) {
      seen.current = { epoch, seq: log.at(-1)?.seq ?? cursor }
      setState(snapshot)
      setEvents(current => {
        const next = fresh ? log : [...current, ...log.filter(event => event.seq > since)]

        return !fresh && sameLog(current, next) ? current : next
      })
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

  // The room's intent is finished with: this window's own words come back, else nothing.
  const releaseComposer = () => {
    const next = heldDraft.current
    heldDraft.current = null
    draftThread.current = undefined
    setPending(null)
    setReply(next?.reply ?? null)
    setRestoredPending(false)
    setDraft(next?.text ?? '')
    setAttachments(next?.attachments ?? [])
  }

  // Another window finished with this intent: resending it could post it twice, and so could
  // offering its words as a new message.
  const finishedElsewhere = () => {
    releaseComposer()
    setSendHint(labels.sendSettledElsewhere)
  }

  // A request of the intent provably posted nothing (`left`: requests that still may have, null when
  // another window finished with it). With none left its words come back for editing while the
  // journal keeps them, until a Send replaces them; this window's own held words wait until then.
  const settleUnsent = (exact: PreparedCanonicalGroupSend, words: Words, left: null | number | undefined, hint: string) => {
    if (left === null) {finishedElsewhere()}
    else if (left === 0) {
      editing.current = exact.params.event_id
      draftThread.current = exact.params.payload.thread_id as string | undefined
      setReply(current => current?.thread === draftThread.current ? current : null)
      setPending(null)
      setRestoredPending(false)
      setDraft(words.text)
      setAttachments(words.attachments)
      setSendHint(hint)
    } else {setSendHint(labels.sendMaybe)}
  }

  const sendIntent = async (run: SendRun) => {
    if (!pending && delivered.current) {
      await retireCanonicalGroupSend(binding, delivered.current)
      delivered.current = null
    }

    const typed = { text: draft, attachments, thread_id: reply?.thread ?? draftThread.current }
    const replacing = editing.current
    const exact = pending ?? (replacing ? await replaceCanonicalGroupSend(binding, replacing, typed) : await prepareCanonicalGroupSend(binding, typed))
    editing.current = null

    if (!alive.current) {return}

    if (!exact) {
      // Another window finished with the intent being edited: these words may already be posted.
      setSendHint(labels.sendSettledElsewhere)

      return
    }

    // Another window won the room's slot: send its intent first, keep this window's words.
    if (!pending && !sameGroupPayload(exact.params.payload, draft, attachments, reply?.thread)) {
      heldDraft.current = { ...typed, reply }
    }

    const words = { text: String(exact.params.payload.text ?? ''), attachments: (exact.params.payload.attachments as Attachment[] | undefined) ?? [] }
    setPending(exact)
    setDraft(words.text)
    setAttachments(words.attachments)

    if (await attemptCanonicalGroupSend(exact.binding, exact.params.event_id) === null) {
      if (alive.current) {finishedElsewhere()}

      return
    }

    if (run.stopped) {
      // Stop came first: this request never leaves.
      const left = await withdrawCanonicalGroupSend(exact.binding, exact.params.event_id)

      if (alive.current) {settleUnsent(exact, words, left, labels.sendStopped)}

      return
    }

    run.posted = true

    try {
      confirmSend(await canonicalGroupRequest(exact.binding, 'groups.send', exact.params), exact.params.event_id, labels.unconfirmedSend)
    } catch (error) {
      // Without the gateway's proof the request may have posted: Retry resends this exact intent,
      // which the gateway recognizes, so it is never posted twice.
      run.posted = !admittedNothing(error)
      const left = run.posted ? undefined : await withdrawCanonicalGroupSend(exact.binding, exact.params.event_id).catch(() => undefined)

      if (alive.current) {settleUnsent(exact, words, left, labels.sendRefused)}

      throw error
    }

    // A valid receipt is authoritative: a failed cleanup only defers it to this window's next Send.
    await retireCanonicalGroupSend(exact.binding, exact.params.event_id).catch(error => {
      delivered.current = exact.params.event_id
      console.warn('[canonical-group-send]', error)
    })

    if (alive.current) {releaseComposer()}
  }

  const send = () => {
    // A Stop still fencing behind the last Send would cancel this one too: it waits until Stop is done.
    if (!restored || busyRef.current || stopPending.current || uploading || !state?.driver_status || (!pending && !draft.trim() && !attachments.length)) {return}
    stopIntent.current = null
    setSendHint('')
    setNotice('')
    let settle!: (mayHavePosted: boolean) => void
    const run: SendRun = { stopped: false, posted: false, settled: new Promise<boolean>(resolve => { settle = resolve }) }
    sending.current = run
    void mutate(() => sendIntent(run).finally(() => {
      if (sending.current === run) {sending.current = null}
      settle(run.posted)
    }))
  }

  const requestStop = async (fresh = false) => {
    if (fresh) {stopIntent.current = null}
    const cancelId = stopIntent.current ??= randomUuid()
    const result = await canonicalGroupRequest<{ cancelled?: unknown } | undefined>(binding, 'groups.stop', { room_id: binding.roomId, cancel_id: cancelId })
    const cancelled = result?.cancelled

    if (typeof cancelled !== 'number' || !Number.isSafeInteger(cancelled) || cancelled < 0) {throw new Error(labels.pendingActionUnconfirmed)}

    if (stopIntent.current === cancelId) {stopIntent.current = null}

    return cancelled
  }

  // Stop has its own busy state: it stays available while a Send is in flight, and covers it. A
  // Send not yet sent is held back; one on the wire may post after this fence, so once it settles
  // (unless it provably posted nothing) a fresh cancel_id fences again behind it.
  const stop = async () => {
    if (stopPending.current) {return}
    stopPending.current = true
    const covered = sending.current

    if (covered) {covered.stopped = true}
    setStopping(true)
    setNotice('')
    setError('')

    try {
      const first = requestStop()
      first.catch(() => undefined)
      const cancelled = covered && await covered.settled ? await first.catch(() => 0) + await requestStop(true) : await first

      if (alive.current) {
        setNotice(cancelled ? labels.stopped : labels.nothingRunning)
        await refresh()
      }
    } catch (e) {
      if (alive.current) {setError(e instanceof Error ? e.message : String(e))}
    } finally {
      stopPending.current = false

      if (alive.current) {setStopping(false)}
    }
  }

  const act = (action: CanonicalPendingAction, choice?: 'once' | 'deny') =>
    mutate(() => {
      if (action.kind === 'retry') {stopIntent.current = null}

      return actCanonicalGroup(binding, action, choice)
    })

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

    if (!composerLocked && offersGroupMethod(methods, 'groups.attachment.upload')) {uploader.current?.upload(files)}
  }

  const canSend = canSendNow({ restored, busy, stopping, uploading, pending: !!pending, draft, attachments: attachments.length, driver: !!driver })

  return <section className="flex h-full min-h-0">
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <CanonicalGroupHeader busy={busy} canDisband={offersGroupMethod(methods, 'groups.disband')} canRename={offersGroupMethod(methods, 'groups.rename')}
        canStop={!!driver && offersGroupMethod(methods, 'groups.stop')} detailsOpen={detailsOpen} driver={driver} onBack={onBack} onDetails={() => setDetailsOpen(open => !open)}
        onDisband={end} onRename={rename} onStop={() => void stop()}
        room={state ? { ...state.room, members: markRoomLead(state.room.members, state.context) } : undefined} statuses={statuses} stopping={stopping} working={working || !!sending.current || stopping} />
      <CanonicalGroupNotices error={error} journalError={journalError} labels={labels}
        notice={notice} onJournalReload={() => setJournalAttempt(value => value + 1)} onRefresh={() => void refresh().catch(e => setReadError(String(e)))}
        readError={readError} unavailable={Boolean(state && !driver)} />
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4" ref={scroller} role="log">
        {/* Anchored to the composer like a chat thread: a short room sits at the bottom, not the top. */}
        <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col justify-end gap-0.5 py-4">
          <CanonicalGroupHistory avatarFor={avatarFor} binding={binding} disabled={!visible} downloads={offersGroupMethod(methods, 'groups.attachment.download')} events={events} members={members} onReply={replyTo} />
          <CanonicalGroupTyping labels={labels} members={members} statuses={statuses} />
          {(driver?.pending_actions || []).map(action => {
            const member = memberById.get(action.member_id)
            const name = member ? canonicalMemberName(member) : action.member_id

            return <CanonicalPendingCard action={action} avatar={avatarFor(member?.profile || action.member_id, 24)} busy={busy}
              key={`${action.kind}:${action.task_id}:${action.execution_generation}`} labels={labels} methods={methods} name={name}
              onAct={act} onDiscard={() => setDiscard({ ...action })}
              onRefresh={() => void refresh().catch(e => setReadError(String(e)))} />
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
      {sendHint && <p aria-live="polite" className="px-4 pb-1 text-xs text-(--ui-text-secondary)">{sendHint}</p>}
      <div className="px-4">
      <RoomComposer
        above={reply && <RoomReplyChip cancelLabel={labels.cancelReply} label={fill(labels.replyingTo, { name: reply.name })}
          onCancel={() => { draftThread.current = undefined; setReply(null) }} quote={reply.quote} />}
        attach={<CanonicalGroupAttachments attachments={attachments} binding={binding} disabled={composerLocked} downloads={offersGroupMethod(methods, 'groups.attachment.download')} onChange={setAttachments} onUploadingChange={setUploading} ref={uploader} uploads={offersGroupMethod(methods, 'groups.attachment.upload')} />}
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
    {detailsOpen && state?.room && <CanonicalGroupDetails avatarFor={avatarFor} labels={labels} onClose={() => setDetailsOpen(false)} room={state.room} statuses={statuses}>
      <CanonicalGroupRoomSettings binding={binding} context={state.context} disabled={busy} members={state.room.members} onSaved={refresh} />
    </CanonicalGroupDetails>}
  </section>
}
