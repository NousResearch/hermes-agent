import * as sdk from '@hermes/plugin-sdk'
import { Codicon, Tip } from '@hermes/plugin-sdk'
import { Fragment, type ReactNode, useEffect, useMemo, useRef, useState } from 'react'

import {
  type CanonicalGroupAttachment,
  CanonicalGroupAttachments,
  fetchCanonicalGroupAttachment
} from './canonical-group-attachments'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { ASSIGNMENT_EVENT_KIND, CanonicalGroupAssignmentRow } from './canonical-group-lead'
import {
  CanonicalGroupContinuedRow,
  CanonicalGroupPausedRow,
  type CanonicalGroupPausedRowProps,
  CanonicalGroupRoutedRow,
  CONTINUED_EVENT_KIND,
  isBoundedActivity,
  pausedDiscussions,
  ROUTED_EVENT_KIND
} from './canonical-group-routed'
import type { CanonicalGroupBinding, CanonicalRoomMember } from './canonical-groups'
import { type RoomHandoff, roomHandoffs, type RoomLogEvent } from './group-room-status'
import { fill, ROOM_AVATAR_SIZE, RoomActivityRow, RoomMessage, RoomNoticeRow } from './group-room-ui'

const MessageTextContent = typeof sdk === 'undefined' ? undefined : sdk.MessageTextContent
const Streamdown = typeof sdk === 'undefined' ? undefined : sdk.Streamdown

export interface CanonicalGroupEvent {
  seq: number
  event_id?: string
  room_id?: string
  kind: string
  payload: {
    text?: string
    content?: string
    attachments?: CanonicalGroupAttachment[]
    member_id?: string
    error?: string
    reason?: string
    reason_code?: string
    status?: string
    [key: string]: unknown
  }
  actor?: { kind?: string; id?: string; display_name?: string }
  created_at?: number
}

interface CanonicalGroupHistoryProps {
  binding: CanonicalGroupBinding
  events: CanonicalGroupEvent[]
  members?: CanonicalRoomMember[]
  disabled?: boolean
  /** Hover Reply on a member's message: address the composer to that member. */
  onReply?: (member: CanonicalRoomMember | undefined, text: string, thread: string | undefined, author: string) => void
  /** A member's face; the workspace passes the Bot Mode avatar. */
  avatarFor?: (name: string, size: number) => ReactNode
  /** Continue/Stop for a paused discussion; absent `maxContinues` (older runtime) keeps the plain limit notice. */
  pause?: Pick<CanonicalGroupPausedRowProps, 'busy' | 'onContinue' | 'onStop'> & { maxContinues?: number }
}

/** The server stamps every `groups.send` from this Desktop as this principal, so it has no name of its own. */
const isOwnDesktopMessage = (event: CanonicalGroupEvent) =>
  event.kind === 'message.user' &&
  event.actor?.kind === 'user' &&
  event.actor.id === 'desktop' &&
  !event.actor.display_name

function eventAuthor(
  event: CanonicalGroupEvent,
  roster: Map<string, CanonicalRoomMember>,
  isMessage: boolean,
  you: string
) {
  if (isOwnDesktopMessage(event)) {
    return you
  }

  const memberId = isMessage ? (event.actor?.kind === 'member' ? event.actor.id : undefined) : event.payload.member_id

  const member = roster.get(memberId ?? '')

  // The log's label preserves who spoke even if the roster has since changed.
  return (
    (isMessage ? event.actor?.display_name : undefined) ||
    member?.display_name ||
    member?.handle ||
    (isMessage ? event.actor?.id : memberId)
  )
}

// The gateway stamps scheduler posts with a `cron:` / `cron-notice:` actor; a client cannot mint one.
const isScheduledPost = (event: CanonicalGroupEvent) =>
  event.kind === 'message.user' && /^cron(-notice)?:/.test(event.actor?.id ?? '')

function eventTimeMs(timestamp?: number) {
  if (typeof timestamp !== 'number') {
    return undefined
  }
  const at = timestamp * 1000

  return Number.isFinite(new Date(at).getTime()) ? at : undefined
}

function EventBody({ event, notice }: { event: CanonicalGroupEvent; notice?: string }) {
  if (notice) {
    const detail = event.payload.error || event.payload.reason || event.payload.reason_code

    return (
      <p className="min-w-0 text-[0.6875rem] text-(--ui-text-tertiary)" data-selectable-text="true">
        <strong className="font-medium text-(--ui-text-secondary)">{notice}</strong>
        {detail && <>: {detail}</>}
      </p>
    )
  }

  const text = event.payload.text || event.payload.content || ''

  // The shared renderer (layout and large-message safeguards). media={false}: room text
  // may come from another machine, so its MEDIA: paths never resolve against this gateway;
  // room files keep their own download path below.
  return MessageTextContent ? (
    <MessageTextContent media={false} text={text} />
  ) : Streamdown ? (
    <Streamdown>{text}</Streamdown>
  ) : (
    <>{text}</>
  )
}

// Previews download the whole file over RPC; larger or unsized images keep only Download.
const IMAGE_PREVIEW_MAX_BYTES = 5 * 1024 * 1024

// Automatic previews share a small download budget per connection, in request order; a file already
// downloading is joined rather than requested again. Manual Download stays outside it. The budget is
// per connection because a remote whose dial hangs holds its slots for good, and must not stall others.
type Preview = Awaited<ReturnType<typeof fetchCanonicalGroupAttachment>>
interface PreviewQueue {
  slots: number
  waiters: Array<() => void>
}
const previewDownloads = new Map<string, Promise<Preview>>()
const previewQueues = new Map<string, PreviewQueue>()

function releasePreviewSlot(queue: PreviewQueue) {
  const next = queue.waiters.shift()

  if (next) {
    next()
  } else {
    queue.slots++
  }
}

function queuePreview(
  connectionId: string,
  key: string,
  fetch: () => Promise<Preview>,
  onPreview: (preview: Preview) => void
) {
  const queue = previewQueues.get(connectionId) ?? { slots: 2, waiters: [] }
  previewQueues.set(connectionId, queue)
  let cancelled = false
  const deliver = (download: Promise<Preview>) =>
    void download.then(
      preview => {
        if (!cancelled) {
          onPreview(preview)
        }
      },
      () => undefined
    )

  const run = () => {
    let download = previewDownloads.get(key)

    if (download) {
      releasePreviewSlot(queue)
    } else {
      download = fetch().finally(() => {
        previewDownloads.delete(key)
        releasePreviewSlot(queue)
      })
      previewDownloads.set(key, download)
    }

    deliver(download)
  }

  const running = previewDownloads.get(key)

  if (running) {
    deliver(running)
  } else if (queue.slots > 0) {
    queue.slots--
    run()
  } else {
    queue.waiters.push(run)
  }

  // A preview that leaves before its turn gives up its place without sending a request.
  return () => {
    cancelled = true
    const at = queue.waiters.indexOf(run)

    if (at >= 0) {
      queue.waiters.splice(at, 1)
    }
  }
}

function ImagePreview({
  binding,
  eventId,
  attachment,
  disabled
}: {
  binding: CanonicalGroupBinding
  eventId: string
  attachment: CanonicalGroupAttachment
  disabled: boolean
}) {
  const [src, setSrc] = useState('')
  const [near, setNear] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  const { attachment_id: attachmentId, size } = attachment
  const { connectionId, profile, roomId } = binding
  const previewable = !!attachmentId && size !== undefined && size <= IMAGE_PREVIEW_MAX_BYTES

  // History loads the whole room: fetch only near the viewport. Leaving before its turn gives up the queued place;
  // a started download runs to completion and is shared if the preview returns.
  useEffect(() => {
    const el = box.current

    if (src || disabled || !el) {
      return
    }
    const observer = new IntersectionObserver(entries => setNear(entries[entries.length - 1].isIntersecting), {
      rootMargin: '200px'
    })
    observer.observe(el)

    return () => observer.disconnect()
  }, [src, disabled])

  useEffect(() => {
    if (disabled || !near || !attachmentId || !previewable) {
      return
    }
    let url = ''
    const key = JSON.stringify([connectionId, profile, roomId, eventId, attachmentId])

    // A failed preview leaves the Download button as the way to the file.
    const cancel = queuePreview(
      connectionId,
      key,
      () => fetchCanonicalGroupAttachment({ connectionId, profile, roomId }, eventId, attachmentId),
      ({ bytes, mime }) => {
        url = URL.createObjectURL(new Blob([bytes], { type: mime }))
        setSrc(url)
      }
    )

    // Dropping `near` makes a re-shown room wait on the observer again, not refetch every preview seen so far at once.
    return () => {
      cancel()
      URL.revokeObjectURL(url)
      setSrc('')
      setNear(false)
    }
  }, [connectionId, profile, roomId, eventId, attachmentId, disabled, near, previewable])

  return previewable ? (
    <div ref={box}>
      {src && <img alt={attachment.name} className="max-h-64 max-w-full rounded object-contain" src={src} />}
    </div>
  ) : null
}

export function CanonicalGroupHistory({
  avatarFor,
  binding,
  events,
  members = [],
  disabled = false,
  onReply,
  pause
}: CanonicalGroupHistoryProps) {
  const labels = useCanonicalGroupLabels()
  const roster = new Map(members.map(member => [member.member_id, member]))
  const maxContinues = pause?.maxContinues
  const pauses = useMemo(
    () => (typeof maxContinues === 'number' ? pausedDiscussions(events, maxContinues) : undefined),
    [events, maxContinues]
  )
  const handoffsAfter = new Map<number, RoomHandoff[]>()

  for (const handoff of roomHandoffs(events as RoomLogEvent[], members)) {
    const after = handoffsAfter.get(handoff.afterSeq)

    if (after) {
      after.push(handoff)
    } else {
      handoffsAfter.set(handoff.afterSeq, [handoff])
    }
  }

  const nameOf = (id: string) => {
    const member = roster.get(id)

    return member?.display_name || member?.handle || id
  }

  const notices = new Map([
    ['turn.failed', labels.turnFailed],
    ['turn.cancelled', labels.turnCancelled],
    ['turn.deferred', labels.turnDeferred],
    ['member.unavailable', labels.memberUnavailable],
    ['room.stop_requested', labels.roomStopped],
    ['room.disbanded', labels.roomDisbanded],
    ['authority.lost', labels.authorityLost]
  ])

  return (
    <>
      {events.map(event => (
        <Fragment key={event.seq}>
          {event.kind === ASSIGNMENT_EVENT_KIND ? (
            <CanonicalGroupAssignmentRow event={event} members={members} />
          ) : event.kind === ROUTED_EVENT_KIND ? (
            <CanonicalGroupRoutedRow event={event} members={members} />
          ) : event.kind === CONTINUED_EVENT_KIND ? (
            <CanonicalGroupContinuedRow />
          ) : pause && pauses && isBoundedActivity(event) ? (
            <CanonicalGroupPausedRow
              busy={pause.busy}
              event={event}
              onContinue={pause.onContinue}
              onStop={pause.onStop}
              paused={pauses.get(event.seq) ?? { count: 0, actionable: false }}
            />
          ) : (
            <HistoryRow
              avatarFor={avatarFor}
              binding={binding}
              disabled={disabled}
              event={event}
              notices={notices}
              onReply={onReply}
              roster={roster}
            />
          )}
          {handoffsAfter.get(event.seq)?.map(handoff => (
            <RoomActivityRow key={`handoff:${event.seq}:${handoff.to}`}>
              {fill(labels.handoff, { from: nameOf(handoff.from), to: nameOf(handoff.to) })}
            </RoomActivityRow>
          ))}
        </Fragment>
      ))}
    </>
  )
}

interface HistoryRowProps extends Pick<CanonicalGroupHistoryProps, 'avatarFor' | 'binding' | 'onReply'> {
  disabled: boolean
  event: CanonicalGroupEvent
  notices: Map<string, string>
  roster: Map<string, CanonicalRoomMember>
}

function HistoryRow({ avatarFor, binding, disabled, event, notices, onReply, roster }: HistoryRowProps) {
  const labels = useCanonicalGroupLabels()
  const notice =
    event.kind === 'room.activity' && event.payload.status === 'bounded'
      ? labels.workLimitReached
      : notices.get(event.kind)
  const isMessage = MESSAGE_KINDS.has(event.kind)

  if (!notice && !isMessage) {
    return null
  }
  const author = eventAuthor(event, roster, isMessage, labels.you)
  const memberId = isMessage ? (event.actor?.kind === 'member' ? event.actor.id : undefined) : event.payload.member_id
  const member = roster.get(memberId ?? '')

  if (notice) {
    return (
      <RoomNoticeRow avatar={member && avatarFor?.(member.profile || member.handle, 16)} kind={event.kind}>
        {author && <span className="shrink-0 font-medium text-(--ui-text-secondary)">{author}</span>}
        <EventBody event={event} notice={notice} />
      </RoomNoticeRow>
    )
  }

  return (
    <MessageRow
      author={author}
      avatarFor={avatarFor}
      binding={binding}
      disabled={disabled}
      event={event}
      member={member}
      memberId={memberId}
      onReply={onReply}
    />
  )
}

const MESSAGE_KINDS = new Set(['message', 'message.user', 'message.member'])

function MessageRow({
  author,
  avatarFor,
  binding,
  disabled,
  event,
  member,
  memberId,
  onReply
}: Omit<HistoryRowProps, 'notices' | 'roster'> & {
  author?: string
  member?: CanonicalRoomMember
  memberId?: string
}) {
  const labels = useCanonicalGroupLabels()
  const isUser = event.actor?.kind !== 'member'
  const text = event.payload.text || event.payload.content || ''
  const canReply = !!onReply && !!text && (!!member || typeof event.payload.thread_id === 'string')
  const face = isUser
    ? undefined
    : avatarFor?.(member?.profile || member?.handle || memberId || author || 'bot', ROOM_AVATAR_SIZE)

  const badge = isScheduledPost(event) ? (
    <span className="shrink-0 rounded bg-(--ui-surface-2) px-1 text-[0.625rem] text-(--ui-text-secondary)">
      {labels.scheduledPost}
    </span>
  ) : undefined

  return (
    <RoomMessage
      actions={
        canReply ? (
          <ReplyAction
            author={author ?? ''}
            event={event}
            member={isUser ? undefined : member}
            onReply={onReply!}
            text={text}
          />
        ) : undefined
      }
      author={author}
      avatar={face}
      badge={badge}
      isUser={isUser}
      timestampMs={eventTimeMs(event.created_at)}
    >
      <EventBody event={event} />
      <EventFiles binding={binding} disabled={disabled} event={event} />
    </RoomMessage>
  )
}

function EventFiles({
  binding,
  disabled,
  event
}: {
  binding: CanonicalGroupBinding
  disabled: boolean
  event: CanonicalGroupEvent
}) {
  const previewEventId = event.room_id === binding.roomId ? event.event_id : undefined

  return (
    <>
      {previewEventId &&
        event.payload.attachments
          ?.filter(a => a.kind === 'image')
          .map(a => (
            <ImagePreview
              attachment={a}
              binding={binding}
              disabled={disabled}
              eventId={previewEventId}
              key={a.attachment_id ?? a.name}
            />
          ))}
      {!!event.payload.attachments?.length && (
        <CanonicalGroupAttachments
          attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
          binding={binding}
          disabled={disabled || !event.event_id || event.room_id !== binding.roomId}
          readOnly
        />
      )}
    </>
  )
}

/** Hover Reply: continue this message's thread, addressed to its author. */
function ReplyAction({
  author,
  event,
  member,
  onReply,
  text
}: {
  author: string
  event: CanonicalGroupEvent
  member?: CanonicalRoomMember
  text: string
  onReply: NonNullable<CanonicalGroupHistoryProps['onReply']>
}) {
  const labels = useCanonicalGroupLabels()
  const label = fill(labels.replyTo, { name: author })
  const thread = typeof event.payload.thread_id === 'string' ? event.payload.thread_id : undefined

  return (
    <Tip label={label}>
      <sdk.Button
        aria-label={label}
        className="text-(--ui-text-tertiary)"
        onClick={() => onReply(member, text, thread, author)}
        size="icon-xs"
        variant="ghost"
      >
        <Codicon name="reply" />
      </sdk.Button>
    </Tip>
  )
}
