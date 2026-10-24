import * as sdk from '@hermes/plugin-sdk'

import { type CanonicalGroupAttachment, CanonicalGroupAttachments } from './canonical-group-attachments'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalGroupBinding, CanonicalRoomMember } from './canonical-groups'

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
  }
  actor?: { kind?: string; id?: string; display_name?: string }
  created_at?: number
}

interface CanonicalGroupHistoryProps {
  binding: CanonicalGroupBinding
  events: CanonicalGroupEvent[]
  members?: CanonicalRoomMember[]
  disabled?: boolean
}

function eventAuthor(event: CanonicalGroupEvent, roster: Map<string, CanonicalRoomMember>, isMessage: boolean) {
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

function EventTime({ timestamp }: { timestamp?: number }) {
  if (typeof timestamp !== 'number') {
    return null
  }
  const at = new Date(timestamp * 1000)

  if (!Number.isFinite(at.getTime())) {
    return null
  }

  return (
    <time className="text-[0.625rem] text-(--ui-text-quaternary)" dateTime={at.toISOString()}>
      {sdk.relativeTime(at.getTime())}
    </time>
  )
}

function EventBody({ event, notice }: { event: CanonicalGroupEvent; notice?: string }) {
  if (notice) {
    const detail = event.payload.error || event.payload.reason || event.payload.reason_code

    return (
      <p className="text-xs text-(--ui-text-secondary)" data-selectable-text="true">
        <strong>{notice}</strong>
        {detail && <>: {detail}</>}
      </p>
    )
  }

  const text = event.payload.text || event.payload.content || ''

  return (
    <div
      className="min-w-0 text-xs text-(--ui-text-secondary) [&_p]:mb-1 [&_p:last-child]:mb-0 [&_ul]:mb-1 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:mb-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_pre]:whitespace-pre-wrap [&_pre]:wrap-anywhere"
      data-selectable-text="true"
    >
      {/* The shared renderer (layout and large-message safeguards). media={false}: room text
        may come from another machine, so its MEDIA: paths never resolve against this gateway;
        room files keep their own download path below. */}
      {MessageTextContent ? (
        <MessageTextContent media={false} text={text} />
      ) : Streamdown ? (
        <Streamdown>{text}</Streamdown>
      ) : (
        text
      )}
    </div>
  )
}

export function CanonicalGroupHistory({ binding, events, members = [], disabled = false }: CanonicalGroupHistoryProps) {
  const labels = useCanonicalGroupLabels()
  const roster = new Map(members.map(member => [member.member_id, member]))

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
      {events.map(event => {
        const notice =
          event.kind === 'room.activity' && event.payload.status === 'bounded'
            ? labels.workLimitReached
            : notices.get(event.kind)

        const isMessage = event.kind === 'message' || event.kind === 'message.user' || event.kind === 'message.member'

        if (!notice && !isMessage) {
          return null
        }
        const author = eventAuthor(event, roster, isMessage)

        return (
          <div className="grid grid-cols-[minmax(0,1fr)] gap-1 px-2 py-1.5" key={event.seq}>
            <div className="flex items-center gap-2">
              {author && <strong className="text-[0.7rem] font-semibold text-(--ui-accent)">{author}</strong>}
              <EventTime timestamp={event.created_at} />
            </div>
            <EventBody event={event} notice={notice} />
            {!!event.payload.attachments?.length && (
              <CanonicalGroupAttachments
                attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
                binding={binding}
                disabled={disabled || !event.event_id || event.room_id !== binding.roomId}
                readOnly
              />
            )}
          </div>
        )
      })}
    </>
  )
}
