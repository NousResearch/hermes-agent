import { type CanonicalGroupAttachment, CanonicalGroupAttachments } from './canonical-group-attachments'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalGroupBinding, CanonicalRoomMember } from './canonical-groups'

export interface CanonicalGroupEvent {
  seq: number
  event_id?: string
  room_id?: string
  kind: string
  payload: { text?: string; content?: string; attachments?: CanonicalGroupAttachment[] }
  actor?: { kind?: string; id?: string; member_id?: string; profile?: string; display_name?: string }
}

function speaker(event: CanonicalGroupEvent, labels: ReturnType<typeof useCanonicalGroupLabels>, members: readonly CanonicalRoomMember[]): string | undefined {
  const actor = event.actor

  // Event-owned metadata, never a same-named Bot from the foreground roster.
  if (typeof actor?.display_name === 'string' && actor.display_name.trim()) {return actor.display_name}

  if (actor?.kind === 'user' || (!actor && event.kind === 'message.user')) {return labels.userSpeaker}

  if (actor?.kind === 'gateway' || actor?.kind === 'system') {return labels.systemSpeaker}

  if (actor?.kind === 'bot' || actor?.kind === 'member' || (!actor?.kind && actor?.member_id)) {
    const member = members.find(value => value.member_id === (actor.member_id || actor.id))

    if (typeof member?.display_name === 'string' && member.display_name.trim()) {return member.display_name}

    if (typeof member?.handle === 'string' && member.handle.trim()) {return `@${member.handle}`}
  }

  return [actor?.id, actor?.member_id].find(value => typeof value === 'string' && value.trim())
}

export function CanonicalGroupHistory({ binding, events, disabled = false, members = [] }: {
  binding: CanonicalGroupBinding; events: CanonicalGroupEvent[]; disabled?: boolean; members?: readonly CanonicalRoomMember[]
}) {
  const labels = useCanonicalGroupLabels()

  return <>{events.map(event => {
    const text = event.payload.text || event.payload.content

    // Suppress only known bookkeeping without user-visible content. Unknown kinds stay visible.
    if ((event.kind === 'turn.settled' || event.kind === 'room.activity') && !text && !event.payload.attachments?.length) {return null}
    const name = speaker(event, labels, event.room_id === binding.roomId ? members : [])

    return <div className="whitespace-pre-wrap wrap-anywhere py-2 text-sm text-(--ui-text-primary)" key={event.seq}>
      {name && <strong className="font-medium text-(--ui-text-secondary)">{name}: </strong>}
      {text || event.kind}
      {!!event.payload.attachments?.length && <CanonicalGroupAttachments
        attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
        binding={binding} disabled={disabled || !event.event_id || event.room_id !== binding.roomId} readOnly />}
    </div>
  })}</>
}
