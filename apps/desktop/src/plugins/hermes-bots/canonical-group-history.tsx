import { type CanonicalGroupAttachment, CanonicalGroupAttachments } from './canonical-group-attachments'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalGroupBinding } from './canonical-groups'

export interface CanonicalGroupEvent {
  seq: number
  event_id?: string
  room_id?: string
  kind: string
  payload: {
    text?: string
    content?: string
    description?: string
    state?: string
    action?: string
    author?: { kind?: string; member_id?: string; name?: string }
    attachments?: CanonicalGroupAttachment[]
  }
  actor?: { id?: string; member_id?: string }
}

export function CanonicalGroupHistory({ binding, events, disabled = false }: {
  binding: CanonicalGroupBinding
  events: CanonicalGroupEvent[]
  disabled?: boolean
}) {
  const labels = useCanonicalGroupLabels()

  return <>{events.map(event => {
    const importedAuthor = event.kind === 'history.imported' ? event.payload.author : undefined
    const author = importedAuthor?.name || event.actor?.member_id

    const body = event.kind === 'history.held'
      ? `${labels.historyHeld} ${event.payload.description || event.payload.state || labels.memberUnknown}`
      : event.payload.text || event.payload.content || event.kind

    return <div className="whitespace-pre-wrap py-2" key={event.event_id || event.seq}>
      {author && <strong><bdi>{author}</bdi>: </strong>}
      {body}
      {!!event.payload.attachments?.length && <CanonicalGroupAttachments
        attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
        binding={binding} disabled={disabled || !event.event_id || event.room_id !== binding.roomId} readOnly />}
    </div>
  })}</>
}
