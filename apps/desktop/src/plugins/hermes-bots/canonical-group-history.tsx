import * as sdk from '@hermes/plugin-sdk'

import { type CanonicalGroupAttachment, CanonicalGroupAttachments } from './canonical-group-attachments'
import type { CanonicalGroupBinding } from './canonical-groups'

const Streamdown = typeof sdk === 'undefined' ? undefined : sdk.Streamdown

export interface CanonicalGroupEvent {
  seq: number
  event_id?: string
  room_id?: string
  kind: string
  payload: { text?: string; content?: string; attachments?: CanonicalGroupAttachment[] }
  actor?: { member_id?: string }
}

export function CanonicalGroupHistory({ binding, events, disabled = false }: { binding: CanonicalGroupBinding; events: CanonicalGroupEvent[]; disabled?: boolean }) {
  return <>{events.filter(event => event.kind === 'message' || event.kind === 'message.user' || event.kind === 'message.member').map(event => <div className="grid grid-cols-[minmax(0,1fr)] gap-1 px-2 py-1.5" key={event.seq}>
    {event.actor?.member_id && <strong className="text-[0.7rem] font-semibold text-(--ui-accent)">{event.actor.member_id}</strong>}
    <div className="min-w-0 text-xs text-(--ui-text-secondary) [&_p]:mb-1 [&_p:last-child]:mb-0 [&_ul]:mb-1 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:mb-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_pre]:whitespace-pre-wrap [&_pre]:wrap-anywhere" data-selectable-text="true">
      {Streamdown ? <Streamdown>{event.payload.text || event.payload.content || ''}</Streamdown> : event.payload.text || event.payload.content}
    </div>
    {!!event.payload.attachments?.length && <CanonicalGroupAttachments
      attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
      binding={binding} disabled={disabled || !event.event_id || event.room_id !== binding.roomId} readOnly />}
  </div>)}</>
}
