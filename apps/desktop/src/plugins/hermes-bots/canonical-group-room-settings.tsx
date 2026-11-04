import {
  Button,
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  Input,
  randomUuid,
  Textarea
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import { CanonicalGroupLeadSetting } from './canonical-group-lead'
import { canonicalGroupRequest, MAX_ROOM_NOTES_BYTES, roomNotesBytes } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalRoomContext, CanonicalRoomMember } from './canonical-groups'

export interface CanonicalGroupRoomSettingsProps {
  binding: CanonicalGroupBinding
  /** `groups.state.context`; null/undefined while loading or on a runtime without room settings. */
  context?: CanonicalRoomContext | null
  disabled?: boolean
  /** The room roster; when given (and the runtime reports routing), the dialog offers lead routing. */
  members?: readonly CanonicalRoomMember[]
  /** Called after the backend accepted a save (the room view refreshes its state). */
  onSaved?: () => unknown
}

export const roomDeliveryTarget = (binding: CanonicalGroupBinding) =>
  `room:${binding.roomId}${binding.profile ? `@${binding.profile}` : ''}`

/** Self-contained room purpose + shared notes: a captioned purpose, a settings button and its dialog,
 *  stacked for the Group details panel; it owns its own `groups.context` save. */
export function CanonicalGroupRoomSettings({
  binding,
  context,
  disabled = false,
  members = [],
  onSaved
}: CanonicalGroupRoomSettingsProps) {
  const labels = useCanonicalGroupLabels()
  const [draft, setDraft] = useState<CanonicalRoomContext | null>(null)
  // The draft is frozen while a save is in flight, so closing on success can never drop newer typing.
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const notesBytes = draft ? roomNotesBytes(draft.notes) : 0
  const notesOver = notesBytes > MAX_ROOM_NOTES_BYTES

  const open = () => {
    setError('')
    setDraft({
      purpose: context?.purpose ?? '',
      notes: context?.notes ?? '',
      ...(context?.routing && { routing: context.routing, lead_member_id: context.lead_member_id ?? '' })
    })
  }

  // A rejected save keeps the draft so the user can shorten it and retry.
  const save = async (submitted: CanonicalRoomContext) => {
    setSaving(true)
    setError('')

    try {
      await canonicalGroupRequest(binding, 'groups.context', {
        room_id: binding.roomId,
        event_id: randomUuid(),
        ...submitted
      })
      setDraft(null)
      await onSaved?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="grid min-w-0 gap-1.5">
      <span className="text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">
        {labels.purposeLabel}
      </span>
      {context?.purpose && (
        <p className="line-clamp-3 text-xs text-foreground" title={context.purpose}>
          {context.purpose}
        </p>
      )}
      <Button className="justify-self-start" disabled={disabled || !context} onClick={open} size="sm" variant="outline">
        {labels.roomSettings}
      </Button>
      {draft && (
        <Dialog onOpenChange={value => !value && !saving && setDraft(null)} open>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>{labels.roomSettings}</DialogTitle>
            </DialogHeader>
            <form
              className="grid gap-3"
              id="room-settings-form"
              onSubmit={event => {
                event.preventDefault()
                void save(draft)
              }}
            >
              <label className="grid gap-1 text-xs">
                {labels.purposeLabel}
                <Input
                  disabled={saving}
                  maxLength={500}
                  onChange={event => setDraft({ ...draft, purpose: event.target.value })}
                  placeholder={labels.purposePlaceholder}
                  value={draft.purpose}
                />
              </label>
              <label className="grid gap-1 text-xs">
                {labels.notesLabel}
                <Textarea
                  className="min-h-24"
                  disabled={saving}
                  onChange={event => setDraft({ ...draft, notes: event.target.value })}
                  placeholder={labels.notesPlaceholder}
                  value={draft.notes}
                />
                <span
                  className={notesOver ? 'text-(--ui-accent)' : 'text-(--ui-text-tertiary)'}
                  data-testid="room-notes-bytes"
                >
                  {notesBytes} / {MAX_ROOM_NOTES_BYTES}
                </span>
              </label>
              {draft.routing && (
                <CanonicalGroupLeadSetting disabled={saving} draft={draft} members={members} onChange={setDraft} />
              )}
              <p className="text-xs text-(--ui-text-secondary)">
                {labels.deliveryTarget} <code data-selectable-text="true">{roomDeliveryTarget(binding)}</code>
              </p>
              {error && (
                <p className="text-xs text-(--ui-accent)" role="alert">
                  {error}
                </p>
              )}
            </form>
            <DialogFooter>
              <Button disabled={saving} onClick={() => setDraft(null)} variant="ghost">
                {labels.cancel}
              </Button>
              <Button disabled={notesOver || saving} form="room-settings-form" type="submit">
                {labels.save}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  )
}
