import { Button, Codicon, ConfirmDialog, Input, Tip } from '@hermes/plugin-sdk'
import { useState } from 'react'

import { CanonicalGroupRoster } from './canonical-group-activity'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { type CanonicalRoom, type CanonicalRoomContext, MAX_ROOM_NOTES_BYTES, roomNotesBytes } from './canonical-groups'
import type { MemberStatus } from './group-room-status'
import { RoomHeader } from './group-room-ui'

interface CanonicalGroupHeaderProps {
  busy: boolean
  canStop: boolean
  context?: CanonicalRoomContext
  deliveryTarget: string
  detailsOpen: boolean
  onBack?: () => void
  onDetails: () => void
  onDisband: () => Promise<void>
  onRename: (name: string) => void
  /** Resolves true once saved; the form keeps the draft otherwise so nothing typed is lost. */
  onSaveContext: (context: CanonicalRoomContext) => Promise<boolean>
  onStop: () => void
  room?: CanonicalRoom
  statuses: Map<string, MemberStatus>
  working: boolean
}

/** Title (opens the details panel), each member's live chip, and quiet icon actions. */
export function CanonicalGroupHeader({ busy, canStop, context, deliveryTarget, detailsOpen, onBack, onDetails, onDisband, onRename, onSaveContext, onStop, room, statuses, working }: CanonicalGroupHeaderProps) {
  const labels = useCanonicalGroupLabels()
  const [renaming, setRenaming] = useState<null | string>(null)
  const [settings, setSettings] = useState<CanonicalRoomContext | null>(null)
  // The draft is frozen while a save is in flight, so closing on success can never drop newer typing.
  const [saving, setSaving] = useState(false)
  const [ending, setEnding] = useState(false)
  const notesBytes = settings ? roomNotesBytes(settings.notes) : 0
  const notesOver = notesBytes > MAX_ROOM_NOTES_BYTES

  const saveSettings = async (draft: CanonicalRoomContext) => {
    setSaving(true)
    const saved = await onSaveContext(draft)
    setSaving(false)

    if (saved) {
      setSettings(null)
    }
  }

  const rename = (name: string) => {
    setRenaming(null)

    if (name) {onRename(name)}
  }

  const action = (label: string, icon: string, onClick: () => void, disabled: boolean) =>
    <Tip label={label}>
      <Button aria-label={label} className="text-(--ui-text-tertiary)" disabled={disabled} onClick={onClick} size="icon-sm" variant="ghost">
        <Codicon name={icon} />
      </Button>
    </Tip>

  return <>
    <RoomHeader backLabel={labels.back} onBack={onBack} onStop={busy || !canStop ? undefined : onStop} onTitle={onDetails}
      stopLabel={labels.stopGroup} title={room?.name || labels.loadingGroup} titleExpanded={detailsOpen} titleLabel={labels.groupDetails} trailing={<>
        {action(labels.rename, 'edit', () => setRenaming(room?.name ?? ''), busy || !room)}
        {action(labels.roomSettings, 'settings-gear', () => setSettings({ purpose: context?.purpose ?? '', notes: context?.notes ?? '' }), busy || !room || !context)}
        {action(labels.disband, 'trash', () => setEnding(true), busy || !room)}
      </>}
      working={working}>
      <CanonicalGroupRoster label={labels.members} labels={labels} members={room?.members} statuses={statuses} />
    </RoomHeader>
    {context?.purpose && <p className="truncate px-3 text-xs text-(--ui-text-secondary)" title={context.purpose}>{context.purpose}</p>}
    {renaming !== null && <form className="flex items-center gap-2 px-3 py-2" onSubmit={event => { event.preventDefault(); rename(renaming.trim()) }}>
      <Input aria-label={labels.renameLabel} autoFocus onChange={event => setRenaming(event.target.value)} value={renaming} />
      <Button disabled={busy || !renaming.trim()} size="sm" type="submit">{labels.save}</Button>
      <Button onClick={() => setRenaming(null)} size="sm" type="button" variant="text">{labels.cancel}</Button>
    </form>}
    {settings !== null && <form aria-label={labels.roomSettings} className="grid gap-2 px-3 py-2" onSubmit={event => { event.preventDefault(); void saveSettings(settings) }}>
      <label className="grid gap-1 text-xs">{labels.purposeLabel}
        <Input disabled={saving} maxLength={500} onChange={event => setSettings({ ...settings, purpose: event.target.value })} placeholder={labels.purposePlaceholder} value={settings.purpose} />
      </label>
      <label className="grid gap-1 text-xs">{labels.notesLabel}
        <textarea className="min-h-24 rounded border border-(--ui-border) bg-transparent p-2 text-xs" disabled={saving} onChange={event => setSettings({ ...settings, notes: event.target.value })}
          placeholder={labels.notesPlaceholder} value={settings.notes} />
        <span className={notesOver ? 'text-(--ui-accent)' : 'text-(--ui-text-tertiary)'} data-testid="room-notes-bytes">{notesBytes} / {MAX_ROOM_NOTES_BYTES}</span>
      </label>
      <p className="text-xs text-(--ui-text-secondary)">{labels.deliveryTarget} <code data-selectable-text="true">{deliveryTarget}</code></p>
      <div className="flex gap-2">
        <Button disabled={busy || notesOver || saving} type="submit">{labels.save}</Button>
        <Button onClick={() => setSettings(null)} type="button">{labels.cancel}</Button>
      </div>
    </form>}
    {ending && <ConfirmDialog busyLabel={labels.disbanding} confirmLabel={labels.disband}
      description={<span>{labels.disbandDescPrefix}<span className="font-medium text-foreground">{room?.name}</span>{labels.disbandDescSuffix(room?.members?.length ?? 0)}</span>}
      destructive doneLabel={labels.disbandDone} onClose={() => setEnding(false)} onConfirm={onDisband} open title={labels.disbandTitle} />}
  </>
}
