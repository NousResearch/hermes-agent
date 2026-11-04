import { Button, Codicon, ConfirmDialog, Input, Tip } from '@hermes/plugin-sdk'
import { useState } from 'react'

import { CanonicalGroupRoster } from './canonical-group-activity'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalRoom } from './canonical-groups'
import type { MemberStatus } from './group-room-status'
import { RoomHeader } from './group-room-ui'

interface CanonicalGroupHeaderProps {
  busy: boolean
  canStop: boolean
  /** The room's own connection, so member faces resolve that connection's bot meta. */
  connectionId?: null | string
  detailsOpen: boolean
  onBack?: () => void
  onDetails: () => void
  onDisband: () => Promise<void>
  onRename: (name: string) => void
  onStop: () => void
  room?: CanonicalRoom
  statuses: Map<string, MemberStatus>
  working: boolean
}

/** Title (opens the details panel), each member's live chip, and quiet icon actions. */
export function CanonicalGroupHeader({ busy, canStop, connectionId, detailsOpen, onBack, onDetails, onDisband, onRename, onStop, room, statuses, working }: CanonicalGroupHeaderProps) {
  const labels = useCanonicalGroupLabels()
  const [renaming, setRenaming] = useState<null | string>(null)
  const [ending, setEnding] = useState(false)

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
        {action(labels.disband, 'trash', () => setEnding(true), busy || !room)}
      </>}
      working={working}>
      <CanonicalGroupRoster connectionId={connectionId} label={labels.members} labels={labels} members={room?.members} statuses={statuses} />
    </RoomHeader>
    {renaming !== null && <form className="flex items-center gap-2 px-3 py-2" onSubmit={event => { event.preventDefault(); rename(renaming.trim()) }}>
      <Input aria-label={labels.renameLabel} autoFocus onChange={event => setRenaming(event.target.value)} value={renaming} />
      <Button disabled={busy || !renaming.trim()} size="sm" type="submit">{labels.save}</Button>
      <Button onClick={() => setRenaming(null)} size="sm" type="button" variant="text">{labels.cancel}</Button>
    </form>}
    {ending && <ConfirmDialog busyLabel={labels.disbanding} confirmLabel={labels.disband}
      description={<span>{labels.disbandDescPrefix}<span className="font-medium text-foreground">{room?.name}</span>{labels.disbandDescSuffix(room?.members?.length ?? 0)}</span>}
      destructive doneLabel={labels.disbandDone} onClose={() => setEnding(false)} onConfirm={onDisband} open title={labels.disbandTitle} />}
  </>
}
