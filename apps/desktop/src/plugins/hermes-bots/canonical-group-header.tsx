import { Button, Codicon, ConfirmDialog, Input, Tip } from '@hermes/plugin-sdk'
import { useState } from 'react'

import { type CanonicalDriverStatus, CanonicalGroupAttention, CanonicalGroupRoster } from './canonical-group-activity'
import type { GroupMethods } from './canonical-group-capabilities'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { CanonicalPeerJoinRows } from './canonical-group-peer-join'
import { peerMemberStates } from './canonical-group-peers'
import type { CanonicalGroupBinding, CanonicalRoom } from './canonical-groups'
import type { MemberStatus } from './group-room-status'
import { RoomHeader } from './group-room-ui'

interface CanonicalGroupHeaderProps {
  binding: CanonicalGroupBinding
  methods: GroupMethods
  onChanged: () => void
  onReplaced?: (key: string) => void
  busy: boolean
  canDisband: boolean
  canRename: boolean
  canStop: boolean
  detailsOpen: boolean
  driver?: CanonicalDriverStatus
  onBack?: () => void
  onDetails: () => void
  onDisband: () => Promise<void>
  onRename: (name: string) => void
  onStop: () => void
  room?: CanonicalRoom
  stopping: boolean
  statuses: Map<string, MemberStatus>
  working: boolean
}

/** Title (opens the details panel), each member's live chip, and quiet icon actions. */
export function CanonicalGroupHeader({
  binding,
  methods,
  onChanged,
  onReplaced,
  busy,
  canDisband,
  canRename,
  canStop,
  detailsOpen,
  driver,
  onBack,
  onDetails,
  onDisband,
  onRename,
  onStop,
  room,
  statuses,
  stopping,
  working
}: CanonicalGroupHeaderProps) {
  const labels = useCanonicalGroupLabels()
  const badges: Record<string, string> = {
    missing: labels.peerNotJoinedBadge,
    unavailable: labels.peerOfflineBadge,
    needs_reauthorization: labels.peerRejoinBadge
  }
  const peerBadges = new Map(
    peerMemberStates(room?.members, driver?.peer_routes).map(({ member, status }) => [member.member_id, badges[status]])
  )
  const [renaming, setRenaming] = useState<null | string>(null)
  const [ending, setEnding] = useState(false)

  const rename = (name: string) => {
    setRenaming(null)

    if (name) {
      onRename(name)
    }
  }

  const action = (label: string, icon: string, onClick: () => void, disabled: boolean) => (
    <Tip label={label}>
      <Button
        aria-label={label}
        className="text-(--ui-text-tertiary)"
        disabled={disabled}
        onClick={onClick}
        size="icon-sm"
        variant="ghost"
      >
        <Codicon name={icon} />
      </Button>
    </Tip>
  )

  return (
    <>
      <RoomHeader
        backLabel={labels.back}
        onBack={onBack}
        onStop={canStop ? onStop : undefined}
        onTitle={onDetails}
        stopDisabled={stopping}
        stopLabel={labels.stopGroup}
        title={room?.name || labels.loadingGroup}
        titleExpanded={detailsOpen}
        titleLabel={labels.groupDetails}
        trailing={
          <>
            {canRename && action(labels.rename, 'edit', () => setRenaming(room?.name ?? ''), busy || !room)}
            {canDisband && action(labels.disband, 'trash', () => setEnding(true), busy || !room)}
          </>
        }
        working={working}
      >
        <CanonicalGroupRoster
          label={labels.members}
          labels={labels}
          members={room?.members}
          peerBadges={peerBadges}
          statuses={statuses}
        />
        <CanonicalGroupAttention driver={driver} labels={labels} />
      </RoomHeader>
      <CanonicalPeerJoinRows
        binding={binding}
        methods={methods}
        onChanged={onChanged}
        onReplaced={onReplaced}
        state={room ? { room, driver_status: driver } : null}
      />
      {renaming !== null && (
        <form
          className="flex items-center gap-2 px-3 py-2"
          onSubmit={event => {
            event.preventDefault()
            rename(renaming.trim())
          }}
        >
          <div className="mx-auto flex w-full max-w-[calc(48rem+0.5rem)] items-center gap-2 px-1">
            <Input
              aria-label={labels.renameLabel}
              autoFocus
              className="min-w-0 flex-1"
              onChange={event => setRenaming(event.target.value)}
              value={renaming}
            />
            <Button disabled={busy || !renaming.trim()} size="sm" type="submit">
              {labels.save}
            </Button>
            <Button onClick={() => setRenaming(null)} size="sm" type="button" variant="text">
              {labels.cancel}
            </Button>
          </div>
        </form>
      )}
      {ending && (
        <ConfirmDialog
          busyLabel={labels.disbanding}
          confirmLabel={labels.disband}
          description={
            <span>
              {labels.disbandDescPrefix}
              <span className="font-medium text-foreground">{room?.name}</span>
              {labels.disbandDescSuffix(room?.members?.length ?? 0)}
            </span>
          }
          destructive
          doneLabel={labels.disbandDone}
          onClose={() => setEnding(false)}
          onConfirm={onDisband}
          open
          title={labels.disbandTitle}
        />
      )}
    </>
  )
}
