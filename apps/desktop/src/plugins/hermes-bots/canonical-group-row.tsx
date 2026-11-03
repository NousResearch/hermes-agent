import {
  Button,
  Codicon,
  ConfirmDialog,
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
  GlyphSpinner,
  useValue
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { $canonicalRoomLive, canonicalRoomLiveKey } from './canonical-group-details'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalGroupBinding, CanonicalRoomMember } from './canonical-groups'
import { RoomBotAvatar } from './group-room-avatar'
import { AvatarStack } from './group-room-ui'
import { useBots } from './i18n'

interface CanonicalGroupRowProps {
  binding?: CanonicalGroupBinding
  memberCount: number
  members?: CanonicalRoomMember[]
  name: string
  /** Rejects with the gateway's refusal so the confirm stays open and shows it. */
  onDelete: () => Promise<void>
  onOpen: () => void
}

/** One gateway Group Chat in the roster list, with the same right-click Open /
 *  Delete a Desktop-local group row has (#143571). */
export function CanonicalGroupRow({
  binding,
  memberCount,
  members = [],
  name,
  onDelete,
  onOpen
}: CanonicalGroupRowProps) {
  const labels = useCanonicalGroupLabels()
  const { group } = useBots()
  const [confirming, setConfirming] = useState(false)
  // Live state an open room published; a room nobody has open shows none.
  const live = useValue($canonicalRoomLive)[binding ? canonicalRoomLiveKey(binding, binding.roomId) : '']
  const state = live?.needsYou ? 'needsYou' : live?.failed ? 'failed' : live?.working ? 'working' : null

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <Button aria-label={name} className="justify-start gap-2" onClick={onOpen} variant="ghost">
            <AvatarStack
              avatars={members.map(member => ({
                key: member.member_id,
                node: <RoomBotAvatar name={member.profile || member.handle} size={16} />
              }))}
              max={4}
            />
            <span className="min-w-0 truncate">{name}</span>
            {state && (
              <span
                className="ml-auto inline-flex shrink-0 items-center gap-1 text-[0.6875rem]"
                data-room-state={state}
              >
                {state === 'working' ? (
                  <span aria-hidden="true" className="text-(--ui-accent)">
                    <GlyphSpinner />
                  </span>
                ) : (
                  <Codicon
                    className={state === 'failed' ? 'text-destructive' : 'text-(--ui-accent)'}
                    name={state === 'failed' ? 'error' : 'bell-dot'}
                  />
                )}
                <span className="sr-only">
                  {state === 'working' ? labels.rowWorking : state === 'failed' ? labels.rowFailed : labels.rowNeedsYou}
                </span>
              </span>
            )}
          </Button>
        </ContextMenuTrigger>
        <ContextMenuContent>
          <ContextMenuItem onSelect={onOpen}>{labels.openGroup}</ContextMenuItem>
          <ContextMenuSeparator />
          <ContextMenuItem onSelect={() => setConfirming(true)} variant="destructive">
            {group.deleteAction}
          </ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>
      {confirming && (
        <ConfirmDialog
          busyLabel={labels.disbanding}
          confirmLabel={labels.disband}
          description={
            <span>
              {labels.disbandDescPrefix}
              <span className="font-medium text-foreground">{name}</span>
              {labels.disbandDescSuffix(memberCount)}
            </span>
          }
          destructive
          doneLabel={labels.disbandDone}
          onClose={() => setConfirming(false)}
          onConfirm={onDelete}
          open
          title={labels.disbandTitle}
        />
      )}
    </>
  )
}
