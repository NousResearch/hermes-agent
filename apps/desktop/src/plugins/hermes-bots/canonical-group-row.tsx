import {
  Button,
  ConfirmDialog,
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import { useBots } from './i18n'

interface CanonicalGroupRowProps {
  memberCount: number
  name: string
  /** Rejects with the gateway's refusal so the confirm stays open and shows it. */
  onDelete: () => Promise<void>
  onOpen: () => void
}

/** One gateway Group Chat in the roster list, with the same right-click Open /
 *  Delete a Desktop-local group row has (#143571). */
export function CanonicalGroupRow({ memberCount, name, onDelete, onOpen }: CanonicalGroupRowProps) {
  const labels = useCanonicalGroupLabels()
  const { group } = useBots()
  const [confirming, setConfirming] = useState(false)

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <Button onClick={onOpen} variant="ghost">
            {name}
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
