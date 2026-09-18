// The slots a server on this Mac fills.
//
// None of this is new: every piece is the MCP tab's own, imported from
// `capabilities/mcp/`. `Advanced` is the mcp.json entry, the logs and Remove;
// `Add your own` is the same editor with a starter entry already seeded. One
// document, one save path, one place a server can be defined.

import { type ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'

import { McpJsonEditor, McpLogPane } from '../mcp/mcp-editor'
import type { McpServersController } from '../mcp/use-mcp-servers'

export interface LocalAdvancedProps {
  controller: McpServersController
  /** The server's key in mcp.json. */
  name: string
  onRemove: () => void
}

/** The `Advanced` section of an opened local server: its mcp.json entry, its
 *  logs, and the one destructive action. */
export function LocalAdvanced({ controller, name, onRemove }: LocalAdvancedProps) {
  const { t } = useI18n()
  const m = t.settings.mcp

  return (
    <div className="grid min-h-0 gap-2">
      <div className="h-56 overflow-hidden rounded-md border border-(--ui-stroke-tertiary)">
        <McpJsonEditor controller={controller} highlightServer={name} />
      </div>

      <div className="h-40 overflow-hidden rounded-md border border-(--ui-stroke-tertiary)">
        <McpLogPane server={name} />
      </div>

      <Button className="justify-self-start text-destructive" onClick={onRemove} size="xs" variant="text">
        {m.remove}
      </Button>
    </div>
  )
}

/** A plain frame around the mcp.json editor. The editor owns its own header and
 *  Save button, so the dialog adds a title and nothing else. */
export function McpDocumentDialog({
  children,
  onOpenChange,
  open,
  title
}: {
  children: ReactNode
  onOpenChange: (open: boolean) => void
  open: boolean
  title: string
}) {
  const { t } = useI18n()

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent bodyClassName="gap-0 overflow-hidden p-0" className="h-[min(40rem,80vh)] min-w-[min(48rem,90vw)]">
        <header className="flex shrink-0 items-center border-b border-(--ui-stroke-tertiary) px-5 py-3">
          <DialogTitle className="text-base font-semibold">{title}</DialogTitle>
        </header>
        <DialogDescription className="sr-only">{t.connectorsPage.dialog.advancedHint}</DialogDescription>
        <div className="flex min-h-0 flex-1 flex-col">{children}</div>
      </DialogContent>
    </Dialog>
  )
}

export interface RemoveServerConfirmProps {
  controller: McpServersController
  /** `null` closes it. */
  name: null | string
  onClose: () => void
  onRemoved: () => void
}

/** Removing a server is the page's one destructive local action, so it asks
 *  through the app's `ConfirmDialog` like every other one. */
export function RemoveServerConfirm({ controller, name, onClose, onRemoved }: RemoveServerConfirmProps) {
  const { t } = useI18n()
  const m = t.settings.mcp

  return (
    <ConfirmDialog
      confirmLabel={m.remove}
      description={t.connectorsPage.dialog.removeServerBody}
      destructive
      onClose={onClose}
      onConfirm={async () => {
        if (!name) {
          return
        }

        // A refused write throws, because `ConfirmDialog` owns the inline error:
        // swallowing it would run the done beat and close as if the entry had
        // left mcp.json.
        if (!(await controller.removeServer(name))) {
          throw new Error(m.removeFailed)
        }

        onRemoved()
      }}
      open={name !== null}
      title={t.connectorsPage.dialog.removeServerTitle(name ?? '')}
    />
  )
}
