import { useStore } from '@nanostores/react'
import { type ReactNode, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { translateNow, useI18n } from '@/i18n'
import { revealDesktopPath } from '@/lib/desktop-fs'
import { pickRevealLabel } from '@/lib/file-manager'
import { isUnsafeRevealPath } from '@/lib/reveal-path-guard'
import { notify, notifyError } from '@/store/notifications'
import { isSessionRemote } from '@/store/session-states'

/**
 * Right-click menu over a transcript file affordance (filename download link,
 * inline image) offering Reveal-in-file-manager + Copy Path — the same two
 * local actions the file trees offer. Local gateway only: on a remote
 * connection the file lives on the gateway machine, and reveal silently
 * no-ops against the wrong disk (same gate as `FileEntryContextMenu`).
 *
 * `children` must be a single element (Radix `asChild`): it receives the
 * contextmenu wiring and must be able to host it (a plain element is fine).
 */
export function RevealInFolderTrigger({ children, path }: { children: ReactNode; path: string }) {
  const { t } = useI18n()
  const storedSessionId = useStore(useSessionView().$storedId)
  const [open, setOpen] = useState(false)

  if (isSessionRemote(storedSessionId)) {
    return <>{children}</>
  }

  const reveal = () => {
    if (isUnsafeRevealPath(path)) {
      notifyError(new Error('Unsafe path'), translateNow('errors.genericFailure'))

      return
    }

    void revealDesktopPath(path).catch(error => notifyError(error, translateNow('errors.genericFailure')))
  }

  const copyPath = () => {
    void navigator.clipboard
      .writeText(path)
      .then(() => notify({ durationMs: 1500, kind: 'info', message: translateNow('fileMenu.pathCopied') }))
      .catch(() => notifyError(new Error('clipboard'), translateNow('common.copyFailed')))
  }

  return (
    <span className="inline-flex max-w-full items-start gap-1">
      <ContextMenu onOpenChange={setOpen} open={open}>
        <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
        <ContextMenuContent onCloseAutoFocus={event => event.preventDefault()}>
          <ContextMenuItem onSelect={reveal}>
            {pickRevealLabel(t.fileMenu.revealFinder, t.fileMenu.revealExplorer, t.fileMenu.revealFileManager)}
          </ContextMenuItem>
          <ContextMenuSeparator />
          <ContextMenuItem onSelect={copyPath}>{t.fileMenu.copyPath}</ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            aria-label={t.fileMenu.actions}
            className="mt-0.5 shrink-0 text-muted-foreground hover:text-foreground"
            type="button"
          >
            <Codicon name="kebab-vertical" size="0.875rem" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={reveal}>
            {pickRevealLabel(t.fileMenu.revealFinder, t.fileMenu.revealExplorer, t.fileMenu.revealFileManager)}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={copyPath}>{t.fileMenu.copyPath}</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </span>
  )
}
