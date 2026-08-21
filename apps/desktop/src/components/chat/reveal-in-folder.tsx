import { type ReactNode, useState } from 'react'

import { pickRevealLabel } from '@/app/right-sidebar/file-actions'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { translateNow, useI18n } from '@/i18n'
import { isDesktopFsRemoteMode, revealDesktopPath } from '@/lib/desktop-fs'
import { notify, notifyError } from '@/store/notifications'

/**
 * Right-click menu over a transcript file affordance ("Open …" fallback link,
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
  const [open, setOpen] = useState(false)

  if (isDesktopFsRemoteMode()) {
    return <>{children}</>
  }

  const reveal = () => {
    void revealDesktopPath(path).catch(error => notifyError(error, translateNow('errors.genericFailure')))
  }

  const copyPath = () => {
    void navigator.clipboard
      .writeText(path)
      .then(() => notify({ durationMs: 1500, kind: 'info', message: translateNow('fileMenu.pathCopied') }))
      .catch(() => notifyError(new Error('clipboard'), translateNow('common.copyFailed')))
  }

  return (
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
  )
}
