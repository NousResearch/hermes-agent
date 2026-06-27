import { useStore } from '@nanostores/react'
import { type KeyboardEvent as ReactKeyboardEvent, type ReactNode, useRef, useState } from 'react'

import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { translateNow, useI18n } from '@/i18n'
import { isDesktopFsRemoteMode } from '@/lib/desktop-fs'
import { IS_MAC } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import {
  $fileActionDialog,
  beginInlineRename,
  cancelInlineRename,
  closeFileActionDialog,
  copyFilePath,
  executeFileDelete,
  executeFileRename,
  type FileActionTarget,
  requestFileDelete,
  revealFile,
  toRelativePath
} from '@/store/file-actions'
import { notifyError } from '@/store/notifications'

const IS_WIN = typeof navigator !== 'undefined' && /win/i.test(navigator.platform || navigator.userAgent || '')

// F2 starts a rename anywhere; Enter starts one when a row is focused (VS Code).
export function isRenameShortcut(event: KeyboardEvent | ReactKeyboardEvent): boolean {
  return event.key === 'F2' || event.key === 'Enter'
}

/** The platform-appropriate "reveal in file manager" label (Finder / Explorer
 *  / containing folder). Shared so every file menu reads consistently. */
export function pickRevealLabel(finder: string, explorer: string, fileManager: string): string {
  return IS_MAC ? finder : IS_WIN ? explorer : fileManager
}

interface FileEntryContextMenuProps {
  children: ReactNode
  isDirectory: boolean
  /** Display name (basename). */
  name: string
  /** Absolute path on disk. */
  path: string
  /** Base dir for "Copy Relative Path" (the cwd / repo root). Omit to hide it. */
  relativeTo?: null | string
}

/** Right-click menu shared by both file trees (browser + review/git). */
export function FileEntryContextMenu({ children, isDirectory, name, path, relativeTo }: FileEntryContextMenuProps) {
  const { t } = useI18n()
  const m = t.fileMenu
  // Reveal / rename / delete need the local filesystem; hide them on a remote
  // backend (copy-path still works everywhere).
  const localFs = !isDesktopFsRemoteMode()
  const target: FileActionTarget = { isDirectory, name, path }
  const revealLabel = pickRevealLabel(m.revealFinder, m.revealExplorer, m.revealFileManager)

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
      {/* Don't restore focus to the row on close: "Rename" mounts an autofocused
          inline input, and the default focus-return would blur it immediately. */}
      <ContextMenuContent onCloseAutoFocus={event => event.preventDefault()}>
        {localFs && (
          <>
            <ContextMenuItem onSelect={() => void revealFile(path)}>{revealLabel}</ContextMenuItem>
            <ContextMenuSeparator />
          </>
        )}
        <ContextMenuItem onSelect={() => void copyFilePath(path)}>{m.copyPath}</ContextMenuItem>
        {relativeTo && (
          <ContextMenuItem onSelect={() => void copyFilePath(toRelativePath(path, relativeTo))}>
            {m.copyRelativePath}
          </ContextMenuItem>
        )}
        {localFs && (
          <>
            <ContextMenuSeparator />
            <ContextMenuItem onSelect={() => beginInlineRename(path)}>{m.rename}</ContextMenuItem>
            <ContextMenuItem onSelect={() => requestFileDelete(target)} variant="destructive">
              {m.delete}
            </ContextMenuItem>
          </>
        )}
      </ContextMenuContent>
    </ContextMenu>
  )
}

/** Mounted once near the app root: the delete confirm dialog for shared file
 *  actions. Rename is inline (see {@link InlineRenameInput}). */
export function FileActionDialogs() {
  const { t } = useI18n()
  const dialog = useStore($fileActionDialog)
  const deleting = dialog?.kind === 'delete'

  return (
    <ConfirmDialog
      confirmLabel={t.fileMenu.delete}
      description={t.fileMenu.deleteBody}
      destructive
      onClose={closeFileActionDialog}
      onConfirm={() => {
        if (deleting) {
          return executeFileDelete(dialog.path)
        }
      }}
      open={deleting}
      title={deleting ? t.fileMenu.deleteTitle(dialog.name) : ''}
    />
  )
}

interface InlineRenameInputProps {
  className?: string
  /** Display name (basename) to seed the editor. */
  name: string
  /** Absolute path being renamed. */
  path: string
}

/** The in-row rename editor (VS Code style): seeded with the name (stem
 *  pre-selected), commits on Enter/blur, cancels on Esc. Render it in place of a
 *  row's label when `$renamingPath === path`. */
export function InlineRenameInput({ className, name, path }: InlineRenameInputProps) {
  const [value, setValue] = useState(name)
  // Enter then the resulting blur must not both commit; latch on first finish.
  const done = useRef(false)
  // Focus churn right after mount (context-menu close, arborist refocus, the
  // fall-through click on the row) would blur→commit→cancel instantly; ignore
  // blurs in this window and grab focus back instead.
  const mountedAt = useRef(Date.now())

  const finish = async (commit: boolean) => {
    if (done.current) {
      return
    }

    done.current = true
    const next = value.trim()

    if (commit && next && next !== name) {
      try {
        await executeFileRename(path, next)
      } catch (error) {
        notifyError(error, translateNow('errors.genericFailure'))
      }
    }

    cancelInlineRename()
  }

  return (
    <input
      aria-label={translateNow('fileMenu.renameLabel')}
      autoCapitalize="off"
      autoComplete="off"
      autoCorrect="off"
      autoFocus
      className={cn(
        'min-w-0 flex-1 rounded-sm border border-[color-mix(in_srgb,var(--dt-composer-ring)_55%,transparent)] bg-(--ui-bg-elevated) px-1 py-0 text-xs text-foreground outline-none',
        className
      )}
      onBlur={event => {
        if (Date.now() - mountedAt.current < 250) {
          event.currentTarget.focus()

          return
        }

        void finish(true)
      }}
      onChange={event => setValue(event.target.value)}
      onClick={event => event.stopPropagation()}
      onDoubleClick={event => event.stopPropagation()}
      onFocus={event => {
        const dot = event.currentTarget.value.lastIndexOf('.')
        event.currentTarget.setSelectionRange(0, dot > 0 ? dot : event.currentTarget.value.length)
      }}
      onKeyDown={event => {
        event.stopPropagation()

        if (event.key === 'Enter') {
          event.preventDefault()
          void finish(true)
        } else if (event.key === 'Escape') {
          event.preventDefault()
          void finish(false)
        }
      }}
      spellCheck={false}
      value={value}
    />
  )
}
