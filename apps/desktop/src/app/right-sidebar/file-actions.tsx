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
  $creatingEntry,
  $fileActionDialog,
  beginInlineRename,
  cancelInlineRename,
  cancelNewEntry,
  closeFileActionDialog,
  copyFilePath,
  downloadRemoteFile,
  executeEntryCreate,
  executeFileDelete,
  executeFileRename,
  type FileActionTarget,
  requestFileDelete,
  requestNewEntry,
  revealFile,
  shouldOfferRemoteFileDownload,
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
  // Reveal needs the local OS file manager; hide it on a remote backend.
  // Rename/Delete work in BOTH modes now (remote goes through the gateway FS
  // API); remote delete is permanent rather than trash-recoverable, which the
  // confirm dialog's body copy reflects. Download uses the existing gateway
  // save bridge so a remote file can land on this machine.
  const localFs = !isDesktopFsRemoteMode()
  const remoteDownload = shouldOfferRemoteFileDownload(isDirectory)
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
        {isDirectory && (
          <>
            <ContextMenuItem
              onSelect={() => void requestNewEntry({ directory: true, parentDir: path })}
            >
              {m.newFolder}
            </ContextMenuItem>
            <ContextMenuItem
              onSelect={() => void requestNewEntry({ directory: false, parentDir: path })}
            >
              {m.newFile}
            </ContextMenuItem>
            <ContextMenuSeparator />
          </>
        )}
        <ContextMenuItem onSelect={() => void copyFilePath(path)}>{m.copyPath}</ContextMenuItem>
        {relativeTo && (
          <ContextMenuItem onSelect={() => void copyFilePath(toRelativePath(path, relativeTo))}>
            {m.copyRelativePath}
          </ContextMenuItem>
        )}
        {remoteDownload && (
          <>
            <ContextMenuSeparator />
            <ContextMenuItem onSelect={() => void downloadRemoteFile(path)}>{m.download}</ContextMenuItem>
          </>
        )}
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => beginInlineRename(path)}>{m.rename}</ContextMenuItem>
        <ContextMenuItem onSelect={() => requestFileDelete(target)} variant="destructive">
          {m.delete}
        </ContextMenuItem>
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
  // Local delete goes to the OS trash (recoverable); remote delete through the
  // gateway FS API is permanent — say so in the confirm copy.
  const remoteDelete = isDesktopFsRemoteMode()

  return (
    <ConfirmDialog
      confirmLabel={t.fileMenu.delete}
      description={remoteDelete ? t.fileMenu.deleteRemoteBody : t.fileMenu.deleteBody}
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
 *  row's label when `$renamingPath === path`, or — for the new-file/new-folder
 *  flow — when `$creatingEntry` targets this folder row (empty seed, no stem
 *  pre-select, and a first-tab-all select). */
export function InlineRenameInput({ className, name, path }: InlineRenameInputProps) {
  const creating = useStore($creatingEntry)
  const seedName = creating && creating.parentDir === path ? '' : name
  const [value, setValue] = useState(seedName)
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
    const creatingHere = creating && creating.parentDir === path
    const next = value.trim()

    if (commit && next) {
      try {
        if (creatingHere) {
          await executeEntryCreate(creating.directory, creating.parentDir, next)
        } else if (next !== name) {
          await executeFileRename(path, next)
        }
      } catch (error) {
        // Caller-owned failure toast (the store functions no longer notify);
        // a create failure keeps the generic message, rename keeps its own.
        notifyError(error, translateNow('errors.genericFailure'))
      }
    } else if (creatingHere) {
      // Esc / empty-blur cancel of a NEW-ENTRY flow: only executeEntryCreate's
      // finally clears $creatingEntry on the commit path, so the non-commit
      // path must clear it here too or the inline row can't be dismissed.
      cancelNewEntry()
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
        if (creating && creating.parentDir === path) {
          event.currentTarget.select()

          return
        }

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
