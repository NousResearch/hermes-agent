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
import { type DesktopFsOperationScope, isDesktopFsRemoteMode } from '@/lib/desktop-fs'
import { isSubmitEnter } from '@/lib/ime'
import { IS_MAC } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import {
  $creatingEntry,
  $fileActionDialog,
  $renamingPath,
  $renamingScope,
  beginInlineRename,
  cancelInlineRename,
  cancelNewEntry,
  closeFileActionDialog,
  copyFilePath,
  downloadRemoteFile,
  executeEntryCreate,
  executeFileDelete,
  executeFileRename,
  type FileActionDialog,
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
          <ContextMenuItem onSelect={() => void requestNewEntry({ directory: true, parentDir: path })}>
            {m.newFolder}
          </ContextMenuItem>
        )}
        {isDirectory && (
          <>
            <ContextMenuItem onSelect={() => void requestNewEntry({ directory: false, parentDir: path })}>
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
const dialogKeys = new WeakMap<FileActionDialog, number>()
let nextDialogKey = 0

export function FileActionDialogs() {
  const dialog = useStore($fileActionDialog)

  if (!dialog) {return null}

  if (!dialogKeys.has(dialog)) {dialogKeys.set(dialog, ++nextDialogKey)}

  return <FileDeleteDialog dialog={dialog} key={dialogKeys.get(dialog)} />
}

function FileDeleteDialog({ dialog }: { dialog: FileActionDialog }) {
  const { t } = useI18n()
  const remoteDelete = dialog.scope.mode === 'remote'

  return (
    <ConfirmDialog
      confirmLabel={t.fileMenu.delete}
      description={remoteDelete ? t.fileMenu.deleteRemoteBody : t.fileMenu.deleteBody}
      destructive
      onClose={() => {
        if ($fileActionDialog.get() === dialog) {closeFileActionDialog()}
      }}
      onConfirm={() => {
        if ($fileActionDialog.get() !== dialog) {throw new Error('File deletion was canceled')}

        return executeFileDelete(dialog.path, dialog.scope)
      }}
      open
      title={t.fileMenu.deleteTitle(dialog.name)}
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
const editorKeys = new WeakMap<object, number>()
let nextEditorKey = 0

export function InlineRenameInput(props: InlineRenameInputProps) {
  const creating = useStore($creatingEntry)
  const scope = useStore($renamingScope)
  const owner = creating?.parentDir === props.path ? creating : scope

  if (owner && !editorKeys.has(owner)) {editorKeys.set(owner, ++nextEditorKey)}

  return <OwnedInlineRenameInput {...props} key={owner ? editorKeys.get(owner) : 0} />
}

function ownsInlineRename(path: string, scope: DesktopFsOperationScope | null): boolean {
  return scope !== null && $renamingPath.get() === path && $renamingScope.get() === scope
}

function OwnedInlineRenameInput({ className, name, path }: InlineRenameInputProps) {
  const creating = useStore($creatingEntry)
  const renamingScope = useStore($renamingScope)
  const creatingHere = creating && creating.parentDir === path
  // Bind the editor to the request that mounted it, not a later request for the same row.
  const owner = useRef({ creating: creatingHere ? creating : null, scope: renamingScope }).current
  const seedName = creatingHere ? '' : name
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
    const ownsCreate = owner.creating !== null && $creatingEntry.get() === owner.creating
    const ownsRename = !owner.creating && ownsInlineRename(path, owner.scope)

    if (!ownsCreate && !ownsRename) {return}

    if (commit && value.trim()) {
      try {
        if (ownsCreate && owner.creating) {
          await executeEntryCreate(owner.creating, value)
        } else if (ownsRename && value !== name) {
          await executeFileRename(path, value, owner.scope || undefined)
        }
      } catch (error) {
        notifyError(error, translateNow('errors.genericFailure'))
      }
    } else if (ownsCreate && $creatingEntry.get() === owner.creating) {
      cancelNewEntry()
    }

    if (ownsRename && ownsInlineRename(path, owner.scope)) {
      cancelInlineRename()
    }
  }

  return (
    <input
      aria-label={
        creatingHere
          ? translateNow(creating.directory ? 'fileMenu.newFolder' : 'fileMenu.newFile')
          : translateNow('fileMenu.renameLabel')
      }
      autoCapitalize="off"
      autoComplete="off"
      autoCorrect="off"
      autoFocus
      className={cn(
        'min-w-0 flex-1 rounded-sm border border-[color-mix(in_srgb,var(--dt-composer-ring)_55%,transparent)] bg-(--ui-bg-elevated) px-1 py-0 text-xs text-foreground',
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

        if (isSubmitEnter(event)) {
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
