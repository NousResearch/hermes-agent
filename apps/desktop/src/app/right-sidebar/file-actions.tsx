import { useStore } from '@nanostores/react'
import { type KeyboardEvent as ReactKeyboardEvent, type ReactNode, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
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
  executeFileMove,
  executeFileRename,
  type FileActionTarget,
  requestFileDelete,
  requestFileMove,
  revealFile,
  toRelativePath
} from '@/store/file-actions'

const IS_WIN = typeof navigator !== 'undefined' && /win/i.test(navigator.platform || navigator.userAgent || '')

function normalizeWorkspacePath(value: string): string {
  return value.replace(/\\/g, '/').replace(/\/+$/, '') || '/'
}

function parentWorkspacePath(value: string): string {
  const normalized = normalizeWorkspacePath(value)
  const slash = normalized.lastIndexOf('/')

  return slash <= 0 ? '/' : normalized.slice(0, slash)
}

function actionErrorMessage(error: unknown): string {
  return error instanceof Error && error.message ? error.message : translateNow('errors.genericFailure')
}

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
  /** Visible browser/repository root used by destructive-operation guards. */
  browserRoot?: null | string
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
export function FileEntryContextMenu({
  browserRoot,
  children,
  isDirectory,
  name,
  path,
  relativeTo
}: FileEntryContextMenuProps) {
  const { t } = useI18n()
  const m = t.fileMenu
  // Reveal is the only action that inherently requires a local shell. Remote
  // rename/move/delete route through the authenticated filesystem API.
  const localFs = !isDesktopFsRemoteMode()
  const target: FileActionTarget = { browserRoot: browserRoot ?? relativeTo ?? undefined, isDirectory, name, path }
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
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => beginInlineRename(path)}>{m.rename}</ContextMenuItem>
        <ContextMenuItem onSelect={() => requestFileMove(target)}>{m.move}</ContextMenuItem>
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
  const remoteFs = isDesktopFsRemoteMode()
  const deleting = dialog?.kind === 'delete'
  const moving = dialog?.kind === 'move'
  const [destination, setDestination] = useState('')
  const [moveBusy, setMoveBusy] = useState(false)
  const [moveError, setMoveError] = useState<string | null>(null)
  const sourceParent = moving ? parentWorkspacePath(dialog.path) : ''
  const destinationUnchanged = moving && normalizeWorkspacePath(destination.trim()) === sourceParent

  useEffect(() => {
    setDestination(moving ? (dialog.browserRoot ?? '') : '')
    setMoveBusy(false)
    setMoveError(null)
  }, [dialog, moving])

  return (
    <>
      <ConfirmDialog
        confirmLabel={t.fileMenu.delete}
        description={remoteFs ? t.fileMenu.deleteRemoteBody : t.fileMenu.deleteBody}
        destructive
        onClose={closeFileActionDialog}
        onConfirm={() => {
          if (deleting) {
            return executeFileDelete(dialog.path, dialog.browserRoot ?? '')
          }
        }}
        open={deleting}
        title={deleting ? t.fileMenu.deleteTitle(dialog.name) : ''}
      />
      <Dialog onOpenChange={open => !open && !moveBusy && closeFileActionDialog()} open={moving}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{moving ? t.fileMenu.moveTitle(dialog.name) : ''}</DialogTitle>
            {moving ? (
              <p className="truncate font-mono text-[0.7rem] text-(--ui-text-secondary)" title={dialog.path}>
                {dialog.path}
              </p>
            ) : null}
          </DialogHeader>
          <label className="grid gap-2 text-xs text-(--ui-text-secondary)">
            {t.fileMenu.moveLabel}
            <input
              aria-label={t.fileMenu.moveLabel}
              autoCapitalize="off"
              autoComplete="off"
              autoCorrect="off"
              className="h-8 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-input) px-2 text-sm text-foreground outline-none focus:border-(--ui-accent)"
              onChange={event => {
                setDestination(event.target.value)
                setMoveError(null)
              }}
              spellCheck={false}
              value={destination}
            />
          </label>
          {moveError ? (
            <div
              className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive"
              role="alert"
            >
              {moveError}
            </div>
          ) : null}
          <DialogFooter>
            <Button disabled={moveBusy} onClick={closeFileActionDialog} variant="secondary">
              {t.common.cancel}
            </Button>
            <Button
              disabled={moveBusy || !moving || !destination.trim() || destinationUnchanged}
              onClick={async () => {
                if (!moving) {
                  return
                }

                setMoveBusy(true)
                setMoveError(null)

                try {
                  await executeFileMove(dialog.path, destination.trim(), dialog.browserRoot ?? '')
                  closeFileActionDialog()
                } catch (error) {
                  setMoveError(actionErrorMessage(error))
                } finally {
                  setMoveBusy(false)
                }
              }}
            >
              {moveBusy ? t.common.loading : t.fileMenu.moveConfirm}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
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
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  // Enter then the resulting blur must not both commit; latch on a successful
  // commit or explicit cancel, while failed commits remain retryable.
  const done = useRef(false)
  const savingRef = useRef(false)
  const inputRef = useRef<HTMLInputElement>(null)
  // Focus churn right after mount (context-menu close, arborist refocus, the
  // fall-through click on the row) would blur→commit→cancel instantly; ignore
  // blurs in this window and grab focus back instead.
  const mountedAt = useRef(Date.now())

  const finish = async (commit: boolean) => {
    if (done.current || savingRef.current) {
      return
    }

    const next = value.trim()

    if (!commit || !next || next === name) {
      done.current = true
      cancelInlineRename()

      return
    }

    savingRef.current = true
    setSaving(true)
    setError(null)

    try {
      await executeFileRename(path, next)
      done.current = true
      cancelInlineRename()
    } catch (renameError) {
      setError(actionErrorMessage(renameError))
      window.setTimeout(() => inputRef.current?.focus(), 0)
    } finally {
      savingRef.current = false
      setSaving(false)
    }
  }

  return (
    <>
      <input
        aria-invalid={Boolean(error)}
        aria-label={translateNow('fileMenu.renameLabel')}
        autoCapitalize="off"
        autoComplete="off"
        autoCorrect="off"
        autoFocus
        className={cn(
          'min-w-0 flex-1 rounded-sm border border-[color-mix(in_srgb,var(--dt-composer-ring)_55%,transparent)] bg-(--ui-bg-elevated) px-1 py-0 text-xs text-foreground outline-none',
          error && 'border-destructive',
          className
        )}
        onBlur={event => {
          if (savingRef.current) {
            return
          }

          if (Date.now() - mountedAt.current < 250) {
            event.currentTarget.focus()

            return
          }

          void finish(true)
        }}
        onChange={event => {
          setValue(event.target.value)
          setError(null)
        }}
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
        readOnly={saving}
        ref={inputRef}
        spellCheck={false}
        title={error ?? undefined}
        value={value}
      />
      {error ? (
        <span className="max-w-32 shrink-0 truncate text-[0.625rem] text-destructive" role="alert" title={error}>
          {error}
        </span>
      ) : null}
    </>
  )
}
