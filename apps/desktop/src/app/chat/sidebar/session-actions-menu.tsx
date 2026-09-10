import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useEffect, useRef, useState } from 'react'

import { openSession } from '@/app/open-session'
import {
  closeableTreeSiblings,
  closeAllTreeTabs,
  closeOtherTreeTabs,
  closeTreeTabsToRight,
  reloadTreePane,
  treeTabCloseTargets
} from '@/components/pane-shell/tree/store'
import {
  type ActionItemSpec,
  ActionsContextMenu,
  ActionsMenu,
  type MenuKit,
  renderActionItem
} from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ColorSwatches } from '@/components/ui/color-swatches'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { CopyButton } from '@/components/ui/copy-button'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { renameSession } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { PROFILE_SWATCHES } from '@/lib/profile-color'
import { exportSession } from '@/lib/session-export'
import { activeGateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $projectTree, moveSessionToProject, projectIdForCwd, projectRootCwd } from '@/store/projects'
import {
  $activeSessionId,
  $connection,
  $selectedStoredSessionId,
  $sessions,
  $unreadFinishedSessionIds,
  markSessionRead,
  sessionMatchesStoredId,
  sessionPinId,
  setSessions
} from '@/store/session'
import { $sessionColorOverrides, setSessionColorOverride } from '@/store/session-color'
import { $sessionTiles, closeAllOpenSessionTiles, closeOpenSessionTilesForPanes } from '@/store/session-states'
import { ackStoredSessionId } from '@/store/session-unread'
import { canOpenSessionInTerminal, canOpenSessionWindow, openSessionInTerminal } from '@/store/windows'

import type { SessionTitleResponse } from '../../types'

// Rename a session, preferring the gateway's session.title RPC over REST.
//
// A freshly *branched* session (and any brand-new chat) lives only in the
// gateway's in-memory _sessions map keyed by its RUNTIME id — no row is
// persisted to state.db until the first turn. REST PATCH /api/sessions/{id}
// resolves against the stored sessions table, so it 404s ("Session not found")
// on these runtime-only sessions. The session.title RPC resolves the live
// runtime session AND persists the row on demand, so it succeeds where REST
// cannot. This mirrors the /title slash command's fix (use-prompt-actions.ts).
//
// We only take the RPC path for the ACTIVE/selected session: its runtime id is
// known ($activeSessionId) and it lives on the active gateway, so there is no
// profile-routing ambiguity. Every other row (already persisted, possibly on a
// background profile) keeps the REST path, which handles profile scoping and a
// non-empty title is required by the RPC (it rejects clears), so clears stay on
// REST too.
export async function renameSessionPreferringRpc(
  storedSessionId: string,
  title: string,
  profile?: string
): Promise<{ title?: string }> {
  const isActiveRow = storedSessionId === $selectedStoredSessionId.get()
  const runtimeId = isActiveRow ? $activeSessionId.get() : null
  const gateway = activeGateway()

  if (title && runtimeId && gateway) {
    try {
      const result = await gateway.request<SessionTitleResponse>('session.title', {
        session_id: runtimeId,
        title
      })

      return { title: result?.title ?? title }
    } catch (err) {
      // Fall through to REST — e.g. the socket is mid-reconnect. REST still
      // works for any session that already has a persisted row. Log so a
      // genuine RPC-side failure (which then surfaces a REST 404 for the
      // runtime id) is at least diagnosable instead of silently swallowed.
      console.warn('session.title RPC rename failed; falling back to REST', err)
    }
  }

  return renameSession(storedSessionId, title, profile)
}

interface SessionActions {
  sessionId: string
  title: string
  pinned?: boolean
  /** Backend-derived read state — drives the Mark as unread/read label. */
  unread?: boolean
  profile?: string
  onPin?: () => void
  /** Toggle the persisted read-state watermark for this row. */
  onToggleUnread?: () => void
  onBranch?: () => void
  onArchive?: () => void
  onDelete?: () => void
  /** Close this surface (a tile tab) — omitted where nothing closes (sidebar
   *  rows, the main tab). */
  onClose?: () => void
  /** TAB surfaces: the session is already a tab, so "Open in new tab" is
   *  nonsense there — sidebar rows/dropdowns keep it. */
  surface?: 'row' | 'tab'
  /** The tab's layout-tree pane id (`session-tile:<id>` or `workspace`) — enables
   *  the Close-others / to-the-right / all tab verbs. Tab surfaces only. */
  tabPaneId?: string
  /** The MAIN tab's escape hatch: hide the zone's tab bar (it sticky-shows
   *  once a tab is ever gained; this is the explicit off switch). */
  onHideTabBar?: () => void
}

// The color picker inside the session menu's Appearance submenu. Its own
// component so only an OPEN submenu subscribes to the stores (not every row's
// menu). Reads/writes the override keyed by the DURABLE id so a color survives
// compression; clearing falls back to the inherited project color.
function SessionColorSwatches({ sessionId }: { sessionId: string }) {
  const { t } = useI18n()
  const overrides = useStore($sessionColorOverrides)
  const session = useStore($sessions).find(s => sessionMatchesStoredId(s, sessionId))
  const durableId = session ? sessionPinId(session) : sessionId

  return (
    <ColorSwatches
      clearIcon="circle-slash"
      clearLabel={t.sidebar.projects.noColor}
      onChange={color => setSessionColorOverride(durableId, color)}
      swatches={PROFILE_SWATCHES}
      value={overrides[durableId] ?? null}
    />
  )
}

// The project list inside the session menu's "Move to project" submenu. Its own
// component so only an OPEN submenu subscribes to the stores (same reasoning as
// SessionColorSwatches). Re-homes the session's workspace at the target
// project's root — the fix for a chat created in the wrong folder. The current
// owner and folderless projects (the Home bucket) are excluded: there is
// nothing to move into.
function MoveToProjectItems({ kit, sessionId, profile }: { kit: MenuKit; sessionId: string; profile?: string }) {
  const { t } = useI18n()
  const p = t.sidebar.projects
  const tree = useStore($projectTree)
  const session = useStore($sessions).find(s => sessionMatchesStoredId(s, sessionId))
  const cwd = session?.cwd?.trim() || ''
  const currentProjectId = cwd ? projectIdForCwd(cwd) : null
  const targets = tree.filter(node => node.id !== currentProjectId && !node.isNoProject && projectRootCwd(node))

  if (targets.length === 0) {
    return <kit.Item disabled>{p.moveNoProjects}</kit.Item>
  }

  return (
    <>
      {targets.map(node => (
        <kit.Item
          key={node.id}
          onSelect={() => {
            triggerHaptic('selection')
            moveSessionToProject(sessionId, node.id, profile)
              .then(() => notify({ durationMs: 2_000, kind: 'success', message: p.movedTo(node.label) }))
              .catch(err => notifyError(err, p.moveFailed))
          }}
        >
          {node.label}
        </kit.Item>
      ))}
    </>
  )
}

function useSessionActions({
  sessionId,
  title,
  pinned = false,
  unread = false,
  profile,
  onPin,
  onToggleUnread,
  onBranch,
  onArchive,
  onDelete,
  onClose,
  onHideTabBar,
  surface = 'row',
  tabPaneId
}: SessionActions) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [renameOpen, setRenameOpen] = useState(false)
  // The rename item opens a Dialog. When a menu closes, Radix restores focus to
  // its trigger — for a sidebar row that trigger is the row's own <button>, so
  // focus lands there instead of the dialog's input: Space then activates the
  // row (selecting the session) and the arrow keys move the list rather than
  // the caret. Suppress that one restore so the dialog keeps focus; every other
  // action leaves the restore alone (it's the correct behavior for them). Mirrors
  // the project menu's appearance-popover guard.
  const suppressCloseFocusRef = useRef(false)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const tiles = useStore($sessionTiles)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const isRemote = useStore($connection)?.mode === 'remote'
  // The row's finished-unread dot is cleared by opening the session (main or
  // tile) — this menu item is the explicit escape hatch for the rest.
  const isUnread = useStore($unreadFinishedSessionIds).includes(sessionId)

  // Already showing as a tab somewhere (a tile, or loaded in main — main IS
  // a tab): offering "Open in new tab" again is noise.
  const alreadyTabbed = sessionId === selectedStoredSessionId || tiles.some(tile => tile.storedSessionId === sessionId)

  const spec = (partial: Omit<ActionItemSpec, 'onSelect'> & { onSelect: () => void }): ActionItemSpec => partial

  // OPEN — where else this session can go. A tab surface IS a tab already,
  // so it only offers the window hop (and its own Close, below).
  const openItems: ActionItemSpec[] = [
    ...(surface === 'row' && !alreadyTabbed
      ? [
          spec({
            disabled: !sessionId,
            icon: 'browser',
            label: r.openInNewTab,
            onSelect: () => {
              triggerHaptic('selection')
              // Stack into the MAIN zone as a tab (center dock; the strip
              // sticky-shows on gain) — the door to the tab bar. Focuses first
              // if the session is already on screen.
              openSession(sessionId, () => undefined, 'tab')
            }
          })
        ]
      : []),
    ...(canOpenSessionWindow()
      ? [
          spec({
            disabled: !sessionId,
            icon: 'link-external',
            label: r.newWindow,
            onSelect: () => {
              triggerHaptic('selection')
              openSession(sessionId, () => undefined, 'window')
            }
          })
        ]
      : []),
    // The user's OWN terminal, not the in-app pane: resumes the session in the
    // TUI. Hidden on a remote connection — the emulator we'd open runs on this
    // machine while the session (and its runtime) lives on the remote host.
    ...(canOpenSessionInTerminal() && !isRemote
      ? [
          spec({
            disabled: !sessionId,
            icon: 'terminal',
            label: r.openInTerminal,
            onSelect: () => {
              triggerHaptic('selection')

              // Read the row lazily: subscribing every row's menu to $sessions
              // would re-render the whole sidebar on each session update.
              const cwd =
                $sessions
                  .get()
                  .find(s => sessionMatchesStoredId(s, sessionId))
                  ?.cwd?.trim() || undefined

              void openSessionInTerminal(sessionId, { cwd, profile })
            }
          })
        ]
      : [])
  ]

  // IDENTITY — name/mark/reference the session.
  const identityItems: ActionItemSpec[] = [
    spec({
      disabled: !sessionId,
      icon: 'edit',
      label: r.rename,
      onSelect: () => {
        triggerHaptic('selection')
        // Keep focus off the row trigger so it lands in the dialog input.
        suppressCloseFocusRef.current = true
        setRenameOpen(true)
      }
    }),
    spec({
      disabled: !onPin,
      icon: 'pin',
      label: pinned ? r.unpin : r.pin,
      onSelect: () => {
        triggerHaptic('selection')
        onPin?.()
      }
    }),
    // One read-state item, driven by BOTH unread sources: the transient
    // finished-unread dot (isUnread) and the backend watermark (unread).
    // "Mark as read" clears whichever is lit; "Mark as unread" arms the
    // persisted watermark so the dot survives restarts.
    spec({
      disabled: !sessionId || (!onToggleUnread && !isUnread),
      // Closed envelope = unread, open envelope = read (codicon has mail and
      // mail-read, but no mail-unread glyph — verified against the font css).
      icon: unread || isUnread ? 'mail-read' : 'mail',
      label: unread || isUnread ? r.markRead : r.markUnread,
      onSelect: () => {
        triggerHaptic('selection')

        if (unread || isUnread) {
          // Clear the transient family dot immediately (and ack the persisted
          // watermark/marker so a list refresh doesn't repaint it)…
          markSessionRead(sessionId)
          ackStoredSessionId(sessionId)

          // …and retire the persisted watermark when the row carries one.
          if (unread) {
            onToggleUnread?.()
          }
        } else {
          onToggleUnread?.()
        }
      }
    })
  ]

  // WORK — derive/extract from the session.
  const workItems: ActionItemSpec[] = [
    spec({
      disabled: !onBranch,
      // Fork glyph to match the inline message action's GitFork icon
      // (assistant-message.tsx). NB: this codicon font has no `git-fork`
      // glyph (only `git-fork-private`); `repo-forked` is the fork icon.
      icon: 'repo-forked',
      label: r.branchFrom,
      onSelect: () => {
        triggerHaptic('selection')
        onBranch?.()
      }
    }),
    spec({
      disabled: !sessionId,
      icon: 'cloud-download',
      label: r.export,
      onSelect: () => {
        triggerHaptic('selection')
        void exportSession(sessionId, { profile, title })
      }
    })
  ]

  // TAB — verbs that act on the strip (tabs only; a row isn't a tab).
  const closeTargets = surface === 'tab' && tabPaneId ? treeTabCloseTargets(tabPaneId) : null

  const tabItems: ActionItemSpec[] =
    surface === 'tab'
      ? [
          ...(tabPaneId
            ? [
                spec({
                  icon: 'refresh',
                  label: t.zones.reload,
                  onSelect: () => {
                    triggerHaptic('selection')
                    reloadTreePane(tabPaneId)
                  }
                })
              ]
            : []),
          ...(onClose
            ? [
                spec({
                  disabled: false,
                  icon: 'close',
                  label: t.common.close,
                  onSelect: () => {
                    triggerHaptic('selection')
                    onClose()
                  }
                })
              ]
            : []),
          ...(tabPaneId
            ? [
                spec({
                  disabled: !closeTargets?.others,
                  icon: 'close-all',
                  label: t.zones.closeOthers,
                  onSelect: () => {
                    triggerHaptic('selection')
                    // Persist-close session tiles before dismissing the
                    // other tree panes, or Bot Mode rehydrates them from the
                    // shared tile bucket — the same gap Close All had (#94137).
                    closeOpenSessionTilesForPanes(closeableTreeSiblings(tabPaneId).others)
                    closeOtherTreeTabs(tabPaneId)
                  }
                }),
                spec({
                  disabled: !closeTargets?.right,
                  icon: 'arrow-right',
                  label: t.zones.closeToRight,
                  onSelect: () => {
                    triggerHaptic('selection')
                    // Persist-close session tiles before dismissing the tree
                    // panes to the right, or Bot Mode rehydrates them from the
                    // shared tile bucket — the same gap Close All had (#94137).
                    closeOpenSessionTilesForPanes(closeableTreeSiblings(tabPaneId).right)
                    closeTreeTabsToRight(tabPaneId)
                  }
                }),
                spec({
                  disabled: !closeTargets?.all,
                  icon: 'clear-all',
                  label: t.zones.closeAll,
                  onSelect: () => {
                    triggerHaptic('selection')
                    // Persist-close session tiles before dismissing the
                    // remaining tree panes, or Bot Mode rehydrates them
                    // from the shared tile bucket (#94137).
                    closeAllOpenSessionTiles(tabPaneId)
                    closeAllTreeTabs(tabPaneId)
                  }
                })
              ]
            : [])
        ]
      : []

  // DANGER — put it away / destroy it (delete stays last, destructive-red).
  const dangerItems: ActionItemSpec[] = [
    spec({
      disabled: !onArchive,
      icon: 'archive',
      label: r.archive,
      onSelect: () => {
        triggerHaptic('selection')
        onArchive?.()
      }
    }),
    {
      className: 'text-destructive focus:text-destructive',
      disabled: !onDelete,
      icon: 'trash',
      label: t.common.delete,
      onSelect: () => {
        triggerHaptic('warning')

        // Deleting is irreversible (the CLI path asks y/N; the desktop used to
        // fire instantly on click). Gate it behind an explicit confirm — see
        // #61470. The dialog owns the delete call, so every surface that routes
        // through this menu (sidebar rows, tab menus, the chat header) gets the
        // guard for free.
        if (onDelete) {
          setDeleteOpen(true)
        }
      },
      variant: 'destructive'
    }
  ]

  const renderItems = (kit: MenuKit) => (
    <>
      {openItems.map(item => renderActionItem(kit, item))}
      {openItems.length > 0 && <kit.Separator />}
      {identityItems.map(item => renderActionItem(kit, item))}
      <kit.Sub>
        <kit.SubTrigger disabled={!sessionId}>
          <Codicon name="symbol-color" size="0.875rem" />
          <span>{t.sidebar.projects.menuAppearance}</span>
        </kit.SubTrigger>
        <kit.SubContent className="p-2">
          <SessionColorSwatches sessionId={sessionId} />
        </kit.SubContent>
      </kit.Sub>
      <CopyButton
        appearance={kit.copyAppearance}
        disabled={!sessionId}
        errorMessage={r.copyIdFailed}
        iconClassName="size-3.5 text-current"
        key={r.copyId}
        label={r.copyId}
        onCopyError={err => notifyError(err, r.copyIdFailed)}
        text={sessionId}
      />
      <kit.Separator />
      {workItems.map(item => renderActionItem(kit, item))}
      <kit.Sub>
        <kit.SubTrigger disabled={!sessionId}>
          <Codicon name="folder" size="0.875rem" />
          <span>{t.sidebar.projects.moveToProject}</span>
        </kit.SubTrigger>
        <kit.SubContent>
          <MoveToProjectItems kit={kit} profile={profile} sessionId={sessionId} />
        </kit.SubContent>
      </kit.Sub>
      {tabItems.length > 0 && (
        <>
          <kit.Separator />
          {tabItems.map(item => renderActionItem(kit, item))}
        </>
      )}
      <kit.Separator />
      {dangerItems.map(item => renderActionItem(kit, item))}
      {onHideTabBar && (
        <>
          <kit.Separator />
          {renderActionItem(kit, {
            disabled: false,
            icon: 'eye-closed',
            label: r.hideTabBar,
            onSelect: () => {
              triggerHaptic('selection')
              onHideTabBar()
            }
          })}
        </>
      )}
    </>
  )

  const renameDialog = (
    <RenameSessionDialog
      currentTitle={title}
      onOpenChange={setRenameOpen}
      open={renameOpen}
      profile={profile}
      sessionId={sessionId}
    />
  )

  // Consumed once per close: when rename was the action that closed the menu,
  // block Radix's focus-restore to the trigger so the dialog input keeps focus.
  const onCloseAutoFocus = (event: Event) => {
    if (suppressCloseFocusRef.current) {
      suppressCloseFocusRef.current = false
      event.preventDefault()
    }
  }

  const deleteDialog = (
    <DeleteSessionDialog
      onConfirm={() => {
        onDelete?.()
      }}
      onOpenChange={setDeleteOpen}
      open={deleteOpen}
      sessionTitle={title}
    />
  )

  return { deleteDialog, onCloseAutoFocus, renameDialog, renderItems }
}

interface DeleteSessionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: () => void
  sessionTitle: string
}

// Thin wrapper over ConfirmDialog — the single choke point for every session
// delete entry point (sidebar rows, tab menus, the chat header). Deleting a
// session is irreversible and the desktop used to fire it instantly on click
// (#61470); this mirrors the CLI's y/N guard. onConfirm is the fire-and-forget
// delete call; ConfirmDialog owns the busy/done beat and Enter-to-confirm.
function DeleteSessionDialog({ open, onOpenChange, onConfirm, sessionTitle }: DeleteSessionDialogProps) {
  const { t } = useI18n()
  const r = t.sidebar.row

  return (
    <ConfirmDialog
      busyLabel={r.deleting}
      confirmLabel={t.common.delete}
      description={r.deleteDesc(sessionTitle)}
      destructive
      doneLabel={r.deleted}
      onClose={() => onOpenChange(false)}
      onConfirm={onConfirm}
      open={open}
      title={r.deleteTitle}
    />
  )
}

interface SessionActionsMenuProps
  extends SessionActions, Pick<React.ComponentProps<typeof ActionsMenu>, 'align' | 'sideOffset'> {
  children: React.ReactNode
}

export function SessionActionsMenu({ children, align = 'end', sideOffset = 6, ...actions }: SessionActionsMenuProps) {
  const { t } = useI18n()
  const { deleteDialog, onCloseAutoFocus, renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <ActionsMenu
        align={align}
        ariaLabel={t.sidebar.row.sessionActions}
        contentClassName="w-40"
        items={renderItems}
        onCloseAutoFocus={onCloseAutoFocus}
        sideOffset={sideOffset}
      >
        {children}
      </ActionsMenu>
      {renameDialog}
      {deleteDialog}
    </>
  )
}

interface SessionContextMenuProps extends SessionActions {
  children: React.ReactNode
}

export function SessionContextMenu({ children, ...actions }: SessionContextMenuProps) {
  const { t } = useI18n()
  const { deleteDialog, onCloseAutoFocus, renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <ActionsContextMenu
        ariaLabel={t.sidebar.row.sessionActions}
        contentClassName="w-40"
        items={renderItems}
        onCloseAutoFocus={onCloseAutoFocus}
      >
        {children}
      </ActionsContextMenu>
      {renameDialog}
      {deleteDialog}
    </>
  )
}

interface RenameSessionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
  currentTitle: string
  profile?: string
}

function RenameSessionDialog({ open, onOpenChange, sessionId, currentTitle, profile }: RenameSessionDialogProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [value, setValue] = useState(currentTitle)
  const [submitting, setSubmitting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setValue(currentTitle)
      window.setTimeout(() => inputRef.current?.select(), 0)
    }
  }, [currentTitle, open])

  const submit = async () => {
    const next = value.trim()

    if (!sessionId || submitting) {
      return
    }

    if (next === currentTitle.trim()) {
      onOpenChange(false)

      return
    }

    setSubmitting(true)

    try {
      const result = await renameSessionPreferringRpc(sessionId, next, profile)
      const finalTitle = result.title || next || ''
      setSessions(prev => prev.map(s => (s.id === sessionId ? { ...s, title: finalTitle || null } : s)))
      notify({ durationMs: 2_000, kind: 'success', message: r.renamed })
      onOpenChange(false)
    } catch (err) {
      notifyError(err, r.renameFailed)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{r.renameTitle}</DialogTitle>
        </DialogHeader>
        <Input
          autoFocus
          disabled={submitting}
          onChange={event => setValue(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
              event.preventDefault()
              void submit()
            } else if (event.key === 'Escape') {
              onOpenChange(false)
            }
          }}
          placeholder={r.untitledPlaceholder}
          ref={inputRef}
          value={value}
        />
        <DialogFooter>
          <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={submitting} onClick={() => void submit()} type="button">
            {t.common.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
