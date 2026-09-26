/**
 * SESSION FOLDERS — the sidebar's half of `store/session-sections`.
 *
 * The store owns the model: which folders exist, which session is filed where,
 * and what a delete's undo puts back. This file owns the chrome — the foldable
 * folder heading with its ⋯ menu, the ONE name dialog behind both New folder
 * and Rename, and the drop handling — and nothing here re-decides any of it.
 *
 * Three invariants worth keeping:
 *
 *  * ONE drag context. The folder list renders a single `ReorderableList`, so a
 *    row can be dragged inside its folder (the list's own reorder path persists
 *    that) AND onto another folder's heading (filed by `moveSessionsToSection`).
 *    A DndContext per folder could never see the cross-folder drop, and the
 *    sidebar already runs exactly one of them for this list.
 *  * Rows arrive sorted. The caller hands over the flat list's own rows in the
 *    flat list's own order (see sessions-section.tsx) — this file never sorts,
 *    ranks, or invents a row.
 *  * Unassigned is not a folder. It borrows the section's own label, has no
 *    record to rename or delete, and is drawn last.
 */

import type { DragEndEvent, DragStartEvent, useSensors } from '@dnd-kit/core'
import { useDndMonitor, useDroppable } from '@dnd-kit/core'
import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import type * as React from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { ActionsMenu, renderActionItem } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { Input } from '@/components/ui/input'
import { type Translations, useI18n } from '@/i18n'
import { isSubmitEnter } from '@/lib/ime'
import { cn } from '@/lib/utils'
import { notify } from '@/store/notifications'
import {
  $draggingSession,
  createSessionSection,
  deleteSessionSection,
  moveSessionsToSection,
  renameSessionSection,
  sessionSectionId,
  UNASSIGNED_SESSION_KEY
} from '@/store/session-sections'

import { SidebarSectionAddButton } from './chrome'
import { useWorkspaceNodeOpen } from './projects/model'
import { ReorderableList } from './reorderable-list'
import { SIDEBAR_ROW_LABEL } from './row-geometry'

// ── drop targets ─────────────────────────────────────────────────────────────
//
// A folder heading is a drop target with an id of its own, prefixed so a
// finished drag can tell "a folder heading" from "a session row" without a
// registry to look ids up in. The same id is the heading's collapse key: both
// name the one thing a heading IS (a folder, or the Unassigned bucket).

const SESSION_FOLDER_DROP_PREFIX = 'session-folder:'

/** The drop-target (and collapse-state) id for a folder heading. `null` is the
 *  Unassigned bucket, which is a target — dropping there un-files a session —
 *  but never a folder. */
export function sessionFolderDropId(sectionId: null | string): string {
  return `${SESSION_FOLDER_DROP_PREFIX}${sectionId ?? UNASSIGNED_SESSION_KEY}`
}

/** The folder a drop target names, or undefined when the id is not a folder
 *  heading at all (a session row, or something outside this list). */
export function sessionFolderDropTarget(overId: unknown): undefined | { sectionId: null | string } {
  const key = String(overId ?? '')

  if (!key.startsWith(SESSION_FOLDER_DROP_PREFIX)) {
    return undefined
  }

  const rest = key.slice(SESSION_FOLDER_DROP_PREFIX.length)

  return { sectionId: rest === UNASSIGNED_SESSION_KEY ? null : rest }
}

/**
 * A finished drag. Dropping on a folder heading files the dragged session into
 * that folder; dropping on the Unassigned heading un-files it; dropping on a
 * ROW of another folder is the same intent (that row IS that folder) and files
 * it there too. A drop on a row of its own folder is an ordinary reorder, which
 * the list's own handler already persisted — this does nothing.
 *
 * Exported because it is the one place a folder drop is interpreted: jsdom has
 * no layout for a sensor gesture to resolve a target against, so the tests
 * drive THIS handler with the ids a drop reports.
 */
export function fileSessionFromDrop(activeId: string, overId: unknown, rows: ReadonlySet<string>): void {
  const session = String(activeId ?? '').trim()

  if (!session) {
    return
  }

  const heading = sessionFolderDropTarget(overId)
  const overRowId = String(overId ?? '')

  // Not a heading and not one of our rows: nothing here owns that drop.
  if (!heading && !rows.has(overRowId)) {
    return
  }

  const target = heading ? heading.sectionId : sessionSectionId(overRowId)

  // Already where it was dropped (the heading sits right above its own rows,
  // so this is the common miss): a no-op write is not worth the churn.
  if (sessionSectionId(session) === target) {
    return
  }

  moveSessionsToSection([session], target)
}

/** The drag lifecycle for the folder list. `$draggingSession` is set for the
 *  duration of the gesture and cleared on EVERY ending — drop, cancel, or a
 *  release over nothing — because a stuck "dragging" state outlives the gesture
 *  and reads as a broken list. Mounted inside the list's DndContext, which is
 *  the only place a drag is visible from. */
function SessionFolderDragMonitor({ rows }: { rows: ReadonlySet<string> }) {
  const listener = useMemo(
    () => ({
      onDragCancel: () => $draggingSession.set(null),
      onDragEnd: ({ active, over }: DragEndEvent) => {
        $draggingSession.set(null)

        if (over) {
          fileSessionFromDrop(String(active.id), over.id, rows)
        }
      },
      onDragStart: ({ active }: DragStartEvent) => $draggingSession.set(String(active.id))
    }),
    [rows]
  )

  useDndMonitor(listener)

  return null
}

// ── name dialog ──────────────────────────────────────────────────────────────

/** The one name dialog's state: New folder, or Rename. Module-level because
 *  the "+" lives in the section header and Rename lives in a folder's ⋯ menu —
 *  one dialog, mounted once, serves both (the same shape the Bot roster's
 *  section dialog uses). */
export type SessionFolderDialogState = null | { mode: 'create' } | { id: string; mode: 'rename'; name: string }

export const $sessionFolderDialog = atom<SessionFolderDialogState>(null)

/** New folder and Rename in one small dialog — the app renames sessions through
 *  the same Dialog + Input + Cancel/Save shape, so a folder rename feels like
 *  every other rename. */
export function SessionFolderDialog() {
  const { t } = useI18n()
  const f = t.sidebar.folders
  const state = useStore($sessionFolderDialog)
  const open = state !== null
  const renaming = state?.mode === 'rename'
  const [name, setName] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setName(renaming && state?.mode === 'rename' ? state.name : '')
      window.setTimeout(() => inputRef.current?.select(), 0)
    }
  }, [open, renaming, state])

  const close = () => $sessionFolderDialog.set(null)

  const submit = () => {
    const clean = name.trim()

    if (!clean) {
      return
    }

    if (state?.mode === 'rename') {
      // A rename to the same name is not a write: the store trims and stores
      // it, and nothing about the folder would change.
      if (clean !== state.name) {
        renameSessionSection(state.id, clean)
      }
    } else {
      createSessionSection(clean)
    }

    close()
  }

  return (
    <Dialog onOpenChange={next => !next && close()} open={open}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>{renaming ? f.renameTitle : f.newFolder}</DialogTitle>
        </DialogHeader>
        <Input
          aria-label={f.nameLabel}
          autoFocus
          maxLength={40}
          onChange={event => setName(event.target.value)}
          onKeyDown={event => {
            if (isSubmitEnter(event)) {
              event.preventDefault()
              submit()
            }
          }}
          placeholder={f.namePlaceholder}
          ref={inputRef}
          value={name}
        />
        <DialogFooter>
          <Button onClick={close} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={!name.trim()} onClick={submit} type="button">
            {renaming ? t.common.save : f.create}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ── headings ─────────────────────────────────────────────────────────────────

/** The Sessions header's "New folder" affordance, beside the new-session "+". */
export function SessionFolderAddButton() {
  const { t } = useI18n()

  return (
    <SidebarSectionAddButton
      ariaLabel={t.sidebar.folders.newFolder}
      icon="new-folder"
      onPlainClick={() => $sessionFolderDialog.set({ mode: 'create' })}
    />
  )
}

interface SessionFolderHeaderProps {
  /** dnd-kit's droppable ref — present only for headings that accept a drop. */
  dropRef?: (node: HTMLElement | null) => void
  /** True while a dragged row is over this heading. */
  isOver?: boolean
  label: string
  /** The ⋯ menu; absent for Unassigned, which has no record to act on. */
  menu?: React.ReactNode
  onToggle: () => void
  open: boolean
}

function SessionFolderHeader({ dropRef, isOver, label, menu, onToggle, open }: SessionFolderHeaderProps) {
  const { t } = useI18n()
  // While a session is in flight every heading says it will take it; the one
  // the pointer is actually over says it louder.
  const dragging = useStore($draggingSession)

  return (
    <div
      className={cn(
        'group/folder flex min-h-6 items-center gap-1 rounded-md px-2 pt-1 text-[0.6875rem] font-medium text-(--ui-text-tertiary)',
        // The heading IS the drop target, so it lights up under the dragged
        // row the way the roster's section zones do.
        dragging && 'ring-1 ring-inset ring-(--ui-stroke-secondary)',
        isOver && 'bg-(--ui-accent)/10 ring-(--ui-accent)'
      )}
      data-session-folder-header=""
      ref={dropRef}
    >
      <button
        aria-expanded={open}
        aria-label={t.sidebar.folders.toggle(label, !open)}
        className="flex min-w-0 flex-1 items-center gap-1.5 bg-transparent text-left hover:text-(--ui-text-secondary)"
        onClick={onToggle}
        type="button"
      >
        <DisclosureCaret className="shrink-0" open={open} />
        <Codicon className="shrink-0" name="folder" size="0.75rem" />
        <span className={cn(SIDEBAR_ROW_LABEL, 'truncate')}>{label}</span>
      </button>
      {menu}
    </div>
  )
}

/** A heading registered as a drop target for its folder (or for Unassigned). */
function DroppableSessionFolderHeader({
  sectionId,
  ...header
}: Omit<SessionFolderHeaderProps, 'dropRef' | 'isOver'> & { sectionId: null | string }) {
  const { isOver, setNodeRef } = useDroppable({ id: sessionFolderDropId(sectionId) })

  return <SessionFolderHeader {...header} dropRef={setNodeRef} isOver={isOver} />
}

/** Delete the folder only — never its sessions. They fall back to Unassigned,
 *  and the toast's Undo puts the folder back in its original slot with the same
 *  sessions refiled, which is why this needs no confirmation. */
function removeSessionFolder(id: string, name: string, t: Translations) {
  const { undo } = deleteSessionSection(id)

  notify({
    action: { label: t.sidebar.folders.undo, onClick: undo },
    durationMs: 8_000,
    kind: 'info',
    message: t.sidebar.folders.deleted(name)
  })
}

function SessionFolderMenu({ id, name }: { id: string; name: string }) {
  const { t } = useI18n()
  const f = t.sidebar.folders

  return (
    <ActionsMenu
      ariaLabel={f.menu}
      contentClassName="w-48"
      items={kit => (
        <>
          {renderActionItem(kit, {
            icon: 'edit',
            key: 'rename',
            label: f.menuRename,
            onSelect: () => $sessionFolderDialog.set({ id, mode: 'rename', name })
          })}
          {renderActionItem(kit, {
            icon: 'trash',
            key: 'delete',
            label: t.common.delete,
            onSelect: () => removeSessionFolder(id, name, t),
            variant: 'destructive'
          })}
        </>
      )}
    >
      <button
        aria-label={f.menu}
        className="grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/folder:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100"
        type="button"
      >
        <Codicon name="kebab-vertical" size="0.75rem" />
      </button>
    </ActionsMenu>
  )
}

// ── the list ─────────────────────────────────────────────────────────────────

/** One row cluster: the root session row the section rendered, plus any branch
 *  children that hang under it. A branch child is never split from its parent,
 *  so the cluster — not the individual row — is what a folder holds. */
export interface SessionFolderRow {
  id: string
  nodes: React.ReactNode[]
}

/** A folder's rows, or the trailing Unassigned bucket's — the shape
 *  `groupSessionsBySection` returns. */
export interface SessionFolderGroup {
  id: null | string
  name: null | string
  rows: SessionFolderRow[]
}

interface SessionFolderBlockProps {
  group: SessionFolderGroup
  /** Unassigned borrows the section's own label: it is not a folder. */
  unassignedLabel: string
}

function SessionFolderBlock({ group, unassignedLabel }: SessionFolderBlockProps) {
  // Folded state lives in the same persisted map the lanes and date buckets
  // use, keyed by the folder's own id — it survives a restart, and nothing has
  // to thread it down here.
  const [open, toggleOpen] = useWorkspaceNodeOpen(sessionFolderDropId(group.id), true)
  const label = group.name ?? unassignedLabel

  return (
    // The block is its own row stack, so a folder's rows keep the flat list's
    // row rhythm without inheriting the outer list's dividers.
    <div
      className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-px"
      data-session-folder={group.id ?? UNASSIGNED_SESSION_KEY}
    >
      <DroppableSessionFolderHeader
        label={label}
        menu={group.id ? <SessionFolderMenu id={group.id} name={group.name ?? ''} /> : undefined}
        onToggle={toggleOpen}
        open={open}
        sectionId={group.id}
      />
      {open && group.rows.flatMap(row => row.nodes)}
    </div>
  )
}

interface SidebarSessionFolderListProps {
  /** Folders in display order, each with its rows, plus the trailing Unassigned
   *  bucket. */
  groups: SessionFolderGroup[]
  /** The Unassigned bucket's label — the section's own. */
  label: string
  onReorder?: (ids: string[]) => void
  sensors?: ReturnType<typeof useSensors>
}

/**
 * The flat list, filed. One ReorderableList wraps every folder, so the sidebar's
 * existing drag context carries both gestures: reordering a row within its
 * folder, and dropping one on a heading to file it.
 */
export function SidebarSessionFolderList({ groups, label, onReorder, sensors }: SidebarSessionFolderListProps) {
  // dnd-kit must see exactly the ids rendered, in render order (the same rule
  // the flat list follows): the folder bodies are what is on screen.
  const ids = useMemo(() => groups.flatMap(group => group.rows.map(row => row.id)), [groups])
  const rows = useMemo(() => new Set(ids), [ids])

  return (
    <ReorderableList ids={ids} onReorder={next => onReorder?.(next)} sensors={sensors}>
      <SessionFolderDragMonitor rows={rows} />
      {groups.map(group => (
        <SessionFolderBlock group={group} key={sessionFolderDropId(group.id)} unassignedLabel={label} />
      ))}
    </ReorderableList>
  )
}
