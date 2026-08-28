/**
 * The window-local state a group-chat room needs but never renders: the
 * per-composer drafts, and the registry of MAIN-window tabs a room is open in.
 *
 * A leaf by design. The room surface and the roster pane both read it and it
 * reads neither, so a disband, a rename or a Back button can retire a room's
 * tab without any of those paths importing a view.
 */

import { atom } from '@hermes/plugin-sdk'

import { $groupChatWorkspace, groupChatRoomKey } from './group-chat'
import type { GroupChatRoom } from './group-chat'
import type { Attachment } from './types'

// Group composer drafts are window-local UI state. They must survive pane
// parking/re-registration and owner switches, but must never enter shared room
// metadata (where another Desktop would see half-typed text or attachment
// bytes). Current rooms key by immutable roomId; legacy rooms fall back to the
// display name until they are upgraded.
export interface GroupComposerDraft {
  activeReplyThread: null | string
  main: string
  /** Keyed by thread id; the main composer parks under 'main'. */
  pendingAttachments: Record<string, Attachment[]>
  replies: Record<string, string>
  revision: number
}
const groupComposerDrafts = new Map<string, GroupComposerDraft | undefined>()

function emptyGroupComposerDraft(): GroupComposerDraft {
  return {
    activeReplyThread: null,
    main: '',
    pendingAttachments: {},
    replies: {},
    revision: 0
  }
}

export function groupComposerDraftKey(group: string, room: GroupChatRoom): string {
  return groupChatRoomKey(group, room)
}

export function groupComposerDraftSnapshot(key: string): GroupComposerDraft {
  return groupComposerDrafts.get(key) || emptyGroupComposerDraft()
}

export function updateGroupComposerDraft(key: string, mutate: (draft: GroupComposerDraft) => GroupComposerDraft) {
  const current = groupComposerDraftSnapshot(key)

  const next = mutate({
    ...current,
    pendingAttachments: Object.fromEntries(
      Object.entries(current.pendingAttachments || {}).map(([thread, attachments]) => [
        thread,
        [...(attachments || [])]
      ])
    ),
    replies: {
      ...(current.replies || {})
    }
  })

  next.revision = current.revision + 1
  groupComposerDrafts.delete(key)
  groupComposerDrafts.set(key, next)

  return next
}

export function restoreGroupComposerDraft(key: string, expectedRevision: number, snapshot: GroupComposerDraft) {
  const current = groupComposerDraftSnapshot(key)

  if (current.revision !== expectedRevision) {
    return null
  }

  const restored = {
    ...snapshot,
    pendingAttachments: Object.fromEntries(
      Object.entries(snapshot.pendingAttachments || {}).map(([thread, attachments]) => [
        thread,
        [...(attachments || [])]
      ])
    ),
    replies: {
      ...(snapshot.replies || {})
    },
    revision: current.revision + 1
  }

  groupComposerDrafts.set(key, restored)

  return restored
}

export function clearGroupComposerDraft(key: string) {
  groupComposerDrafts.delete(key)
}

export function migrateGroupComposerDraft(oldKey: string, newKey: string) {
  if (oldKey === newKey || !groupComposerDrafts.has(oldKey)) {
    return
  }

  if (!groupComposerDrafts.has(newKey)) {
    groupComposerDrafts.set(newKey, groupComposerDrafts.get(oldKey))
  }

  groupComposerDrafts.delete(oldKey)
}

/** React-style setter argument — the next value, or a function of the current. */
export type GroupDraftSetter<T> = T | ((current: T) => T)

/** Live closers for group-chat MAIN-window tabs, by group name — so a
 *  disband (or the room view's own Back) can retire the tab it opened. */
export const groupChatMainTabs = new Map<string, () => void>()

/** Reactive shadow of `groupChatMainTabs` membership. The Map itself can't
 *  notify React, and #89788's first fix read it non-reactively: a BotsPane
 *  render that landed between selecting the group and recording its main
 *  tab kept the in-pane room on screen forever (the map write repaints
 *  nothing). Every map mutation goes through the two helpers below so the
 *  rev bump re-evaluates the gate. */
export const $groupMainTabsRev = atom(0)

export function recordGroupMainTab(group: string, close: () => void) {
  groupChatMainTabs.set(group, close)
  $groupMainTabsRev.set($groupMainTabsRev.get() + 1)
}

export function dropGroupMainTab(group: string) {
  if (groupChatMainTabs.delete(group)) {
    $groupMainTabsRev.set($groupMainTabsRev.get() + 1)
  }
}

/** The in-panel room is the FALLBACK surface, not a second copy: it renders
 *  only while no main-window tab owns the group. On desktops with the door
 *  the room already lives in a main tab, and painting it here too produced
 *  two live panes with independent drafts driving one shared engine (#89788).
 *  The selection atom stays set either way so the roster row still
 *  highlights. Callers must subscribe to `$groupMainTabsRev` (BotsPane does)
 *  so ownership changes re-run this gate. */
export function shouldRenderGroupChatInPane(group: null | string): group is string {
  return Boolean(group && !groupChatMainTabs.has(group))
}

export function closeGroupChatMainTab(group: string) {
  const close = groupChatMainTabs.get(group)
  dropGroupMainTab(group)

  if ($groupChatWorkspace.get() === group) {
    $groupChatWorkspace.set(null)
  }

  if (typeof close === 'function') {
    try {
      close()
    } catch {
      /* tab already gone */
    }
  }
}
