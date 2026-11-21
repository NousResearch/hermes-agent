/**
 * Kanban terminal-event popup preferences.
 *
 * Two gates already exist for a worker event's OS notification (Settings ▸
 * Notifications ▸ "Plugin notifications", plus the shell firing it only while
 * the user is away). The in-app toast has no equivalent, and warning toasts
 * never auto-dismiss (`store/notifications.ts` returns duration 0 for
 * warning/error), so a task that blocks repeatedly leaves a stack of popups
 * over the board with nothing but the board's own blocked column to say what
 * happened.
 *
 * This module owns the plugin-level switch for those popups. It is persisted
 * through `ctx.storage`, so it survives reloads and follows the plugin scope
 * like every other kanban key.
 */

import type { PluginStorage } from '@hermes/plugin-sdk'
import { atom } from 'nanostores'

/** Terminal kinds a blocked-popup switch silences: a worker block and the
 *  block-loop handoff to triage. Both are worker-side states the board already
 *  shows in the blocked / triage columns. */
export const BLOCKED_KINDS = ['blocked', 'block_loop_detected'] as const

export interface KanbanNotifyPrefs {
  /** Popups for `blocked` and `block_loop_detected` events. On by default:
   *  a block is normally the one thing a user wants to hear about. */
  blockedPopups: boolean
}

export const DEFAULT_KANBAN_NOTIFY_PREFS: KanbanNotifyPrefs = { blockedPopups: true }

export const NOTIFY_PREFS_KEY = 'notify.prefs.v1'

export const $kanbanNotifyPrefs = atom<KanbanNotifyPrefs>(DEFAULT_KANBAN_NOTIFY_PREFS)

/** Kinds to silence for the current preference. */
export function silencedKindsFor(prefs: KanbanNotifyPrefs): readonly string[] {
  return prefs.blockedPopups ? [] : BLOCKED_KINDS
}

/** Stored values are plugin-owned but can predate a shape change, so anything
 *  that is not a boolean falls back to the default rather than silencing a
 *  popup the user never turned off. */
export function normalizeNotifyPrefs(raw: unknown): KanbanNotifyPrefs {
  if (typeof raw !== 'object' || raw === null) {
    return DEFAULT_KANBAN_NOTIFY_PREFS
  }

  const blocked = (raw as { blockedPopups?: unknown }).blockedPopups

  return { blockedPopups: typeof blocked === 'boolean' ? blocked : true }
}

/** Hydrate the atom from storage and mirror every change back. Returns the
 *  listener's unsubscribe so the caller can hand it to the plugin disposer. */
export function bindNotifyPrefs(storage: PluginStorage): () => void {
  $kanbanNotifyPrefs.set(normalizeNotifyPrefs(storage.get<unknown>(NOTIFY_PREFS_KEY, DEFAULT_KANBAN_NOTIFY_PREFS)))

  return $kanbanNotifyPrefs.listen(prefs => storage.set(NOTIFY_PREFS_KEY, prefs))
}

export function setBlockedPopups(on: boolean): void {
  $kanbanNotifyPrefs.set({ ...$kanbanNotifyPrefs.get(), blockedPopups: on })
}
