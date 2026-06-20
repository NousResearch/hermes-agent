/**
 * Session restore after updates/crashes.
 *
 * The Electron main process persists open session windows on quit and emits a
 * 'session:restore-available' event on next launch when a pending snapshot
 * exists. This store holds the renderer-side copy so the prompt component can
 * render it and the desktop controller can drive confirm/discard.
 *
 * No localStorage — the main-process session-restore.json is the single
 * source of truth.
 */

import { atom } from 'nanostores'

export type SessionRestoreEntry = {
  sessionId: string
  watch?: boolean
  bounds?: { x: number; y: number; width: number; height: number }
}

export type SessionRestoreSnapshot = {
  schemaVersion: number
  createdAt: number
  entries: SessionRestoreEntry[]
}

export const $sessionRestoreSnapshot = atom<SessionRestoreSnapshot | null>(null)
export const $sessionRestorePromptVisible = atom(false)

export function showSessionRestorePrompt(snapshot: SessionRestoreSnapshot | null): void {
  $sessionRestoreSnapshot.set(snapshot)
  $sessionRestorePromptVisible.set(snapshot !== null && snapshot.entries.length > 0)
}

export function clearSessionRestorePrompt(): void {
  $sessionRestoreSnapshot.set(null)
  $sessionRestorePromptVisible.set(false)
}
