import { atom } from 'nanostores'

import { $activeSessionStoredIdRotation, type ActiveSessionStoredIdRotation } from './session'
import { $temporaryChats, $temporaryDraft, type TemporaryChat } from './temporary-chat-registry'

export {
  $temporaryChats,
  $temporaryDraft,
  $temporarySessionIds,
  isTemporarySession,
  type TemporaryChat
} from './temporary-chat-registry'

/**
 * Temporary chat (`session.create {temporary: true}`): the backend keeps the row hidden, unlisted,
 * unsearchable and memory-free, and deletes it on `session.close`. The renderer owns only the
 * window-local state below — deliberately never persisted, so nothing about a temporary chat
 * outlives the window.
 *
 * A live temporary chat is STICKY: leaving it (another chat, a page, a plugin) never deletes it —
 * a turn may still be running and the user must be able to come back and stop it. It ends when the
 * user ends it (End / ⌘W on it), when it has sat idle off-screen for TEMPORARY_CHAT_IDLE_TTL_MS
 * after its last reply, or when the app quits (backend shutdown teardown).
 */
export const TEMPORARY_CHAT_IDLE_TTL_MS = 10 * 60_000

/** Bumped by every "New temporary chat" entry point (keybind, palette, app menu); the session
 *  controller answers with a fresh temporary draft (useTemporaryChatLifecycle). */
export const $newTemporaryChatRequest = atom(0)

/** Stored ids the user asked to end; drained by useTemporaryChatLifecycle (which owns deletion). */
export const $temporaryChatEndRequests = atom<readonly string[]>([])

export function requestNewTemporaryChat(): void {
  $newTemporaryChatRequest.set($newTemporaryChatRequest.get() + 1)
}

export function setTemporaryDraft(on: boolean): void {
  $temporaryDraft.set(on)
}

export function toggleTemporaryDraft(): void {
  $temporaryDraft.set(!$temporaryDraft.get())
}

export function registerTemporaryChat(storedId: string, runtimeId: null | string = null): void {
  if (!storedId) {
    return
  }

  $temporaryChats.set({ ...$temporaryChats.get(), [storedId]: { idleSince: null, runtimeId, storedId } })
}

export function forgetTemporaryChat(storedId: string): void {
  if (!$temporaryChats.get()[storedId]) {
    return
  }

  const { [storedId]: _gone, ...rest } = $temporaryChats.get()
  $temporaryChats.set(rest)
}

/** The temporary chat actually on screen: selected AND not covered by a workspace page (plugins,
 *  skills, artifacts keep the selection while they own the workspace). */
export function temporaryChatOnScreen(selectedStoredId: null | string, workspaceIsPage: boolean): null | string {
  return selectedStoredId && !workspaceIsPage && $temporaryChats.get()[selectedStoredId] ? selectedStoredId : null
}

export function endTemporaryChat(storedId: string): void {
  if ($temporaryChats.get()[storedId] && !$temporaryChatEndRequests.get().includes(storedId)) {
    $temporaryChatEndRequests.set([...$temporaryChatEndRequests.get(), storedId])
  }
}

export function takeTemporaryChatEndRequests(): string[] {
  const pending = [...$temporaryChatEndRequests.get()]

  if (pending.length) {
    $temporaryChatEndRequests.set([])
  }

  return pending
}

/** Restart or stop each chat's idle clock: on screen or running → not idle; otherwise idle since
 *  the first time it was seen idle. Returns the same object when nothing changed. */
export function updateTemporaryIdleClocks(
  chats: Readonly<Record<string, TemporaryChat>>,
  isActive: (chat: TemporaryChat) => boolean,
  now: number
): Readonly<Record<string, TemporaryChat>> {
  let changed = false
  const next: Record<string, TemporaryChat> = {}

  for (const [id, chat] of Object.entries(chats)) {
    const idleSince = isActive(chat) ? null : (chat.idleSince ?? now)
    changed ||= idleSince !== chat.idleSince
    next[id] = idleSince === chat.idleSince ? chat : { ...chat, idleSince }
  }

  return changed ? next : chats
}

/** Stored ids whose idle clock has run out. */
export function expiredTemporaryChats(
  chats: Readonly<Record<string, TemporaryChat>>,
  now: number,
  ttlMs = TEMPORARY_CHAT_IDLE_TTL_MS
): string[] {
  return Object.values(chats)
    .filter(chat => chat.idleSince !== null && now - chat.idleSince >= ttlMs)
    .map(chat => chat.storedId)
}

/**
 * Auto-compression rotates a live chat's stored id to the new tip. The tip is the same temporary
 * chat: re-key it so its indicator, idle clock and End action follow the conversation.
 */
export function followTemporaryRotation(rotation: ActiveSessionStoredIdRotation | null): void {
  const chat = rotation ? $temporaryChats.get()[rotation.previousStoredSessionId] : undefined

  if (!rotation || !chat) {
    return
  }

  const { [rotation.previousStoredSessionId]: _old, ...rest } = $temporaryChats.get()
  $temporaryChats.set({ ...rest, [rotation.nextStoredSessionId]: { ...chat, storedId: rotation.nextStoredSessionId } })
}

// Synchronous, before React's selection effects observe the rotation.
$activeSessionStoredIdRotation.listen(followTemporaryRotation)
