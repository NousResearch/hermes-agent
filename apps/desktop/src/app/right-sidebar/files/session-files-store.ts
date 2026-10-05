// Live view of the files the focused chat created or edited, for the file
// tree's "This session" markers and filter (#133481). Mirrors the Git
// decoration stores in `store/coding-status.ts`: one shared derived map, plus a
// per-row boolean store so a row only re-renders when its own answer flips.

import { computed, type ReadableAtom } from 'nanostores'

import type { ChatMessage } from '@/lib/chat-messages'
import { $activeSessionId, $messages } from '@/store/session'
import { $focusedRuntimeId, $focusedWorkspaceCwd, $sessionStates } from '@/store/session-states'

import { deriveSessionFiles, sessionFileKey } from './session-files'

const NO_MESSAGES: readonly ChatMessage[] = []

/**
 * The messages of the chat the file tree belongs to. The tree shows
 * `$focusedWorkspaceCwd`, which follows the focused tile when one is focused
 * and the main chat otherwise; the markers must come from that same chat, or a
 * focused tile would show the main chat's files in its own folder.
 *
 * The main chat's transcript may live only in `$messages` (the same dual-store
 * read as `reloadFromMessage`); a tile never falls back to it.
 */
export function focusedChatMessages(
  focusedRuntimeId: null | string,
  primaryRuntimeId: null | string,
  states: Readonly<Record<string, { messages?: readonly ChatMessage[] } | undefined>>,
  primaryMessages: readonly ChatMessage[]
): readonly ChatMessage[] {
  if (!focusedRuntimeId) {
    return NO_MESSAGES
  }

  const own = states[focusedRuntimeId]?.messages

  if (own) {
    return own
  }

  return focusedRuntimeId === primaryRuntimeId ? primaryMessages : NO_MESSAGES
}

// Returns the same array until the focused chat's messages change, so a
// stream in another session does not re-derive this chat's files.
const $focusedMessages = computed([$focusedRuntimeId, $activeSessionId, $sessionStates, $messages], focusedChatMessages)

/** Comparison key -> path as reported, for the focused chat. */
export const $sessionFiles = computed([$focusedMessages, $focusedWorkspaceCwd], (messages, cwd) =>
  deriveSessionFiles(messages, cwd || undefined)
)

/** Per-row subscription: true when the focused chat created or edited `path`. */
export function sessionFileTouchedForPath(path: string): ReadableAtom<boolean> {
  const key = sessionFileKey(path)

  return computed($sessionFiles, files => files.has(key))
}
