// Live view of the files the active session created or edited, for the file
// tree's "This session" markers and filter (#133481). Mirrors the Git
// decoration stores in `store/coding-status.ts`: one shared derived map, plus a
// per-row boolean store so a row only re-renders when its own answer flips.

import { computed, type ReadableAtom } from 'nanostores'

import { $messages } from '@/store/session'
import { $focusedWorkspaceCwd } from '@/store/session-states'

import { deriveSessionFiles, sessionFileKey } from './session-files'

/** Comparison key -> path as reported, for the active session. */
export const $sessionFiles = computed([$messages, $focusedWorkspaceCwd], (messages, cwd) =>
  deriveSessionFiles(messages, cwd || undefined)
)

/** Per-row subscription: true when the active session created or edited `path`. */
export function sessionFileTouchedForPath(path: string): ReadableAtom<boolean> {
  const key = sessionFileKey(path)

  return computed($sessionFiles, files => files.has(key))
}
