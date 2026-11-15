import type { SerializeAddon } from '@xterm/addon-serialize'

import { cleanReviveSnapshot, isIdlePromptOnly, PERSISTENT_SESSION_SCROLLBACK } from './revive-snapshot'
import { updateTerminalReviveBuffer } from './terminals'

// Leading-edge throttle window for capturing history. The first output after an
// idle gap persists almost immediately (so `cmd; quit` is on disk before the
// renderer tears down), then at most once per window while output streams.
const SNAPSHOT_THROTTLE_MS = 750

interface SnapshotPersisterOptions {
  hasSessionActivity: () => boolean
  isDisposed: () => boolean
  /** Server-owned shells replay their own history; nothing to persist here. */
  isPersistent: () => boolean
  /** Refresh the persisted cwd after each capture (throttled by the tracker). */
  probeCwd: () => void
  /** The revive buffer replayed at mount; cleared on an idle session when it holds only prompts. */
  reviveBuffer: string
  serialize: SerializeAddon
}

// Capture the buffer on a leading-edge throttle and persist synchronously via
// the store. No unload hook: by the time the user quits, a recent snapshot is
// already on disk (the prior beforeunload-based attempt lost the last output).
export function createSnapshotPersister(
  id: string,
  { hasSessionActivity, isDisposed, isPersistent, probeCwd, reviveBuffer, serialize }: SnapshotPersisterOptions
): { cancel: () => void; schedule: () => void } {
  let snapshotTimer = 0
  let lastSnapshotAt = 0

  const persistSnapshot = () => {
    if (isDisposed() || isPersistent()) {
      return
    }

    lastSnapshotAt = Date.now()

    // No user input this session: never re-serialize. The live buffer now holds
    // the replayed history plus a fresh boot prompt, and re-saving that is
    // exactly what grew idle tabs by one prompt line per relaunch (#61572).
    // If the buffer we loaded carried no real scrollback (empty, or only a
    // repeated prompt), clear it so the next launch shows a single fresh prompt
    // and any pre-existing accumulation heals. Otherwise leave the prior
    // snapshot untouched so real history from an earlier active session
    // survives an idle reopen instead of being overwritten.
    if (!hasSessionActivity()) {
      if (isIdlePromptOnly(reviveBuffer)) {
        updateTerminalReviveBuffer(id, '')
      }

      return
    }

    try {
      const snapshot = serialize.serialize({ scrollback: PERSISTENT_SESSION_SCROLLBACK })
      updateTerminalReviveBuffer(id, cleanReviveSnapshot(snapshot))
    } catch {
      // Best-effort restore: never let serialization break a live terminal.
    }

    // A user command may have `cd`'d; refresh the persisted cwd (throttled).
    probeCwd()
  }

  const scheduleSnapshot = () => {
    if (snapshotTimer) {
      return
    }

    const elapsed = Date.now() - lastSnapshotAt

    if (elapsed >= SNAPSHOT_THROTTLE_MS) {
      persistSnapshot()

      return
    }

    snapshotTimer = window.setTimeout(() => {
      snapshotTimer = 0
      persistSnapshot()
    }, SNAPSHOT_THROTTLE_MS - elapsed)
  }

  return {
    cancel: () => {
      if (snapshotTimer) {
        window.clearTimeout(snapshotTimer)
      }
    },
    schedule: scheduleSnapshot
  }
}
