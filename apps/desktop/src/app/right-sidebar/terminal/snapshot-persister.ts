import type { SerializeAddon } from '@xterm/addon-serialize'
import type { IMarker, Terminal } from '@xterm/xterm'

import { mergeReviveSnapshot, PERSISTENT_SESSION_SCROLLBACK, resolveLiveSnapshotWindow } from './revive-snapshot'
import { updateTerminalReviveBuffer } from './terminals'

// Leading-edge throttle window for capturing history. The first output after an
// idle gap persists almost immediately (so `cmd; quit` is on disk before the
// renderer tears down), then at most once per window while output streams.
const SNAPSHOT_THROTTLE_MS = 750

interface SnapshotPersisterOptions {
  /** Boundary between replayed history and this session's PTY output. */
  getLiveStartMarker: () => IMarker | undefined
  getShellName: () => string
  hasSessionActivity: () => boolean
  isDisposed: () => boolean
  /** Server-owned shells replay their own history; nothing to persist here. */
  isPersistent: () => boolean
  /** Refresh the persisted cwd after each capture (throttled by the tracker). */
  probeCwd: () => void
  /** The revive buffer replayed at mount, kept ahead of live output byte-for-byte. */
  reviveBuffer: string
  serialize: SerializeAddon
  term: Terminal
}

// Capture the buffer on a leading-edge throttle and persist synchronously via
// the store. No unload hook: by the time the user quits, a recent snapshot is
// already on disk (the prior beforeunload-based attempt lost the last output).
export function createSnapshotPersister(
  id: string,
  {
    getLiveStartMarker,
    getShellName,
    hasSessionActivity,
    isDisposed,
    isPersistent,
    probeCwd,
    reviveBuffer,
    serialize,
    term
  }: SnapshotPersisterOptions
): { cancel: () => void; schedule: () => void } {
  let snapshotTimer = 0
  let lastSnapshotAt = 0

  const persistSnapshot = () => {
    if (isDisposed() || isPersistent()) {
      return
    }

    lastSnapshotAt = Date.now()

    // No user input this session: never re-serialize. The live buffer now holds
    // replayed history plus fresh boot output, and re-saving that is exactly
    // what grew idle tabs by one prompt per relaunch (#61572). Preserve the
    // prior snapshot byte-for-byte; legacy text is ambiguous and must not be
    // auto-deleted merely because it resembles a prompt.
    if (!hasSessionActivity()) {
      return
    }

    try {
      const liveStartMarker = getLiveStartMarker()

      if (term.buffer.active.type !== 'normal' || !liveStartMarker) {
        return
      }

      const normal = term.buffer.normal
      const cursorLine = normal.baseY + normal.cursorY
      let lastContentLine = normal.length - 1

      while (lastContentLine > cursorLine && !normal.getLine(lastContentLine)?.translateToString(true)) {
        lastContentLine -= 1
      }

      // `normal.length` includes blank viewport rows below the cursor. A range
      // budget based on that capacity can start after the cursor and serialize
      // nothing (for example after a tall resize). The cursor is the live end;
      // if real content exists below it, provenance is uncertain and falls back.
      const end = cursorLine

      const liveWindow = resolveLiveSnapshotWindow(
        liveStartMarker.line,
        end,
        cursorLine,
        PERSISTENT_SESSION_SCROLLBACK,
        term.markers.includes(liveStartMarker) && lastContentLine <= cursorLine
      )

      let restored = reviveBuffer
      let live: string
      let nextSnapshot: string

      if (liveWindow) {
        // Once live output alone exceeds the replay budget, the restored prefix
        // has scrolled out and should no longer be carried into future sessions.
        if (!liveWindow.keepRestored) {
          restored = ''
        }

        live = serialize.serialize({ excludeAltBuffer: true, range: { end, start: liveWindow.start } })
        nextSnapshot = mergeReviveSnapshot(restored, live, getShellName(), liveWindow.keepRestored)
      } else {
        // A reset, clear-screen, or scrollback trim can invalidate the logical
        // boundary. The current normal buffer is then authoritative, but its
        // restored/live provenance is unknown, so persist it without text-based
        // greeting or prompt cleanup rather than risk deleting real output.
        live = serialize.serialize({ excludeAltBuffer: true, scrollback: PERSISTENT_SESSION_SCROLLBACK })
        nextSnapshot = live
      }

      updateTerminalReviveBuffer(id, nextSnapshot)
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
