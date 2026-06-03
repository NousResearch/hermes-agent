/**
 * Desktop state persistence — durable store for renderer state.
 *
 * The Electron renderer's localStorage can be wiped on app restarts,
 * auto-updates, or when switching between dev and prod builds. This
 * module provides:
 *
 *   1. `hydrateDesktopState()` — seeds localStorage from the main-process
 *      filesystem on startup, then notifies store atoms to re-read.
 *   2. `persistDesktopState()` — explicitly persist a value (redundant when
 *      used via storage.ts persist functions, which already write through IPC).
 *
 * Each persisted key lives as a JSON file under the main process's
 * `userData/desktop-state/` directory.
 */

/** State keys that receive IPC-backed persistence. */
export const DESKTOP_STATE_KEYS = [
  'hermes.desktop.pinnedSessions',
  'hermes.desktop.paneStates.v1',
  'hermes.desktop.agentsGroupedByWorkspace',
] as const

/**
 * Seed localStorage from main-process filesystem state, then rehydrate
 * nanostore atoms that depend on these keys.
 *
 * Call once at app startup, ideally before the first render so atoms
 * initialise with correct values (start as early as possible — the calls
 * are fire-and-forget and return before module evaluation finishes in
 * practice, since IPC + local SSD is faster than JS module resolution).
 *
 * Returns a promise that resolves once all keys have been read and stores
 * notified. The caller can `await` this if they want a guaranteed-before-
 * render hydration.
 */
export async function hydrateDesktopState(): Promise<void> {
  if (typeof window === 'undefined' || !window.hermesDesktop?.getDesktopState) {
    return
  }

  let changed = false

  for (const key of DESKTOP_STATE_KEYS) {
    try {
      const value = await window.hermesDesktop.getDesktopState(key)

      if (value === null || value === undefined) {
        continue // No persisted data yet — first launch or never saved.
      }

      const stringValue = typeof value === 'string' ? value : JSON.stringify(value)
      const current = window.localStorage.getItem(key)

      if (stringValue !== current) {
        window.localStorage.setItem(key, stringValue)
        changed = true
      }
    } catch {
      // Best-effort; localStorage continues to work either way.
    }
  }

  if (changed) {
    // Dynamic import avoids circular dependencies at module evaluation
    // time — by this point all modules are already loaded.
    try {
      const { rehydrateStores } = await import('./storage-rehydration')
      rehydrateStores()
    } catch {
      // Rehydration helpers may not exist or may fail; state will
      // correct itself on the next user interaction.
    }
  }
}

/**
 * Persist a key-value pair to the main-process filesystem.
 *
 * This is useful for one-shot saves outside the normal persist-flow
 * (e.g. saving state that isn't backed by a nanostore subscription).
 * Most callers should use the `persist*` functions in storage.ts
 * which already write through IPC automatically.
 */
export async function persistDesktopState(key: string, value: unknown): Promise<void> {
  if (typeof window === 'undefined' || !window.hermesDesktop?.setDesktopState) return

  try {
    await window.hermesDesktop.setDesktopState(key, value)
  } catch {
    // Best-effort.
  }
}
