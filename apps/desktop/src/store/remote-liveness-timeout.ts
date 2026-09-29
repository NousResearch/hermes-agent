/**
 * Remote liveness/dispatch probe timeout — how long the desktop CLIENT waits
 * for a remote backend's health probe before treating it as unreachable.
 *
 * A device-local preference (each client machine trades probe patience
 * against how fast it notices a truly dead remote). The MAIN process is
 * authoritative: it owns the live value and the persisted copy, and applies
 * a new value immediately — no restart. This store mirrors the live value
 * for the Settings row.
 */

import { atom } from 'nanostores'

export const REMOTE_LIVENESS_TIMEOUT_DEFAULT_MS = 10_000

export const $remoteLivenessTimeoutMs = atom<number>(REMOTE_LIVENESS_TIMEOUT_DEFAULT_MS)

/** Seed from main's authoritative state once at startup; no-op without the
 *  bridge (web/older builds just keep the default for the UI). */
export async function loadRemoteLivenessTimeout(): Promise<void> {
  try {
    const result = await window.hermesDesktop?.getRemoteLivenessTimeout?.()

    if (typeof result?.timeoutMs === 'number') {
      $remoteLivenessTimeoutMs.set(result.timeoutMs)
    }
  } catch {
    // Keep the default — the Settings row still renders and can retry on save.
  }
}

/** Push a new timeout to main; adopt the post-clamp value it reports. */
export async function saveRemoteLivenessTimeout(next: number): Promise<void> {
  const current = $remoteLivenessTimeoutMs.get()

  // Optimistic paint, then honest reconciliation with the clamped result.
  $remoteLivenessTimeoutMs.set(next)

  try {
    const result = await window.hermesDesktop?.setRemoteLivenessTimeout?.(next)

    if (typeof result?.timeoutMs === 'number') {
      $remoteLivenessTimeoutMs.set(result.timeoutMs)
    }
  } catch {
    $remoteLivenessTimeoutMs.set(current)
    throw new Error('Applying the remote liveness timeout failed')
  }
}
