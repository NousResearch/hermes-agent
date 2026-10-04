/**
 * The update gate's marker probe (R6, SPEC section 6 "Electron gate").
 *
 * A marker whose owner and delegate are dead is not necessarily a finished
 * update: an inheriting completion process can still hold the checkout's
 * kernel lock after the script's immediate child died. The Desktop cannot see
 * that lock portably and must never delete the marker (A7 rule 3), so a
 * dead/malformed marker is routed through the checkout script's `reclaim`
 * helper, which decides under the `<marker>.lock` sidecar:
 *
 * - `held` / `busy` / `live <pid>` => an update still owns the checkout: keep waiting;
 * - `reclaimed` / `absent` => nothing runs: proceed;
 * - `unsupported` (older checkout, no helper) => proceed without deleting
 *   (dead = not running, as before minus the deletion).
 *
 * The helper is asked once per distinct dead marker body per wait; a `held` /
 * `busy` / `live` answer is re-asked every `reprobeMs` (5 s) because only the
 * script can see the lock being released.
 */

import { type CreateTimeProbe, inspectUpdateMarker } from './update-marker'
import type { MarkerHelperVerdict } from './updater/marker-helper'

export const HELD_REPROBE_MS = 5_000

export interface LiveMarkerProbeOptions {
  hermesHome: string
  /** The script helper `reclaim`, or null when the checkout's script predates protocol 2. */
  reclaim: (() => Promise<MarkerHelperVerdict>) | null
  createTime?: CreateTimeProbe
  onLiveMarker?: (marker: { startedAt: number | null }) => void
  log?: (line: string) => void
  now?: () => number
  reprobeMs?: number
}

const STILL_RUNNING = new Set(['held', 'busy', 'live'])

/** `hasLiveMarker` for one gate wait (create it per wait, never module-wide). */
export function liveMarkerProbe({
  hermesHome,
  reclaim,
  createTime,
  onLiveMarker,
  log,
  now = Date.now,
  reprobeMs = HELD_REPROBE_MS
}: LiveMarkerProbeOptions): () => Promise<boolean> {
  const asked = new Map<string, { running: boolean; at: number }>()

  return async () => {
    const inspection = await inspectUpdateMarker(hermesHome, { createTime, now })

    if (inspection.state === 'live') {
      onLiveMarker?.({ startedAt: inspection.marker?.startedAt ?? null })

      return true
    }

    if (inspection.state !== 'dead' || !reclaim) {
      return false
    }

    const key = inspection.raw.toString('hex')
    const previous = asked.get(key)

    if (previous && (!previous.running || now() - previous.at < reprobeMs)) {
      return previous.running
    }

    const verdict = await reclaim()
    const running = STILL_RUNNING.has(verdict.kind)

    if (!previous || previous.running !== running) {
      log?.(`[updates] dead update marker: script helper says ${verdict.kind}${'pid' in verdict ? ` ${verdict.pid}` : ''}`)
    }

    asked.set(key, { running, at: now() })

    if (running) {
      onLiveMarker?.({ startedAt: inspection.marker?.startedAt ?? null })
    }

    return running
  }
}
