/**
 * Swapped-bundle detection.
 *
 * The detached updater (scripts/desktop-update/posix.sh mac_swap /
 * windows.ps1) rebuilds and swaps the packaged app on disk AFTER
 * `hermes update` exits. An instance that was launched from the PRE-swap
 * bundle — the user reopened Hermes mid-update, the #50238 gesture the boot
 * gate exists for — would otherwise proceed to run the NEW runtime under the
 * OLD renderer. The updater's own `open`/relaunch leg cannot rescue it: the
 * single-instance lock turns that into a focus of the parked process, so no
 * process ever loads the new build.
 *
 * That is the stale-renderer tail of a FULLY SUCCESSFUL update: the "App
 * build out of date" banner appears right after the update, while the Updates
 * card says "You're on the latest version" and so offers nothing that would
 * clear it.
 *
 * Detection: compare the install stamp this process loaded at boot with the
 * one on disk now. A different commit — or a different builtAt at the same
 * commit (a dirty-tree or content-hash rebuild) — means the bundle under our
 * feet is not the one we are running, and a plain relaunch loads it.
 *
 * Fail-quiet like bundle-skew: a missing stamp on either side (dev runs,
 * unreadable resources) or a fallback all-zero commit reports "not swapped".
 * This must never false-positive — a positive triggers an automatic relaunch.
 *
 * Pure so it is testable without booting Electron.
 */

import fs from 'node:fs'
import path from 'node:path'

import { isFallbackCommit } from './bundle-skew'

export interface BundleSwapStamp {
  /** write-build-stamp.mjs build timestamp — differs on every rebuild. */
  builtAt: null | string
  commit: string | null
  /** write-build-stamp.mjs source tag — 'fallback' means the commit is fake. */
  source?: null | string
}

/** Read the current installed artifact only to detect replacement by an update.
 * This never chooses a runtime, normalizes a schema, or reads dev build output.
 */
export function readBundleSwapStamp(resourcesPath: string): BundleSwapStamp | null {
  try {
    return JSON.parse(fs.readFileSync(path.join(resourcesPath, 'install-stamp.json'), 'utf8'))
  } catch {
    // An unreadable replacement is not proof of a swap.
    return null
  }
}

/** True only on positive proof that the bundle on disk is not the running one. */
export function detectBundleSwap(running: BundleSwapStamp | null, onDisk: BundleSwapStamp | null): boolean {
  if (!running?.commit || !onDisk?.commit) {
    return false
  }

  if (running.source === 'fallback' || isFallbackCommit(running.commit)) {
    return false
  }

  if (onDisk.source === 'fallback' || isFallbackCommit(onDisk.commit)) {
    return false
  }

  if (running.commit !== onDisk.commit) {
    return true
  }

  return running.builtAt !== onDisk.builtAt
}

// One-shot guard for the automatic bundle-swap relaunch below: the relaunched
// instance carries this flag so a stamp that still mismatches (unreadable
// resources, exotic packaging) can never produce a relaunch loop.
export const BUNDLE_SWAP_RELAUNCH_FLAG = '--hermes-bundle-swap-relaunched'

// How long the parked instance waits for its own scheduled exit to land before
// giving up and booting the stale build anyway. Better a torn renderer with a
// banner than a window that never comes back.
export const BUNDLE_SWAP_RELAUNCH_FAILSAFE_MS = 15_000

export interface BundleSwapRelaunchHost {
  isPackaged: boolean
  argv: string[]
  /** The stamp this process was built with. */
  running: BundleSwapStamp | null
  resourcesPath: string
  /** Schedule a relaunch with these extra args (may throw). */
  relaunch: (extraArgs: string[]) => void
  /** Exit after shutting the backend down (the relaunch lands on exit). */
  exit: () => void
  log: (line: string) => void
}

// The detached updater swaps the packaged bundle on disk AFTER `hermes update`
// exits (posix.sh mac_swap / windows.ps1). An instance reopened mid-update —
// the #50238 gesture the boot update gate exists for — was launched from the
// PRE-swap bundle, and the updater's `open` leg then merely focuses us (single
// instance), so no process ever loads the new build. Letting boot proceed here
// runs the new runtime under the old renderer: exactly the skew
// detectRendererSkew() warns about, except the Updates card already says
// "latest", so the warning's own remedy has nothing to run.
//
// waitForUpdateToFinish (main.ts) calls this at the earliest point where the
// swap is PROVABLE — it happens while we are parked on the gate, so checking any sooner (at `ready`, before the gate)
// only ever compares a stamp with itself. Relaunching here also keeps the
// boot-progress window up for the whole wait instead of leaving the user with
// no window at all.
//
// Returns true when the relaunch was scheduled; the caller must park rather
// than continue booting, because the process exits underneath it.
export function relaunchIntoSwappedBundle(host: BundleSwapRelaunchHost): boolean {
  if (!host.isPackaged || host.argv.includes(BUNDLE_SWAP_RELAUNCH_FLAG)) {
    return false
  }

  if (!detectBundleSwap(host.running, readBundleSwapStamp(host.resourcesPath))) {
    return false
  }

  host.log('[updates] app bundle was swapped during the update; relaunching into the new build')

  try {
    host.relaunch([BUNDLE_SWAP_RELAUNCH_FLAG])
  } catch (err) {
    host.log(
      `[updates] bundle-swap relaunch failed: ${(err as Error)?.message || err}; continuing with the current build`
    )

    return false
  }

  host.exit()

  return true
}
