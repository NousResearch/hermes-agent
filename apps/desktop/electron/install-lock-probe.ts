'use strict'

/**
 * install-lock-probe.ts
 *
 * "Is this install free for the updater to rewrite?" — the whole answer, in
 * one module.
 *
 * The pieces underneath are already separated: install-mutation-set.ts knows
 * WHICH files an update rewrites and how an exclusive open classifies a lock,
 * windows-restart-manager.ts asks the kernel WHO holds a locked file, and
 * backend-release-gate.ts knows how often the release poll may afford to ask.
 * The glue that binds them to an install root lived in main.ts, so every
 * caller that wanted the question answered reached through three modules to
 * assemble it. It lives here now; main.ts asks.
 *
 * Nothing here mutates. Every entry point takes an install root and a budget
 * and returns evidence.
 */

import fs from 'node:fs'
import path from 'node:path'

import { createInstallLockGateProbe } from './backend-release-gate'
import {
  getInstallMutationSet,
  type InstallResourceLocks,
  probeInstallResourceLocks
} from './install-mutation-set'
import {
  listRestartManagerHoldersForResources,
  RESTART_MANAGER_DEFAULT_TIMEOUT_MS
} from './windows-restart-manager'
import type { ForceReleaseHolder } from './windows-update-force-release'

const IS_WINDOWS = process.platform === 'win32'

// Path to the venv shim whose lock decides whether `hermes update` can write
// fresh entry points. On Windows this is the file the running backend
// `hermes.exe` holds open; on POSIX it's never mandatory-locked.
export function venvHermesShimPath(updateRoot: string): string {
  return IS_WINDOWS
    ? path.join(updateRoot, 'venv', 'Scripts', 'hermes.exe')
    : path.join(updateRoot, 'venv', 'bin', 'hermes')
}

// Best-effort lock probe mirroring the Rust updater's is_locked(): a running
// .exe on Windows refuses an O_RDWR open with a sharing violation. On POSIX
// this practically always succeeds (no mandatory locking), so it returns false
// — correct, since the shim-contention brick is Windows-only.
function isShimLocked(shimPath: string): boolean {
  if (!IS_WINDOWS) {
    return false
  }

  let fd

  try {
    fd = fs.openSync(shimPath, 'r+')

    return false
  } catch (err: any) {
    // ENOENT ⇒ not there ⇒ nothing locking it. Anything else (EBUSY/EPERM/
    // EACCES) on Windows means a live handle holds it.
    return Boolean(err) && err.code !== 'ENOENT'
  } finally {
    if (fd !== undefined) {
      try {
        fs.closeSync(fd)
      } catch {
        void 0
      }
    }
  }
}

// The files the updater will replace or delete: every native module, DLL,
// and executable under venv\. This is what pip/uv actually needs free. The
// shim alone only proves the uv launcher is gone; the real interpreter runs
// from .hermes-runtime and keeps site-packages .pyd files mapped without
// touching hermes.exe, so a shim-only probe let the handoff proceed into the
// July 2026 brotlicffi/_sodium.pyd half-updated venv.
export function installLockResources(updateRoot: string): string[] {
  return getInstallMutationSet(updateRoot)
}

// Exclusive-open probe over the mutation set, split into files only our
// link can lock (definite) and uv-shared hard links that need per-process
// attribution. Falls back to the shim probe on a checkout without a venv.
//
// `limit` bounds the main-thread cost. Enumerating the whole set costs ~270
// synchronous openSync calls (measured on a real install: ~27 ms median, and
// seconds when a filter driver is cold), which the release gate paid on every
// 300 ms poll. A caller that only needs "is anything locked" passes limit: 1
// and stops at the first lock of either kind (~0.3 ms while the install is
// held); attribution re-probes the full set when it needs the whole picture.
export function probeInstallLocks(updateRoot: string, options: { limit?: number } = {}): InstallResourceLocks {
  const resources = installLockResources(updateRoot)

  if (resources.length === 0) {
    const shim = venvHermesShimPath(updateRoot)

    return { definite: isShimLocked(shim) ? [shim] : [], shared: [] }
  }

  return probeInstallResourceLocks(resources, options)
}

// Holders proven by the kernel: Restart Manager over the locked files, with
// per-process module attribution for uv-shared files so a foreign venv that
// maps the same wheel through its own hard link is never listed.
export async function attributedInstallHolders(
  updateRoot: string,
  timeoutMs = RESTART_MANAGER_DEFAULT_TIMEOUT_MS
): Promise<ForceReleaseHolder[]> {
  const locks = probeInstallLocks(updateRoot)

  if (locks.definite.length === 0 && locks.shared.length === 0) {return []}

  return listRestartManagerHoldersForResources(locks.definite, {
    shared: locks.shared,
    // Attribute against the venv, the only tree the sync rewrites: a process
    // that maps runtime DLLs but no venv file is not a holder of this update.
    attributionRoot: path.join(updateRoot, 'venv'),
    timeoutMs
  })
}

// One boolean answer for the force-release discovery loop (a handful of passes,
// each with its own budget — not a 300 ms poll; the release gate uses
// createInstallLockGateProbe). Stops at the first definite lock.
export async function isAnyInstallResourceLocked(updateRoot: string): Promise<boolean> {
  const locks = probeInstallLocks(updateRoot, { limit: 1 })

  if (locks.definite.length > 0) {return true}

  if (locks.shared.length === 0) {return false}

  return (await attributedInstallHolders(updateRoot)).length > 0
}


/**
 * The release gate's bounded lock probe for one install root. The gate polls
 * every 300 ms, so it stops at the first lock and attributes at most once per
 * cooldown; see createInstallLockGateProbe for why both bounds exist.
 */
export function createInstallLockGateProbeForRoot(updateRoot: string): () => Promise<boolean> {
  return createInstallLockGateProbe({
    probeLocks: limit => probeInstallLocks(updateRoot, { limit }),
    countAttributedHolders: async budgetMs => (await attributedInstallHolders(updateRoot, budgetMs)).length
  })
}
