#!/usr/bin/env node
// set-exe-identity.mjs — stamp the Hermes icon + version metadata onto the
// built Hermes.exe using rcedit, completely decoupled from electron-builder's
// signing path.
//
// WHY THIS EXISTS
// ---------------
// apps/desktop/package.json sets build.win.signAndEditExecutable=false. That
// flag is load-bearing FOR THE LOCAL AND DEVELOPER PATHS: turning
// electron-builder's own exe-editing ON also re-enables its signtool step, which
// fetches winCodeSign-2.6.0.7z, whose macOS symlinks crash 7-Zip on a Windows
// host that cannot create symlinks without elevation (Developer Mode off = no
// SeCreateSymbolicLinkPrivilege). On such a host that is an unfixable dead end,
// which is why the local path does NOT try to extract winCodeSign. Note the
// constraint is conditional on Developer Mode, not universal: a dev box that has
// it on can extract the archive (verified on 2026-09-13 — the probe in the
// release preflight passes locally there).
//
// THE RELEASE PATH IS THE DOCUMENTED EXCEPTION. electron-builder's own signing
// step is what Azure Trusted Signing hooks into, so
// apps/desktop/electron-builder.release.cjs sets signAndEditExecutable=true and
// does extract winCodeSign. That works only where a non-elevated process may
// create symlinks: true on the x64 GitHub-hosted Windows images, which ship
// Developer Mode on (actions/runner-images Configure-DeveloperMode.ps1). Because
// the condition is a property of the host rather than of this repo, the release
// workflow asserts it in a preflight step before it exposes the signing secrets
// instead of assuming it — see `.github/workflows/desktop-release.yml`.
//
// The cost of disabling signAndEditExecutable is that electron-builder also
// skips rcedit, so the unpacked Hermes.exe keeps the stock Electron icon and
// "Electron" taskbar name. This script restores the icon + identity by calling
// rcedit DIRECTLY. rcedit is a pure PE resource editor: no signing, no certs,
// no winCodeSign, no symlinks.
//
// HOW IT RUNS
// -----------
// Primarily as an electron-builder `afterPack` hook (scripts/after-pack.mjs),
// so EVERY packed build — first install, `hermes desktop`, the installer's
// --update rebuild, or a dev's manual `npm run pack` — gets a branded exe from
// one place. Previously this stamp lived only in install.ps1, so the update
// path (which rebuilds via `hermes desktop --build-only`, never install.ps1)
// shipped a stock "Electron" exe. Keeping it in afterPack closes that gap.
//
// Also runnable standalone for ad-hoc re-stamping:
//   node scripts/set-exe-identity.mjs <path-to-Hermes.exe>
//
// Exits 0 on success, non-zero on failure when run as a CLI. As a hook,
// stampExeIdentity() resolves on success and rejects on failure; the caller
// (after-pack.mjs) swallows the rejection so a stamp failure never fails an
// otherwise-good build (worst case: stock icon, not a broken app).
//
// Official releases are the deliberate exception. The release-only builder
// overlay enables signing on a controlled CI runner after this afterPack stamp
// and makes any branding failure fatal.

import { resolve, join } from 'node:path'
import { existsSync } from 'node:fs'

import { rcedit } from 'rcedit'

import { isMain } from './utils.mjs'

// A real-time file scanner (AV/EDR) holds a short exclusive handle on a freshly
// written exe; rcedit's resource commit then fails with "Unable to commit
// changes" and succeeds seconds later on identical input. Delays sized to the
// field report: the lock was still held 5 s after a first attempt in some runs.
const RCEDIT_COMMIT_RETRY_DELAYS_MS = [500, 1000, 2000]

// A failure to spawn the rcedit binary itself (missing or not executable) is
// permanent; waiting 3.5 s on it only delays after-pack.mjs's warning. The npm
// rcedit wrapper surfaces the spawn error as `originalError` on its rejection,
// while a non-zero rcedit exit carries a numeric `code`.
const RCEDIT_PERMANENT_SPAWN_CODES = new Set(['ENOENT', 'EACCES'])

function isPermanentRceditFailure(err) {
  return RCEDIT_PERMANENT_SPAWN_CODES.has(err?.originalError?.code ?? err?.code)
}

function wait(delay) {
  return new Promise(resolve => setTimeout(resolve, delay))
}

// Stamp the Hermes icon + identity onto `exe`. Resolves on success, throws on
// failure. `desktopRoot` defaults to this script's package root so the icon and
// the rcedit dependency resolve regardless of cwd.
async function stampExeIdentity(
  exe,
  desktopRoot = resolve(import.meta.dirname, '..'),
  { rcedit: runRcedit = rcedit, sleep = wait } = {}
) {
  if (!exe || !existsSync(exe)) {
    throw new Error(`target exe not found: ${exe}`)
  }

  // Icon lives at apps/desktop/assets/icon.ico
  const icon = join(desktopRoot, 'assets', 'icon.ico')
  if (!existsSync(icon)) {
    throw new Error(`icon not found: ${icon}`)
  }

  console.log(`[set-exe-identity] stamping ${exe}`)
  console.log(`[set-exe-identity] icon: ${icon}`)

  const options = {
    icon,
    'version-string': {
      ProductName: 'Hermes',
      FileDescription: 'Hermes',
      CompanyName: 'Nous Research',
      LegalCopyright: 'Copyright (c) 2026 Nous Research'
    }
  }

  for (let attempt = 0; ; attempt += 1) {
    try {
      await runRcedit(exe, options)
      break
    } catch (err) {
      const delay = RCEDIT_COMMIT_RETRY_DELAYS_MS[attempt]
      if (delay === undefined || isPermanentRceditFailure(err)) {
        throw err
      }
      console.warn(`[set-exe-identity] rcedit failed; retrying in ${delay}ms (${err.message})`)
      await sleep(delay)
    }
  }

  console.log('[set-exe-identity] done — Hermes icon + identity stamped')
}

export { RCEDIT_COMMIT_RETRY_DELAYS_MS, stampExeIdentity }

// CLI entry point: `node scripts/set-exe-identity.mjs <exe>`.
if (isMain(import.meta.url)) {
  const exe = process.argv[2]
  if (!exe) {
    console.error('[set-exe-identity] usage: set-exe-identity.mjs <path-to-exe>')
    process.exit(2)
  }
  stampExeIdentity(exe).catch(err => {
    console.error(`[set-exe-identity] ${err.message}`)
    process.exit(1)
  })
}
