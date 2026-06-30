#!/usr/bin/env node
// set-exe-identity.cjs — stamp the Reuben icon + version metadata onto the
// built Reuben.exe using rcedit, completely decoupled from electron-builder's
// signing path.
//
// WHY THIS EXISTS
// ---------------
// apps/desktop/package.json sets build.win.signAndEditExecutable=false. That
// flag is load-bearing: turning electron-builder's own exe-editing ON also
// re-enables its signtool step, which fetches winCodeSign-2.6.0.7z, whose
// macOS symlinks crash 7-Zip on non-admin Windows (no Developer Mode = no
// SeCreateSymbolicLinkPrivilege). That is an unfixable dead end — we do NOT
// try to extract winCodeSign.
//
// The cost of disabling signAndEditExecutable is that electron-builder also
// skips rcedit, so the unpacked Reuben.exe keeps the stock Electron icon and
// "Electron" taskbar name. This script restores the icon + identity by calling
// rcedit DIRECTLY. rcedit is a pure PE resource editor: no signing, no certs,
// no winCodeSign, no symlinks.
//
// HOW IT RUNS
// -----------
// Primarily as an electron-builder `afterPack` hook (scripts/after-pack.cjs),
// so EVERY packed build — first install, `hermes desktop`, the installer's
// --update rebuild, or a dev's manual `npm run pack` — gets a branded exe from
// one place. Previously this stamp lived only in install.ps1, so the update
// path (which rebuilds via `hermes desktop --build-only`, never install.ps1)
// shipped a stock "Electron" exe. Keeping it in afterPack closes that gap.
//
// Also runnable standalone for ad-hoc re-stamping:
//   node scripts/set-exe-identity.cjs <path-to-Reuben.exe>
//
// Exits 0 on success, non-zero on failure when run as a CLI. As a hook,
// stampExeIdentity() resolves on success and rejects on failure; the caller
// (after-pack.cjs) swallows the rejection so a stamp failure never fails an
// otherwise-good build (worst case: stock icon, not a broken app).

const path = require('node:path')
const fs = require('node:fs')

// Stamp the Reuben icon + identity onto `exe`. Resolves on success, throws on
// failure. `desktopRoot` defaults to this script's package root so the icon and
// the rcedit dependency resolve regardless of cwd.
async function stampExeIdentity(exe, desktopRoot = path.resolve(__dirname, '..')) {
  if (!exe || !fs.existsSync(exe)) {
    throw new Error(`target exe not found: ${exe}`)
  }

  // Icon lives at apps/desktop/assets/icon.ico
  const icon = path.join(desktopRoot, 'assets', 'icon.ico')
  if (!fs.existsSync(icon)) {
    throw new Error(`icon not found: ${icon}`)
  }

  // rcedit is a direct devDependency of apps/desktop, so it resolves whether
  // we're run from the desktop dir or the repo root (workspace hoist).
  // rcedit@5 exports a NAMED `rcedit` function (CommonJS: { rcedit }), not a
  // default export.
  const mod = require('rcedit')
  const rcedit = typeof mod === 'function' ? mod : mod.rcedit
  if (typeof rcedit !== 'function') {
    throw new Error(`unexpected rcedit export shape: ${typeof mod} keys=${Object.keys(mod)}`)
  }

  console.log(`[set-exe-identity] stamping ${exe}`)
  console.log(`[set-exe-identity] icon: ${icon}`)

  await rcedit(exe, {
    icon,
    'version-string': {
      ProductName: 'Reuben',
      FileDescription: 'Reuben',
      CompanyName: 'Nous Research',
      LegalCopyright: 'Copyright (c) 2026 Nous Research'
    }
  })

  console.log('[set-exe-identity] done — Reuben icon + identity stamped')
}

module.exports = { stampExeIdentity }

// CLI entry point: `node scripts/set-exe-identity.cjs <exe>`.
if (require.main === module) {
  const exe = process.argv[2]
  if (!exe) {
    console.error('[set-exe-identity] usage: set-exe-identity.cjs <path-to-exe>')
    process.exit(2)
  }
  stampExeIdentity(exe).catch(err => {
    console.error(`[set-exe-identity] ${err.message}`)
    process.exit(1)
  })
}
