/**
 * after-pack.mjs — electron-builder afterPack hook.
 *
 * Stamps the Hermes icon + identity onto the packed Windows Hermes.exe via
 * rcedit (delegated to set-exe-identity.mjs). This runs for EVERY packed build
 * — first install, `hermes desktop`, the installer's --update rebuild, and a
 * dev's manual `npm run pack` — so the branded exe can never silently revert
 * to the stock "Electron" icon/name (the bug when the stamp lived only in
 * install.ps1, which the update path doesn't use).
 *
 * Windows-only: rcedit edits PE resources, irrelevant on macOS/Linux where the
 * app identity comes from the bundle Info.plist / desktop entry. Best-effort:
 * a stamp failure must never fail an otherwise-good build (worst case is the
 * stock icon, not a broken app), so we log and resolve rather than throw.
 *
 * Also verifies the packed Hermes.exe PE Machine matches the pack target
 * (#69179). A wrong-arch electronDist can otherwise produce a launchable-
 * looking win-unpacked tree that Windows rejects with 「此应用无法在你的电脑上运行」.
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - arch:                 Arch enum (0=ia32, 1=x64, 2=armv7l, 3=arm64, ...)
 *   - packager.appInfo.productFilename: the exe basename (e.g. 'Hermes')
 */

import fs from 'node:fs'
import path from 'node:path'

import { Arch } from 'electron-builder'

import { normalizeCpuArch, peArchMatches, readPeArch } from './pe-arch.mjs'
import { stampExeIdentity } from './set-exe-identity.mjs'

export default async function afterPack(context) {
  if (context.electronPlatformName !== 'win32') {
    return
  }

  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const exe = path.join(context.appOutDir, `${productName}.exe`)
  const desktopRoot = path.resolve(import.meta.dirname, '..')

  const archName =
    context && typeof context.arch === 'number' ? Arch[context.arch] : undefined
  const want = normalizeCpuArch(archName)
  // Only gate when the packed exe is actually on disk. A missing path is
  // electron-builder's problem (or a wrong productFilename); do not mis-report
  // it as a PE arch mismatch.
  if (want && fs.existsSync(exe) && !peArchMatches(exe, want)) {
    const got = readPeArch(exe)
    // Remove the bad exe so stamp/existence checks cannot treat it as ready
    // and so the next rebuild cannot no-op on a poisoned tree.
    try {
      fs.rmSync(exe, { force: true })
    } catch {
      /* best-effort */
    }
    throw new Error(
      `[after-pack] ${exe} PE arch is ${got ?? 'unreadable'} but target is ` +
        `${want}. Refusing to ship a Windows desktop binary that would fail ` +
        `with "This app can't run on your PC" (#69179).`
    )
  }

  try {
    await stampExeIdentity(exe, desktopRoot)
  } catch (err) {
    // Never fail the build over a cosmetic stamp.
    console.warn(`[after-pack] exe identity stamp failed (${err.message}); Hermes.exe keeps the stock Electron icon`)
  }
}
