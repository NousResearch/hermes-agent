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
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - packager.appInfo.productFilename: the exe basename (e.g. 'Hermes')
 */

import { execFile } from 'node:child_process'
import path from 'node:path'
import { promisify } from 'node:util'

import { stampExeIdentity } from './set-exe-identity.mjs'

const run = promisify(execFile)

/**
 * Re-sign the macOS bundle ad-hoc when electron-builder left it unsigned.
 *
 * Without a Developer ID, electron-builder skips signing and the bundle keeps
 * Electron's own linker-signed signature — which our resources invalidate
 * ("code has no resources but signature indicates they must be present"), and
 * whose identifier is the generic `Electron`. macOS binds a keychain ACL to a
 * VERIFIED code identity, so an invalid signature means "Always Allow" on the
 * safeStorage prompt can never stick: every launch and every token read
 * re-prompts for the login keychain password.
 *
 * An ad-hoc signature is enough to fix that — it yields a stable identity
 * (com.nousresearch.hermes) with sealed resources. We only touch a bundle that
 * FAILS verification, so a real Developer ID signature is never clobbered.
 */
async function adhocSignMac(appPath) {
  try {
    await run('codesign', ['--verify', '--deep', '--strict', appPath])

    return
  } catch {
    // Unsigned or invalid — fall through and ad-hoc sign.
  }

  await run('codesign', ['--force', '--deep', '--sign', '-', appPath])
  await run('codesign', ['--verify', '--deep', '--strict', appPath])
}

export default async function afterPack(context) {
  const productName = context.packager?.appInfo?.productFilename || 'Hermes'

  if (context.electronPlatformName === 'darwin') {
    const appPath = path.join(context.appOutDir, `${productName}.app`)

    try {
      await adhocSignMac(appPath)
    } catch (err) {
      // Never fail the build; the cost is the recurring keychain prompt.
      console.warn(`[after-pack] ad-hoc codesign failed (${err.message}); expect repeated keychain prompts`)
    }

    return
  }

  if (context.electronPlatformName !== 'win32') {
    return
  }

  const exe = path.join(context.appOutDir, `${productName}.exe`)
  const desktopRoot = path.resolve(import.meta.dirname, '..')

  try {
    await stampExeIdentity(exe, desktopRoot)
  } catch (err) {
    // Never fail the build over a cosmetic stamp.
    console.warn(`[after-pack] exe identity stamp failed (${err.message}); Hermes.exe keeps the stock Electron icon`)
  }
}
