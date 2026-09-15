/**
 * after-extract.mjs — electron-builder afterExtract hook.
 *
 * Stamps the Hermes icon + identity onto the unpacked Windows Electron binary
 * before electron-builder renames it and injects the ASAR integrity resource.
 * Keeping the rcedit mutation before electron-builder's resedit rewrite makes
 * both resource edits compose into the final Hermes.exe.
 *
 * Windows-only: rcedit edits PE resources, irrelevant on macOS/Linux where the
 * app identity comes from the bundle Info.plist / desktop entry. Best-effort:
 * a stamp failure must never fail an otherwise-good build (worst case is the
 * stock icon, not a broken app), so we log and resolve rather than throw.
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked Electron directory
 *   - packager.config.electronBranding.projectName: source binary basename
 */

import path from 'node:path'

import { stampExeIdentity } from './set-exe-identity.mjs'

export default async function afterExtract(context) {
  if (context.electronPlatformName !== 'win32') {
    return
  }

  const projectName = context.packager?.config?.electronBranding?.projectName || 'electron'
  const exe = path.join(context.appOutDir, `${projectName}.exe`)
  const desktopRoot = path.resolve(import.meta.dirname, '..')

  try {
    await stampExeIdentity(exe, desktopRoot)
  } catch (err) {
    // Never fail the build over a cosmetic stamp.
    console.warn(`[after-extract] exe identity stamp failed (${err.message}); ${projectName}.exe keeps the stock Electron icon`)
  }
}
