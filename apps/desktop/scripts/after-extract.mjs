/**
 * after-extract.mjs — electron-builder afterExtract hook.
 *
 * Stamps the Hermes icon + identity onto the unpacked Windows Electron binary
 * before electron-builder renames it and injects the ASAR integrity resource.
 *
 * WHY afterExtract and not afterPack (#105629): with ASAR integrity on (the
 * default) electron-builder's `beforeCopyExtraFiles` rebuilds the whole PE in
 * memory with resedit to push the ELECTRONASAR resource. rcedit cannot commit
 * changes to that rewritten PE ("Fatal error: Unable to commit changes"),
 * deterministically, on every build. afterExtract fires on the pristine
 * electron.exe, before the rename and the integrity rewrite, and resedit then
 * carries the stamped resources through — so the exe keeps both its identity
 * and its integrity checksum. Disabling `disableAsarIntegrity` would also
 * "fix" it, at the cost of the integrity check on every Windows build.
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
