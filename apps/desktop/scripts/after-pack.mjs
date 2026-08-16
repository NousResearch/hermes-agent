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
 * app identity comes from the bundle Info.plist / desktop entry. On Windows,
 * identity stamping is required: a successful build with Electron metadata is
 * not a successful Hermes Desktop update.
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - packager.appInfo.productFilename: the exe basename (e.g. 'Hermes')
 */

import path from 'node:path'

import { stampExeIdentity } from './set-exe-identity.mjs'

export async function afterPack(context, { stamp = stampExeIdentity } = {}) {
  if (context.electronPlatformName !== 'win32') {
    return
  }

  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const exe = path.join(context.appOutDir, `${productName}.exe`)
  const desktopRoot = path.resolve(import.meta.dirname, '..')

  await stamp(exe, desktopRoot)
}

export default afterPack
