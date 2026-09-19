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
 * On macOS, restore the empty app-level locale directories that electron-builder
 * drops while copying Electron. Chromium uses them to select the renderer locale.
 * Windows identity stamping stays best-effort so a cosmetic failure cannot fail a package.
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - packager.appInfo.productFilename: the exe basename (e.g. 'Hermes')
 */

import { mkdir, readdir } from 'node:fs/promises'
import path from 'node:path'

import { stampExeIdentity } from './set-exe-identity.mjs'

async function restoreMacLocaleMarkers(appOutDir, productName) {
  try {
    const appContents = path.join(appOutDir, `${productName}.app`, 'Contents')
    const frameworkResources = path.join(
      appContents,
      'Frameworks',
      'Electron Framework.framework',
      'Versions',
      'A',
      'Resources'
    )
    const appResources = path.join(appContents, 'Resources')
    const entries = await readdir(frameworkResources, { withFileTypes: true })
    const localeMarkers = entries.filter(entry => entry.isDirectory() && entry.name.endsWith('.lproj'))

    await Promise.all(localeMarkers.map(entry => mkdir(path.join(appResources, entry.name), { recursive: true })))

    return localeMarkers.length
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err)
    console.warn(`[after-pack] macOS locale markers were not restored: ${detail}`)
    return 0
  }
}

export default async function afterPack(context) {
  const productName = context.packager?.appInfo?.productFilename || 'Hermes'

  if (context.electronPlatformName === 'darwin') {
    const restored = await restoreMacLocaleMarkers(context.appOutDir, productName)
    console.log(`[after-pack] restored ${restored} macOS locale markers`)
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

export { restoreMacLocaleMarkers }
