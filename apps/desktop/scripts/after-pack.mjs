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
 * On macOS, stages the Arabic InfoPlist.strings after Electron's locale pruning
 * so permission prompts and the localized app name ship in the final bundle.
 * On Windows, rcedit stamps the PE identity. The Windows stamp is best-effort:
 * a failure must never fail an otherwise-good build (worst case is the stock
 * icon, not a broken app), so we log and resolve rather than throw.
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - packager.appInfo.productFilename: the exe basename (e.g. 'Hermes')
 */

import { copyFile, mkdir } from 'node:fs/promises'
import path from 'node:path'

import { stampExeIdentity } from './set-exe-identity.mjs'

export async function stageMacLocalizedInfoPlist(context) {
  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const desktopRoot = path.resolve(import.meta.dirname, '..')
  const source = path.join(desktopRoot, 'electron', 'localization', 'ar.lproj', 'InfoPlist.strings')
  const destinationDir = path.join(
    context.appOutDir,
    `${productName}.app`,
    'Contents',
    'Resources',
    'ar.lproj'
  )

  await mkdir(destinationDir, { recursive: true })
  await copyFile(source, path.join(destinationDir, 'InfoPlist.strings'))
}

export default async function afterPack(context) {
  if (context.electronPlatformName === 'darwin') {
    await stageMacLocalizedInfoPlist(context)

    return
  }

  if (context.electronPlatformName !== 'win32') {
    return
  }

  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const exe = path.join(context.appOutDir, `${productName}.exe`)
  const desktopRoot = path.resolve(import.meta.dirname, '..')

  try {
    await stampExeIdentity(exe, desktopRoot)
  } catch (err) {
    // Never fail the build over a cosmetic stamp.
    console.warn(`[after-pack] exe identity stamp failed (${err.message}); Hermes.exe keeps the stock Electron icon`)
  }
}
