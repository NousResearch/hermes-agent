/** Restore the empty app-level localizations dropped during Electron extraction.
 * Run after language filtering and before signing; derive the markers from the
 * packaged framework, not the host's Electron (which may be another version).
 * Windows identity stamping stays in afterExtract, before ASAR integrity.
 */
import { mkdir, readdir } from 'node:fs/promises'
import path from 'node:path'

export default async function afterPack({ electronPlatformName, appOutDir, packager }) {
  if (electronPlatformName !== 'darwin') {
    return
  }

  try {
    const resources = packager.getResourcesDir(appOutDir)
    const framework = packager.getMacOsElectronFrameworkResourcesDir(appOutDir)
    const entries = await readdir(framework, { withFileTypes: true })
    // Chromium also ships grammatical-gender packs; these are not macOS locales.
    const locales = entries.filter(
      entry =>
        entry.isDirectory() && entry.name.endsWith('.lproj') && !/_(FEMININE|MASCULINE|NEUTER)\.lproj$/.test(entry.name)
    )
    await Promise.all(locales.map(entry => mkdir(path.join(resources, entry.name), { recursive: true })))
  } catch (error) {
    // Keep an otherwise usable package, but make failed locale restoration visible.
    console.warn(
      `[after-pack] macOS locale markers were not restored: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}
