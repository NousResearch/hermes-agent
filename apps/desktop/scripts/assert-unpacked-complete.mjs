/**
 * Fail the pack when the unpacked Electron tree is missing files the app cannot
 * start without (#102013). A locked file (EBUSY on v8_context_snapshot.bin, AV,
 * indexer) can abort electron-builder's copy and leave a torn win-unpacked:
 * "Invalid file descriptor to ICU data" or a blank window from a missing
 * renderer bundle. macOS bundles use a framework layout and are not checked here.
 *
 * Mirrors hermes_cli/main_desktop.py::_unpacked_tree_error; keep the lists in sync.
 */

import fs from 'node:fs'
import path from 'node:path'

const REQUIRED_FILES = ['icudtl.dat', 'resources.pak', 'resources/app.asar', 'resources/app.asar.unpacked/dist/index.html']

/** @param {string} appOutDir @returns {string[]} what is missing, empty when complete */
export function missingUnpackedFiles(appOutDir) {
  const missing = REQUIRED_FILES.filter(rel => !fs.existsSync(path.join(appOutDir, rel)))
  const locales = path.join(appOutDir, 'locales')
  const hasLocale = fs.existsSync(locales) && fs.readdirSync(locales).some(name => name.endsWith('.pak'))
  if (!hasLocale) missing.push('locales/*.pak')
  return missing
}

/** @param {{ appOutDir: string, electronPlatformName: string }} context @returns {void} */
export function assertUnpackedComplete({ appOutDir, electronPlatformName }) {
  if (electronPlatformName === 'darwin') return
  const missing = missingUnpackedFiles(appOutDir)
  if (missing.length) {
    throw new Error(
      `Unpacked app is incomplete (${appOutDir}); missing: ${missing.join(', ')}. ` +
        'A running Hermes.exe, antivirus or indexer probably held a file open. ' +
        'Close Hermes, then pack again.'
    )
  }
}
