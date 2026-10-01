// Uninstall of a STANDALONE desktop plugin: delete one folder directly under
// the app-level `<HERMES_HOME>/desktop-plugins` root. The renderer names the
// folder, never a path — containment is by construction (single segment) and
// re-checked after resolution so nothing outside the root can be addressed.
//
// A unified package's desktop half (folder carrying `.hermes-package.json`) is
// refused here: the reconcile re-copies it from `plugins/<name>/desktop` on
// the next pass while the agent package exists, so deleting the copy would
// only make it flicker. That half goes away with `plugins.manage remove`.
import fs from 'node:fs'
import path from 'node:path'

import { PACKAGE_MARKER } from './desktop-plugins-root'

export interface RemoveDesktopPluginResult {
  ok: boolean
  error?: string
  /** The folder that was deleted (on success). */
  path?: string
}

export async function removeDesktopPlugin(appRoot: string, rawName: unknown): Promise<RemoveDesktopPluginResult> {
  const name = String(rawName ?? '').trim()

  if (!name || name === '.' || name === '..' || /[\\/]/.test(name)) {
    return { ok: false, error: 'invalid plugin folder name' }
  }

  const root = path.resolve(appRoot)
  const target = path.resolve(root, name)

  if (path.relative(root, target) !== name) {
    return { ok: false, error: `${name} is not inside the desktop-plugins folder` }
  }

  let stat: fs.Stats

  try {
    // lstat: a symlinked folder is removed as the LINK, never by following it.
    stat = await fs.promises.lstat(target)
  } catch {
    return { ok: false, error: `${name} is not installed` }
  }

  if (!stat.isDirectory() && !stat.isSymbolicLink()) {
    return { ok: false, error: `${name} is not a plugin folder` }
  }

  if (stat.isDirectory() && fs.existsSync(path.join(target, PACKAGE_MARKER))) {
    return { ok: false, error: `${name} is the desktop half of an agent plugin — uninstall that plugin instead` }
  }

  try {
    await fs.promises.rm(target, { force: true, recursive: true })
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }

  return { ok: true, path: target }
}
