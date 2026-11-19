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

import { PACKAGE_MARKER, withDesktopPluginsRoot } from './desktop-plugins-root'
import { stripLikePython } from './plugin-files-guards'

export interface RemoveDesktopPluginResult {
  ok: boolean
  error?: string
  /** The folder that was deleted (on success). */
  path?: string
}

export function removeDesktopPlugin(appRoot: string, rawName: unknown): Promise<RemoveDesktopPluginResult> {
  // Under the root's writer lock: a reconcile pass adopting this folder (it
  // stamps the marker) must not land between the marker check and the delete.
  return withDesktopPluginsRoot(appRoot, () => removeFolder(appRoot, rawName))
}

function isSingleSegment(name: string): boolean {
  return Boolean(name) && name !== '.' && name !== '..' && !/[\\/]/.test(name)
}

async function removeFolder(appRoot: string, rawName: unknown): Promise<RemoveDesktopPluginResult> {
  const root = path.resolve(appRoot)
  const raw = String(rawName ?? '')

  // An existing folder's exact name removes that folder (one an older version named differently, or copied in
  // by hand); any other name is trimmed as the installer trims the folder name it creates.
  const exists =
    isSingleSegment(raw) &&
    (await fs.promises.lstat(path.join(root, raw)).then(
      () => true,
      () => false
    ))

  const name = exists ? raw : stripLikePython(raw)

  if (!isSingleSegment(name)) {
    return { ok: false, error: 'invalid plugin folder name' }
  }

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
