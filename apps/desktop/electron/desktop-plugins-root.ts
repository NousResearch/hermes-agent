// The standalone desktop-plugin root (`<HERMES_HOME>/desktop-plugins`) and the
// one-time migrations that make it the ONLY place desktop code loads from.
//
// A desktop plugin extends THIS APP — panes, palette commands, themes — not an
// agent. Profiles are agents; anything discovered through a profile's folder
// appeared and disappeared with the active profile, which read as "my plugin
// vanished" every time the user switched. Bundled plugins never had that
// problem (they ship in the app), which hid the bug for disk installs.
//
// Two profile-shaped sources are lifted into the app root:
//   1. `profiles/<name>/desktop-plugins/<id>` — earlier builds scoped the
//      standalone root per profile.
//   2. `plugins/<name>/desktop/plugin.js` (default home AND every profile) —
//      the desktop half of a unified agent+desktop package. The agent half
//      stays where it is (it runs in that profile's gateway); the desktop half
//      is COPIED out as `<root>/<name>/` with a `.hermes-package.json` marker
//      so the UI can pair it back to the agent row and re-copy on update.
import fs from 'node:fs'
import path from 'node:path'

export const DESKTOP_PLUGINS_DIR = 'desktop-plugins'
/** Marker inside a materialized desktop half: which agent package it came from. */
export const PACKAGE_MARKER = '.hermes-package.json'

export interface DesktopHalfMarker {
  /** Agent package folder name (the `plugins/<name>` key). */
  package: string
  /** Where the half was copied from — refreshed whenever that source changes. */
  source: string
  /** mtimeMs of the source `plugin.js` at copy time; a newer source re-copies. */
  sourceMtimeMs: number
  /** Where the PACKAGE came from, so "Install here" can install its agent half
   *  into another profile: the catalog sidecar's repo/sha, else the git remote. */
  repo?: string
  sha?: string
  catalogName?: string
}

/** Provenance of an installed agent package: catalog sidecar first, then the
 *  git remote. Undefined for a folder that was copied in by hand. */
async function packageOrigin(packageDir: string): Promise<Pick<DesktopHalfMarker, 'catalogName' | 'repo' | 'sha'>> {
  try {
    const sidecar = JSON.parse(await fs.promises.readFile(path.join(packageDir, '.hermes-catalog.json'), 'utf8')) as {
      catalog_name?: string
      repo?: string
      sha?: string
    }

    if (sidecar.repo) {
      return { catalogName: sidecar.catalog_name, repo: sidecar.repo, sha: sidecar.sha }
    }
  } catch {
    // No sidecar — not a catalog install.
  }

  try {
    const config = await fs.promises.readFile(path.join(packageDir, '.git', 'config'), 'utf8')
    const match = /\[remote "origin"\][^[]*?url\s*=\s*(\S+)/.exec(config)

    if (match) {
      return { repo: match[1] }
    }
  } catch {
    // Not a git checkout.
  }

  return {}
}

export async function ensureDir(dir: string): Promise<string> {
  try {
    await fs.promises.mkdir(dir, { recursive: true })
  } catch {
    // Best-effort create; return the path regardless so a reveal action can
    // still surface a real openPath error and the scanner can retry later.
  }

  return dir
}

async function listDirs(dir: string): Promise<string[]> {
  try {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true })

    return entries.filter(entry => entry.isDirectory()).map(entry => entry.name)
  } catch {
    return []
  }
}

/** Every hermes home the app knows about locally: the default plus each profile. */
export async function localHomes(hermesHome: string): Promise<string[]> {
  const profiles = await listDirs(path.join(hermesHome, 'profiles'))

  return [hermesHome, ...profiles.map(name => path.join(hermesHome, 'profiles', name))]
}

/** Move every `profiles/<name>/desktop-plugins/<id>` folder into the app-level
 *  root. A plugin already present at the root wins (folders are keyed by plugin
 *  id, so a duplicate is the same plugin installed twice); the profile copy is
 *  left in place for the user to delete rather than destroyed. Emptied profile
 *  roots are removed so the migration is a no-op on the next launch. */
export async function migrateProfileScopedDesktopPlugins(hermesHome: string, appRoot: string): Promise<string[]> {
  const moved: string[] = []

  for (const profile of await listDirs(path.join(hermesHome, 'profiles'))) {
    const scopedRoot = path.join(hermesHome, 'profiles', profile, DESKTOP_PLUGINS_DIR)

    for (const entry of await listDirs(scopedRoot)) {
      const from = path.join(scopedRoot, entry)
      const to = path.join(appRoot, entry)

      if (fs.existsSync(to)) {
        continue
      }

      try {
        await fs.promises.rename(from, to)
        moved.push(to)
      } catch {
        // Cross-device or permission failure: leave the folder; the user can
        // still reach it through the profile directory.
      }
    }

    try {
      if ((await fs.promises.readdir(scopedRoot)).length === 0) {
        await fs.promises.rmdir(scopedRoot)
      }
    } catch {
      // Not empty or already gone — either is fine.
    }
  }

  return moved
}

async function readMarker(dir: string): Promise<DesktopHalfMarker | null> {
  try {
    const raw = await fs.promises.readFile(path.join(dir, PACKAGE_MARKER), 'utf8')
    const parsed = JSON.parse(raw) as Partial<DesktopHalfMarker>

    return parsed.package && parsed.source ? (parsed as DesktopHalfMarker) : null
  } catch {
    return null
  }
}

/** Copy one unified package's `desktop/` half into the app root as
 *  `<appRoot>/<packageName>/`, stamping the marker. Skips when the root copy is
 *  already current for this source; replaces it when the source is newer. A
 *  root folder of the same name WITHOUT a marker is a standalone install the
 *  user made on purpose and is never overwritten. Returns the target path
 *  when a copy happened. */
export async function materializeDesktopHalf(
  packageDir: string,
  appRoot: string,
  packageName = path.basename(packageDir)
): Promise<null | string> {
  const sourceDir = path.join(packageDir, 'desktop')
  const entry = path.join(sourceDir, 'plugin.js')

  let stat: fs.Stats

  try {
    stat = await fs.promises.stat(entry)
  } catch (error) {
    if (!isMissing(error)) {
      console.warn(`[desktop-plugins] cannot read ${packageName}: ${String(error)}`)
    }

    return null
  }

  if (!stat.isFile()) {
    return null
  }

  const target = path.join(appRoot, packageName)
  const existing = await readMarker(target)

  if (fs.existsSync(target)) {
    if (!existing) {
      // A real standalone install has its entry point. A marker-less folder
      // without one is an interrupted unified-package copy: the old copy
      // wrote the marker last, so leaving it here would block every retry.
      if (fs.existsSync(path.join(target, 'plugin.js'))) {
        return null
      }
    }

    if (existing && existing.source === sourceDir && existing.sourceMtimeMs >= stat.mtimeMs) {
      return null
    }
  }

  const marker: DesktopHalfMarker = {
    package: packageName,
    source: sourceDir,
    sourceMtimeMs: stat.mtimeMs,
    ...(await packageOrigin(packageDir))
  }

  await publishDesktopTree(sourceDir, target, staged =>
    fs.promises.writeFile(path.join(staged, PACKAGE_MARKER), JSON.stringify(marker, null, 2) + '\n')
  )

  return target
}

/** Copy `sourceDir` to `target` through a staging sibling (`<parent>/.<name>.staging-*`)
 *  and rename the finished tree into place. `finalize` runs on the staged tree
 *  before publication, so a marker is never missing from a published folder.
 *  Directory replacement is not atomic on every platform Electron supports, but
 *  the complete copy exists before the old one is removed, so a failure leaves
 *  either the old folder or none — never a partial, marker-less one that a
 *  later pass would mistake for a manual install (#112450). */
export async function publishDesktopTree(
  sourceDir: string,
  target: string,
  finalize?: (staged: string) => Promise<void>
): Promise<void> {
  const parent = path.dirname(target)
  const name = path.basename(target)

  await fs.promises.mkdir(parent, { recursive: true })
  const stagingRoot = await fs.promises.mkdtemp(path.join(parent, `.${name}.staging-`))
  const staged = path.join(stagingRoot, name)

  try {
    await fs.promises.cp(sourceDir, staged, { force: true, recursive: true })
    await finalize?.(staged)
    // rename() refuses to replace a non-empty directory, so the old copy goes first.
    await fs.promises.rm(target, { force: true, recursive: true })
    await fs.promises.rename(staged, target)
  } finally {
    await fs.promises.rm(stagingRoot, { force: true, recursive: true })
  }
}

function isMissing(error: unknown): boolean {
  const code = (error as NodeJS.ErrnoException | null)?.code

  return code === 'ENOENT' || code === 'ENOTDIR'
}

/** `true` only when the package's `desktop/plugin.js` is genuinely gone. A
 *  source the app is not ALLOWED to stat (Windows ACL EPERM, a mode-000 folder)
 *  is not an uninstall — pruning its root copy would silently drop the pane. */
async function sourceGone(name: string, entry: string): Promise<boolean> {
  try {
    await fs.promises.stat(entry)

    return false
  } catch (error) {
    if (isMissing(error)) {
      return true
    }

    console.warn(`[desktop-plugins] keeping desktop half of unreadable package ${name}: ${String(error)}`)

    return false
  }
}

/** Walk every local home's `plugins/` root and materialize each package's
 *  desktop half. First home wins for a name that appears in several profiles
 *  (the default home is first). Also drops root copies whose source package
 *  is gone — an uninstalled agent package must not leave a ghost pane. */
export async function reconcileUnifiedDesktopHalves(hermesHome: string, appRoot: string): Promise<string[]> {
  const touched: string[] = []
  const seen = new Set<string>()

  for (const home of await localHomes(hermesHome)) {
    const pluginsRoot = path.join(home, 'plugins')

    for (const name of await listDirs(pluginsRoot)) {
      if (seen.has(name)) {
        continue
      }

      let result: null | string

      try {
        result = await materializeDesktopHalf(path.join(pluginsRoot, name), appRoot, name)
      } catch (error) {
        // One package the app cannot read (Windows ACL EPERM on lstat/copy, a
        // mode-000 folder) must not reject the whole reconcile — the root would
        // never resolve and EVERY desktop plugin would silently stop loading.
        console.warn(`[desktop-plugins] skipping unreadable package ${name}: ${String(error)}`)

        continue
      }

      if (result || fs.existsSync(path.join(pluginsRoot, name, 'desktop', 'plugin.js'))) {
        seen.add(name)
      }

      if (result) {
        touched.push(result)
      }
    }
  }

  for (const name of await listDirs(appRoot)) {
    const dir = path.join(appRoot, name)
    const marker = await readMarker(dir)

    if (marker && (await sourceGone(name, path.join(marker.source, 'plugin.js')))) {
      await fs.promises.rm(dir, { force: true, recursive: true })
      touched.push(dir)
    }
  }

  return touched
}
