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
  /** The package folder is a symlink (a dev checkout linked into `plugins/`):
   *  the copy re-syncs on ANY byte difference and the renderer watches the
   *  source, so saving the checkout hot-reloads the pane. */
  linked?: boolean
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

const rootWriters = new Map<string, Promise<unknown>>()

/** Run `task` once every earlier writer of the same app root has settled.
 *  Every writer of `<root>` (each window's 5 s disk poll and its Plugins page,
 *  the install/uninstall IPC) runs in this one main process, and each one
 *  decides "copy or keep" before it publishes with rm + rename. Two passes
 *  that overlap both copy, one `rename` lands on the other's fresh folder
 *  (ENOTEMPTY), and that pass then skips the package, so a same-named package
 *  in a later profile overwrote the default home's copy. */
export function withDesktopPluginsRoot<T>(root: string, task: () => Promise<T>): Promise<T> {
  const key = path.resolve(root)
  const run = (rootWriters.get(key) ?? Promise.resolve()).then(task, task)
  const settled = run.catch(() => undefined)

  rootWriters.set(key, settled)
  void settled.then(() => {
    if (rootWriters.get(key) === settled) {
      rootWriters.delete(key)
    }
  })

  return run
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

async function listDirs(dir: string, followLinks = false): Promise<string[]> {
  try {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true })

    // `Dirent.isDirectory()` is false for a symlink. A package root follows
    // links to directories: `ln -s ~/src/my-plugin ~/.hermes/plugins/` is how a
    // plugin is developed in place, and the Python loader already follows it.
    const linkedDir = async (name: string) => {
      try {
        return (await fs.promises.stat(path.join(dir, name))).isDirectory()
      } catch {
        return false
      }
    }

    const names: string[] = []

    for (const entry of entries) {
      if (entry.isDirectory() || (followLinks && entry.isSymbolicLink() && (await linkedDir(entry.name)))) {
        names.push(entry.name)
      }
    }

    return names
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
export function migrateProfileScopedDesktopPlugins(hermesHome: string, appRoot: string): Promise<string[]> {
  return withDesktopPluginsRoot(appRoot, () => migrateProfileScoped(hermesHome, appRoot))
}

async function migrateProfileScoped(hermesHome: string, appRoot: string): Promise<string[]> {
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

/** Same bytes on both paths. Unreadable either side answers `false` — a
 *  comparison we cannot make is never evidence of a match. */
async function sameFile(a: string, b: string): Promise<boolean> {
  try {
    const [left, right] = await Promise.all([fs.promises.readFile(a), fs.promises.readFile(b)])

    return left.equals(right)
  } catch {
    return false
  }
}

/** Write the marker into a desktop-half folder. The one place that serializes
 *  it, so the git installer and this reconcile cannot drift. */
export async function writeDesktopHalfMarker(dir: string, marker: DesktopHalfMarker): Promise<void> {
  await fs.promises.writeFile(path.join(dir, PACKAGE_MARKER), JSON.stringify(marker, null, 2) + '\n')
}

/** Where a package's desktop plugin lives, in the order every installer and
 *  the install probe read it (the gateway's `desktop_source` and
 *  `detect_components` agree): the package root when a `plugin.js` sits there,
 *  else its `desktop/` folder. */
const DESKTOP_SOURCE_DIRS = ['.', 'desktop'] as const

/** The folder that is `packageDir`'s desktop plugin, or `null`. */
export function desktopHalfSource(packageDir: string): null | string {
  for (const subdir of DESKTOP_SOURCE_DIRS) {
    try {
      if (fs.statSync(path.join(packageDir, subdir, 'plugin.js')).isFile()) {
        return path.join(packageDir, subdir)
      }
    } catch {
      // Not this layout.
    }
  }

  return null
}

/** `desktopHalfSource` with the entry's stat. A candidate the app may not stat
 *  ends the search: falling through to the other layout would copy a half the
 *  installer would not have picked. */
async function desktopHalfEntry(
  packageDir: string,
  packageName: string
): Promise<null | { sourceDir: string; stat: fs.Stats }> {
  for (const subdir of DESKTOP_SOURCE_DIRS) {
    const sourceDir = path.join(packageDir, subdir)

    try {
      const stat = await fs.promises.stat(path.join(sourceDir, 'plugin.js'))

      if (stat.isFile()) {
        return { sourceDir, stat }
      }
    } catch (error) {
      if (!isMissing(error)) {
        console.warn(`[desktop-plugins] cannot read ${packageName}: ${String(error)}`)

        return null
      }
    }
  }

  return null
}

/** A root-layout half publishes the package root, never its clone's `.git` or
 *  the agent half's `__pycache__` (the gateway's `ROOT_LAYOUT_SKIP`). */
const ROOT_LAYOUT_SKIP = new Set(['.git', '__pycache__'])

const rootLayoutFilter = (src: string) => !ROOT_LAYOUT_SKIP.has(path.basename(src))

/** Copy one unified package's desktop half into the app root as
 *  `<appRoot>/<packageName>/`, stamping the marker: its `desktop/` folder, or
 *  the package root (minus `ROOT_LAYOUT_SKIP`) when `plugin.js` sits at the root, as the
 *  installers publish a root-layout repository. Skips when the root copy is
 *  already current for this source; replaces it when the source is newer.
 *  Returns the target path when a copy (or an adoption) happened. */
export async function materializeDesktopHalf(
  packageDir: string,
  appRoot: string,
  packageName = path.basename(packageDir)
): Promise<null | string> {
  const found = await desktopHalfEntry(packageDir, packageName)

  if (!found) {
    return null
  }

  const { sourceDir, stat } = found
  const entry = path.join(sourceDir, 'plugin.js')
  const rootLayout = sourceDir === path.join(packageDir, '.')

  const target = path.join(appRoot, packageName)
  const existing = await readMarker(target)
  const linked = (await fs.promises.lstat(packageDir)).isSymbolicLink()
  // A link at `<appRoot>/<name>` is never a copy this pass may keep or stamp: a marker-less one is
  // the user's own linked plugin, and one carrying a marker is an earlier copy of a linked checkout
  // that kept the link instead of the files, whose marker sits inside that checkout.
  const targetIsLink = await isLink(target)

  if (targetIsLink && !existing) {
    return null
  }

  const marker: DesktopHalfMarker = {
    package: packageName,
    source: sourceDir,
    sourceMtimeMs: stat.mtimeMs,
    ...(linked ? { linked } : {}),
    ...(await packageOrigin(packageDir))
  }

  if (fs.existsSync(target)) {
    if (!existing) {
      // A marker-less folder carrying the entry point is either a standalone
      // plugin the user installed on purpose (never touch it) or a desktop half
      // this app copied out before it stamped markers — `installDesktopPluginFromGit`
      // published without one, which left the Plugins page waiting on "copying…"
      // beside a second, already-enabled row, forever, because this function
      // then refused the folder on every pass.
      //
      // Identical entry points tell the two apart: our own copy of this
      // package's half still matches it byte for byte, so adopting it is a
      // no-op on disk — stamp the marker in place and the row pairs, with the
      // opt-in posture a marker implies. Anything that differs is the user's
      // and is left exactly as it was (#112450). A marker-less folder with no
      // entry point is an interrupted copy (the marker is written last) and is
      // replaced as before.
      if (fs.existsSync(path.join(target, 'plugin.js'))) {
        if (await sameFile(path.join(target, 'plugin.js'), entry)) {
          await writeDesktopHalfMarker(target, marker)

          return target
        }

        return null
      }
    }

    // An installed package re-copies when its source is newer. A linked dev
    // checkout compares bytes: mtime alone misses a `cp -p` / `git stash pop`
    // that lands an older timestamp, and the copy would stay stale silently.
    if (
      existing &&
      !targetIsLink &&
      existing.source === sourceDir &&
      Boolean(existing.linked) === linked &&
      (linked ? await sameFile(path.join(target, 'plugin.js'), entry) : existing.sourceMtimeMs >= stat.mtimeMs)
    ) {
      return null
    }
  }

  // A package folder symlinked in by hand is the user's own development checkout and its desktop/
  // folder is copied as it stands. An installed package is checked like a fresh clone (a link inside
  // it publishes host files), and so is a root-layout half, linked or not: it publishes the whole
  // package root, agent payload included.
  await publishDesktopTree(
    { dir: sourceDir, plainWithin: linked && !rootLayout ? null : packageDir },
    target,
    staged => writeDesktopHalfMarker(staged, marker),
    rootLayout ? rootLayoutFilter : undefined
  )

  return target
}

/** The whole published folder (the Webapp's host-side install applies the same cap); the loader reads
 *  at most a 16 MB `plugin.js` from it. */
const DESKTOP_TREE_MAX_BYTES = 64 * 1024 * 1024

/** What `publishDesktopTree` copies. `plainWithin` names the clone or installed package the tree must
 *  stay inside as plain files and folders; `null` only for a development checkout the user linked in. */
export interface DesktopTreeSource {
  dir: string
  plainWithin: null | string
}

/** Refuse a desktop tree that reaches outside its package or is too big to copy. A symlink anywhere on
 *  the way to it, or inside it, would publish whatever it points at on this machine (or a dangling link
 *  into a deleted clone) as part of the plugin, so the tree must be plain files and folders. */
async function assertPlainDesktopTree(
  root: string,
  sourceDir: string,
  filter: (src: string) => boolean = () => true
): Promise<void> {
  const realRoot = await fs.promises.realpath(root)

  if ((await fs.promises.realpath(sourceDir)) !== path.join(realRoot, path.relative(root, sourceDir))) {
    throw new Error('The desktop plugin folder is a symbolic link; it is not installed.')
  }

  let total = 0
  const pending = [sourceDir]

  while (pending.length > 0) {
    const dir = pending.pop()!

    for (const entry of await fs.promises.readdir(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name)

      // What the copy leaves out (a root-layout half's `.git`, `__pycache__`) is not published.
      if (!filter(full)) {
        continue
      }

      if (entry.isSymbolicLink()) {
        throw new Error(
          `The desktop plugin contains a symbolic link (${path.relative(sourceDir, full)}); it is not installed.`
        )
      }

      if (entry.isDirectory()) {
        pending.push(full)
      } else {
        total += (await fs.promises.lstat(full)).size

        if (total > DESKTOP_TREE_MAX_BYTES) {
          throw new Error(`The desktop plugin is larger than ${DESKTOP_TREE_MAX_BYTES / (1024 * 1024)} MB.`)
        }
      }
    }
  }
}

/** Copy `source.dir` to `target` through a staging sibling (`<parent>/.<name>.staging-*`)
 *  and rename the finished tree into place. Every route that publishes a desktop tree (git install,
 *  package reconcile) comes through here, so the plain-tree check runs once for all of them, before
 *  anything is staged. `finalize` runs on the staged tree before publication, so a marker is never
 *  missing from a published folder.
 *  Directory replacement is not atomic on every platform Electron supports, but
 *  the complete copy exists before the old one is removed, so a failure leaves
 *  either the old folder or none — never a partial, marker-less one that a
 *  later pass would mistake for a manual install (#112450). */
export async function publishDesktopTree(
  source: DesktopTreeSource,
  target: string,
  finalize?: (staged: string) => Promise<void>,
  filter?: (src: string) => boolean
): Promise<void> {
  if (source.plainWithin !== null) {
    await assertPlainDesktopTree(source.plainWithin, source.dir, filter)
  }

  const parent = path.dirname(target)
  const name = path.basename(target)

  await fs.promises.mkdir(parent, { recursive: true })
  const stagingRoot = await fs.promises.mkdtemp(path.join(parent, `.${name}.staging-`))
  const staged = path.join(stagingRoot, name)

  try {
    // From the resolved folder: `cp` keeps a source that is itself a link (a dev checkout linked into
    // `plugins/`) as a link, so the published "copy" was the checkout and the marker landed in it.
    await fs.promises.cp(await fs.promises.realpath(source.dir), staged, { filter, force: true, recursive: true })
    await finalize?.(staged)
    // rename() refuses to replace a non-empty directory, so the old copy goes first.
    await fs.promises.rm(target, { force: true, recursive: true })
    await fs.promises.rename(staged, target)
  } finally {
    await fs.promises.rm(stagingRoot, { force: true, recursive: true })
  }
}

async function isLink(target: string): Promise<boolean> {
  try {
    return (await fs.promises.lstat(target)).isSymbolicLink()
  } catch {
    return false
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
  return (await reconcileDesktopHalvesReport(hermesHome, appRoot)).published
}

/** A package whose desktop half this pass did not publish, and why (the plain-tree check, an unreadable folder). */
export interface RefusedDesktopHalf {
  name: string
  reason: string
}

/** `published`: the root copies written or pruned. `refused`: packages whose desktop half was not published. */
export interface DesktopReconcileReport {
  published: string[]
  refused: RefusedDesktopHalf[]
}

/** `reconcileUnifiedDesktopHalves`, also naming each package it refused, so an install that just put one
 *  down can report its desktop half as failed instead of done. */
export function reconcileDesktopHalvesReport(hermesHome: string, appRoot: string): Promise<DesktopReconcileReport> {
  return withDesktopPluginsRoot(appRoot, () => reconcilePass(hermesHome, appRoot))
}

async function reconcilePass(hermesHome: string, appRoot: string): Promise<DesktopReconcileReport> {
  const touched: string[] = []
  const refused: RefusedDesktopHalf[] = []
  const seen = new Set<string>()

  for (const home of await localHomes(hermesHome)) {
    const pluginsRoot = path.join(home, 'plugins')

    for (const name of await listDirs(pluginsRoot, true)) {
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
        // A desktop half refused by the plain-tree check lands here too.
        console.warn(`[desktop-plugins] skipping package ${name}: ${String(error)}`)
        refused.push({ name, reason: error instanceof Error ? error.message : String(error) })

        continue
      }

      if (result || desktopHalfSource(path.join(pluginsRoot, name))) {
        seen.add(name)
      }

      if (result) {
        touched.push(result)
      }
    }
  }

  // Links count too: a marker read through one is an earlier copy of a linked checkout that kept the
  // link, and `rm` drops only the link, never the folder it points at.
  for (const name of await listDirs(appRoot, true)) {
    const dir = path.join(appRoot, name)
    const marker = await readMarker(dir)

    if (marker && (await sourceGone(name, path.join(marker.source, 'plugin.js')))) {
      await fs.promises.rm(dir, { force: true, recursive: true })
      touched.push(dir)
    }
  }

  // A same-named package in a later home that did publish owns the root copy; its refused twin is moot.
  return { published: touched, refused: refused.filter(entry => !seen.has(entry.name)) }
}
