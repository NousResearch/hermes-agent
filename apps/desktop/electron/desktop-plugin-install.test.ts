import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

import { afterEach, describe, expect, it } from 'vitest'

import {
  desktopPluginFolderName,
  detectPluginComponents,
  findDesktopEntry,
  installDesktopPluginFromGit,
  probePluginRepo,
  resolvePluginGitUrl,
  resolveSubdirWithin,
  setGitRunnerForTests
} from './desktop-plugin-install'
import { PACKAGE_MARKER, reconcileUnifiedDesktopHalves } from './desktop-plugins-root'

// The pairing of a marker-stamped desktop half with its agent row lives in the
// renderer's own project (src/app/capabilities/plugins/plugin-packages.test.ts,
// 'pairs a catalog alias install with its manifest-named agent half'): this
// electron project excludes `src`, so importing the renderer module here would
// walk its whole graph into a tsc project that must not include it.

const here = path.dirname(fileURLToPath(import.meta.url))

function mkdtemp(prefix: string) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix))
}

describe('resolvePluginGitUrl', () => {
  it('maps owner/repo shorthand to github git url', () => {
    expect(resolvePluginGitUrl('NousResearch/hermes-example-plugins')).toEqual({
      gitUrl: 'https://github.com/NousResearch/hermes-example-plugins.git',
      subdir: null
    })
  })

  it('supports monorepo subdir shorthand', () => {
    expect(resolvePluginGitUrl('owner/repo/plugins/foo')).toEqual({
      gitUrl: 'https://github.com/owner/repo.git',
      subdir: 'plugins/foo'
    })
  })

  it('supports hash subdir fragment', () => {
    expect(resolvePluginGitUrl('https://github.com/o/r.git#nested/plugin')).toEqual({
      gitUrl: 'https://github.com/o/r.git',
      subdir: 'nested/plugin'
    })
  })
})

describe('desktopPluginFolderName', () => {
  it('uses the repo name for a root-level plugin, not the clone path', () => {
    expect(desktopPluginFolderName('https://github.com/o/my-plugin.git', null)).toBe('my-plugin')
  })

  it('uses the last meaningful subdir, not a generic desktop folder', () => {
    expect(desktopPluginFolderName('https://github.com/o/monorepo.git', 'plugins/alerts/desktop')).toBe('alerts')
  })
})

describe('resolveSubdirWithin', () => {
  it('rejects path traversal', () => {
    const root = mkdtemp('hermes-plugin-root-')

    expect(() => resolveSubdirWithin(root, '../escape')).toThrow(/escapes/)
  })
})

describe('findDesktopEntry', () => {
  it('finds root plugin.js', () => {
    const root = mkdtemp('hermes-plugin-detect-')
    fs.mkdirSync(path.join(root, 'desktop'), { recursive: true })
    fs.writeFileSync(path.join(root, 'plugin.js'), 'export default {}')

    expect(findDesktopEntry(root)).toEqual({ entryFile: path.join(root, 'plugin.js'), sourceSubdir: '.' })
  })

  it('finds desktop/plugin.js', () => {
    const root = mkdtemp('hermes-plugin-detect-')
    fs.mkdirSync(path.join(root, 'desktop'), { recursive: true })
    fs.writeFileSync(path.join(root, 'desktop', 'plugin.js'), 'export default {}')

    expect(findDesktopEntry(root)).toEqual({
      entryFile: path.join(root, 'desktop', 'plugin.js'),
      sourceSubdir: 'desktop'
    })
  })
})

describe('detectPluginComponents', () => {
  const roots: string[] = []

  afterEach(() => {
    for (const root of roots.splice(0)) {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('detects agent-only layout', async () => {
    const root = mkdtemp('hermes-plugin-agent-')
    roots.push(root)
    fs.writeFileSync(path.join(root, 'plugin.yaml'), 'name: hello-agent\n')
    fs.writeFileSync(path.join(root, '__init__.py'), 'def register(ctx): pass\n')

    await expect(detectPluginComponents(root)).resolves.toMatchObject({
      agent: true,
      desktop: false,
      agentName: 'hello-agent'
    })
  })

  it('detects dual layout', async () => {
    const root = mkdtemp('hermes-plugin-dual-')
    roots.push(root)
    fs.mkdirSync(path.join(root, 'desktop'), { recursive: true })
    fs.writeFileSync(path.join(root, 'plugin.yaml'), 'name: dual\n')
    fs.writeFileSync(path.join(root, '__init__.py'), 'def register(ctx): pass\n')
    fs.writeFileSync(path.join(root, 'desktop', 'plugin.js'), 'export default { id: "dual-ui" }')

    await expect(detectPluginComponents(root)).resolves.toMatchObject({
      agent: true,
      desktop: true,
      agentName: 'dual',
      desktopName: 'desktop'
    })
  })
})

describe('probePluginRepo', () => {
  const roots: string[] = []

  afterEach(() => {
    for (const root of roots.splice(0)) {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('probes a monorepo subdirectory through the sparse partial clone', async () => {
    const repo = mkdtemp('hermes-plugin-monorepo-')
    roots.push(repo)
    const git = (...args: string[]) => execFileSync('git', args, { cwd: repo, stdio: 'pipe' })
    const plugin = path.join(repo, 'integrations', 'hermes')
    fs.mkdirSync(plugin, { recursive: true })
    fs.writeFileSync(path.join(plugin, 'plugin.yaml'), 'name: nested-agent\n')
    fs.writeFileSync(path.join(plugin, '__init__.py'), 'def register(ctx): pass\n')
    fs.writeFileSync(path.join(repo, 'unrelated.bin'), 'x'.repeat(4096))
    git('init', '-q')
    git('config', 'uploadpack.allowFilter', 'true')
    git('add', '.')
    git('-c', 'user.email=fixture@example.com', '-c', 'user.name=Fixture', 'commit', '-qm', 'init')

    const result = await probePluginRepo('git', `${pathToFileURL(repo).href}#integrations/hermes`)

    expect(result).toMatchObject({
      ok: true,
      agent: true,
      agentName: 'nested-agent',
      sha: git('rev-parse', 'HEAD').toString().trim()
    })
  })

  it('probes a catalog pick at its pin when the default branch tip lost the plugin folder', async () => {
    // The unbrowse catalog entry: its reviewed pin is not on the default branch, which has no
    // plugins/hermes, so probing the tip refused an install the backend would do at the pin.
    const repo = mkdtemp('hermes-plugin-pinned-')
    roots.push(repo)
    const git = (...args: string[]) => execFileSync('git', args, { cwd: repo, stdio: 'pipe' }).toString().trim()
    const plugin = path.join(repo, 'plugins', 'hermes')
    fs.mkdirSync(plugin, { recursive: true })
    fs.writeFileSync(path.join(plugin, 'plugin.yaml'), 'name: pinned-agent\n')
    fs.writeFileSync(path.join(plugin, '__init__.py'), 'def register(ctx): pass\n')
    git('init', '-q')
    git('config', 'uploadpack.allowFilter', 'true')
    git('add', '.')
    git('-c', 'user.email=fixture@example.com', '-c', 'user.name=Fixture', 'commit', '-qm', 'plugin')
    const pin = git('rev-parse', 'HEAD')
    fs.rmSync(path.join(repo, 'plugins'), { recursive: true })
    fs.writeFileSync(path.join(repo, 'README.md'), 'moved elsewhere\n')
    git('add', '-A')
    git('-c', 'user.email=fixture@example.com', '-c', 'user.name=Fixture', 'commit', '-qm', 'drop plugin')
    const identifier = `${pathToFileURL(repo).href}#plugins/hermes`

    expect(await probePluginRepo('git', identifier)).toMatchObject({
      ok: false,
      error: "Plugin subdirectory 'plugins/hermes' does not exist in the repository."
    })
    expect(await probePluginRepo('git', identifier, { ref: pin.toUpperCase() })).toMatchObject({
      ok: true,
      agent: true,
      agentName: 'pinned-agent'
    })
  }, 30_000)

  it.each(['main', 'abc1234', '0'.repeat(39), '--upload-pack=touch /tmp/x'])(
    'refuses a probe pin that is not a full commit SHA (%s)',
    async ref => {
      const result = await probePluginRepo('git', 'https://example.invalid/owner/repo.git', { ref })

      expect(result).toMatchObject({ ok: false, error: '--ref must be a full 40-character commit SHA.' })
    }
  )
})

describe('installDesktopPluginFromGit', () => {
  const roots: string[] = []

  afterEach(() => {
    for (const root of roots.splice(0)) {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  /** A git repo whose root is one plugin: `desktop/plugin.js` plus, when
   *  `agentName` is given, the agent half that makes it a unified package. */
  function pluginRepo(agentName: null | string): string {
    const repo = mkdtemp('hermes-plugin-install-')
    roots.push(repo)
    const git = (...args: string[]) => execFileSync('git', args, { cwd: repo, stdio: 'pipe' })

    fs.mkdirSync(path.join(repo, 'desktop'), { recursive: true })
    fs.writeFileSync(path.join(repo, 'desktop', 'plugin.js'), 'export function register() {}\n')

    if (agentName) {
      fs.writeFileSync(path.join(repo, 'plugin.yaml'), `name: ${agentName}\n`)
      fs.writeFileSync(path.join(repo, '__init__.py'), 'def register(ctx): pass\n')
    }

    git('init', '-q')
    git('add', '.')
    git('-c', 'user.email=fixture@example.com', '-c', 'user.name=Fixture', 'commit', '-qm', 'init')

    return repo
  }

  it('installs the commit a manual probe inspected after the branch tip moves', async () => {
    // Install from Git with no pin: the probe resolves the tip once and the install fetches that
    // commit, so a push between review and Install cannot change what gets published.
    const repo = pluginRepo(null)
    const identifier = pathToFileURL(repo).href
    const entry = path.join(repo, 'desktop', 'plugin.js')
    const probe = await probePluginRepo('git', identifier)

    expect(probe).toMatchObject({ ok: true, desktop: true })
    expect(probe.sha).toMatch(/^[0-9a-f]{40}$/)
    fs.writeFileSync(entry, 'export const version = "unreviewed"\n')
    execFileSync(
      'git',
      ['-c', 'user.email=fixture@example.com', '-c', 'user.name=Fixture', 'commit', '-qam', 'moved'],
      { cwd: repo, stdio: 'pipe' }
    )

    const appRoot = mkdtemp('hermes-plugin-root-')
    roots.push(appRoot)
    const result = await installDesktopPluginFromGit('git', identifier, appRoot, false, { ref: probe.sha })

    expect(result.ok).toBe(true)
    expect(fs.readFileSync(path.join(appRoot, String(result.pluginName), 'plugin.js'), 'utf8')).toBe(
      'export function register() {}\n'
    )
  })

  it.each(['', 'integrations/widget'])(
    'pins catalog bytes and provenance without destructive ref failures (%s)',
    async subdir => {
      const repo = pluginRepo(subdir ? null : 'manifest-name')
      const git = (...args: string[]) =>
        execFileSync('git', args, { cwd: repo, encoding: 'utf8', stdio: 'pipe' }).trim()
      const pluginRoot = path.join(repo, subdir)
      if (subdir) {
        fs.mkdirSync(pluginRoot, { recursive: true })
        fs.renameSync(path.join(repo, 'desktop'), path.join(pluginRoot, 'desktop'))
        fs.writeFileSync(path.join(pluginRoot, 'plugin.yaml'), JSON.stringify({ name: 'manifest-name' }))
        fs.writeFileSync(path.join(pluginRoot, '__init__.py'), 'def register(ctx): pass')
      }
      const entry = path.join(pluginRoot, 'desktop', 'plugin.js')
      const olderBytes = 'export const version = "reviewed"'
      const newerBytes = 'export const version = "unreviewed"'
      fs.writeFileSync(entry, olderBytes)
      git('add', '.')
      git('-c', 'user.email=fixture@example.com', '-c', 'user.name=Fixture', 'commit', '--amend', '-qm', 'reviewed')
      const sha = git('rev-parse', 'HEAD')
      git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com', 'tag', '-a', 'reviewed', '-m', 'reviewed')
      const requestedRef = subdir ? git('rev-parse', 'reviewed') : sha
      fs.writeFileSync(entry, newerBytes)
      git('add', '.')
      git('-c', 'user.email=fixture@example.com', '-c', 'user.name=Fixture', 'commit', '-qm', 'unreviewed')
      git('config', 'uploadpack.allowFilter', 'true')
      const identifier = pathToFileURL(repo).href + (subdir ? `#${subdir}` : '')
      const appRoot = mkdtemp('hermes-plugin-catalog-')
      roots.push(appRoot)
      const catalogName = 'catalog-widget'
      const target = path.join(appRoot, 'manifest-name')

      const result = await installDesktopPluginFromGit('git', identifier, appRoot, false, {
        ref: requestedRef.toUpperCase(),
        catalogName
      })

      expect(result.ok, result.error).toBe(true)
      expect(fs.readFileSync(path.join(result.path!, 'plugin.js'), 'utf8')).toBe(olderBytes)
      expect(result).toMatchObject({ pluginName: 'manifest-name', path: target })
      const markerPath = path.join(target, PACKAGE_MARKER)
      const markerBytes = fs.readFileSync(markerPath, 'utf8')
      expect(JSON.parse(markerBytes)).toMatchObject({
        package: 'manifest-name',
        catalogName,
        sha,
        repo: identifier,
        source: target
      })

      // Malformed refs must be rejected before ANY git process runs, and a
      // valid ref must reach git — observable by counting invocations through
      // the runner seam, not by a missing binary (whose 'could not invoke git'
      // failure is indistinguishable from a pre-Git rejection).
      const withCountingRunner = async (run: (invocations: string[][]) => Promise<void>) => {
        const invocations: string[][] = []
        setGitRunnerForTests(async (_gitBin, args) => {
          invocations.push(args)
          return { code: 128, stderr: 'injected runner' }
        })
        try {
          await run(invocations)
        } finally {
          setGitRunnerForTests(null)
        }
      }

      for (const ref of [undefined, 'HEAD', sha.slice(0, 12), `${sha}\n`, '', '--upload-pack=other']) {
        await withCountingRunner(async invocations => {
          const failed = await installDesktopPluginFromGit('git', identifier, appRoot, true, { ref, catalogName })
          expect(failed.ok, ref).toBe(false)
          expect(invocations, ref).toEqual([])
          expect(failed.error, ref).toMatch(/40.*SHA/i)
          expect(fs.readFileSync(path.join(target, 'plugin.js'), 'utf8')).toBe(olderBytes)
          expect(fs.readFileSync(markerPath, 'utf8')).toBe(markerBytes)
        })
      }
      // A well-formed-but-unknown 40-hex ref passes the guard and fails IN git,
      // never destructively: the published half and its marker stay intact.
      await withCountingRunner(async invocations => {
        const failed = await installDesktopPluginFromGit('git', identifier, appRoot, true, {
          ref: '0'.repeat(40),
          catalogName
        })
        expect(failed.ok).toBe(false)
        expect(invocations.length).toBeGreaterThan(0)
        expect(failed.error).toMatch(/git.*failed/i)
        expect(fs.readFileSync(path.join(target, 'plugin.js'), 'utf8')).toBe(olderBytes)
        expect(fs.readFileSync(markerPath, 'utf8')).toBe(markerBytes)
      })
      // The guard must not merely refuse everything: a well-formed ref DOES
      // reach git (and the real install above already proved the happy path).
      await withCountingRunner(async invocations => {
        const failed = await installDesktopPluginFromGit('git', identifier, appRoot, true, { ref: sha, catalogName })
        expect(failed.ok).toBe(false)
        expect(invocations.length).toBeGreaterThan(0)
        expect(failed.error).toMatch(/git.*failed/i)
      })
      // The catalog-name rule matches the backend's `_sanitize_plugin_name`
      // reject set exactly: separators, traversal and the root itself.
      for (const unsafeName of ['../escape', '..\\escape', 'sub/dir', '.', '..', '']) {
        await withCountingRunner(async invocations => {
          const failed = await installDesktopPluginFromGit('git', identifier, appRoot, true, {
            ref: sha,
            catalogName: unsafeName
          })
          expect(failed.ok, unsafeName).toBe(false)
          expect(failed.error).toMatch(/catalog.*name/i)
          expect(invocations, unsafeName).toEqual([])
          expect(fs.readFileSync(path.join(target, 'plugin.js'), 'utf8')).toBe(olderBytes)
        })
      }

      // The four-argument path still follows HEAD and its original naming rules.
      const unpinned = await installDesktopPluginFromGit('git', identifier, appRoot, true)
      expect(unpinned.ok, unpinned.error).toBe(true)
      expect(fs.readFileSync(path.join(unpinned.path!, 'plugin.js'), 'utf8')).toBe(newerBytes)
      expect(unpinned.pluginName).toBe('manifest-name')
    },
    120_000
  )

  it.each(['name: manifest-name # package identity\n', '{"name":"manifest-name"}', 'name: "manifest-name"\n'])(
    'pairs a catalog alias with its manifest-named agent half (%s)',
    async manifest => {
      const repo = pluginRepo('manifest-name')
      fs.writeFileSync(path.join(repo, 'plugin.yaml'), manifest)
      const git = (...args: string[]) => execFileSync('git', args, { cwd: repo, encoding: 'utf8' }).trim()
      git('add', '.')
      git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com', 'commit', '--amend', '-qm', 'manifest')
      const home = mkdtemp('hermes-plugin-alias-')
      roots.push(home)
      const appRoot = path.join(home, 'desktop-plugins')
      const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, false, {
        ref: git('rev-parse', 'HEAD'),
        catalogName: 'catalog-alias'
      })
      expect(result.ok, result.error).toBe(true)
      const marker = JSON.parse(fs.readFileSync(path.join(result.path!, PACKAGE_MARKER), 'utf8'))
      // The join the Plugins page makes: the marker's `package` is the MANIFEST
      // identity, so `mergePluginPackages` pairs this half with the agent row
      // of the same name (pinned in plugin-packages.test.ts, 'pairs a catalog
      // alias install with its manifest-named agent half').
      expect(marker).toMatchObject({ package: 'manifest-name', catalogName: 'catalog-alias' })
      const localDesktop = path.join(home, 'plugins', 'manifest-name', 'desktop')
      fs.mkdirSync(localDesktop, { recursive: true })
      fs.writeFileSync(path.join(localDesktop, 'plugin.js'), 'local agent bytes')
      await reconcileUnifiedDesktopHalves(home, appRoot)
      expect(fs.readdirSync(appRoot)).toEqual(['manifest-name'])
      expect(fs.readFileSync(path.join(appRoot, 'manifest-name', 'plugin.js'), 'utf8')).toBe('local agent bytes')
    }
  )

  it.each(['name: ../escape', 'name: NUL/is/a/path', 'name: sub/dir', 'name: 123', 'name: [broken', '- not-a-mapping'])(
    'rejects invalid manifest identity without replacing a published half (%s)',
    async manifest => {
      const repo = pluginRepo('valid-name')
      const appRoot = mkdtemp('hermes-plugin-invalid-')
      roots.push(appRoot)
      const installed = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)
      expect(installed.ok, installed.error).toBe(true)
      const original = fs.readFileSync(path.join(installed.path!, 'plugin.js'), 'utf8')
      fs.writeFileSync(path.join(repo, 'plugin.yaml'), manifest)
      execFileSync('git', ['add', '.'], { cwd: repo })
      execFileSync(
        'git',
        ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com', 'commit', '-qm', 'invalid'],
        { cwd: repo }
      )
      const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, true)
      expect(result.ok).toBe(false)
      expect(fs.readdirSync(appRoot)).toEqual(['valid-name'])
      expect(fs.readFileSync(path.join(installed.path!, 'plugin.js'), 'utf8')).toBe(original)
    }
  )

  it('never deletes or writes outside the plugins root when a force reinstall names an escaping manifest', async () => {
    // `name: ../../victim` once joined onto the plugins root; with Force reinstall the installer
    // removed the escaped folder and published the plugin there.
    const repo = pluginRepo('../../victim')
    const home = mkdtemp('hermes-plugin-containment-')
    roots.push(home)
    const appRoot = path.join(home, 'nested', 'desktop-plugins')
    const sentinel = path.join(home, 'victim', 'sentinel.txt')
    fs.mkdirSync(appRoot, { recursive: true })
    fs.mkdirSync(path.dirname(sentinel), { recursive: true })
    fs.writeFileSync(sentinel, 'outside')

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, true)

    expect(result.ok).toBe(false)
    expect(fs.readFileSync(sentinel, 'utf8')).toBe('outside')
    expect(fs.existsSync(path.join(home, 'victim', 'plugin.js'))).toBe(false)
    expect(fs.readdirSync(appRoot)).toEqual([])
  })

  // The manifest name rule must match the backend's `_sanitize_plugin_name`
  // exactly: the agent half installs under names like these without complaint
  // (a single segment is enough), so a stricter Desktop-side rule would reject
  // a package the backend accepts and break the paired-install surface.
  it.each(['spaced name', 'Mixed Case', 'héllo', '.hidden', 'NUL', 'con', 'trailing.'])(
    'installs the desktop half under a manifest name the backend accepts (%s)',
    async name => {
      const repo = pluginRepo(name)
      const appRoot = mkdtemp('hermes-plugin-backend-name-')
      roots.push(appRoot)

      const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)

      expect(result.ok, result.error).toBe(true)
      expect(result.pluginName).toBe(name)
      const marker = JSON.parse(fs.readFileSync(path.join(result.path!, PACKAGE_MARKER), 'utf8'))
      expect(marker.package).toBe(name)
    }
  )

  it.each(['', 'catalog/widget'])(
    'uses matching probe/install fallbacks when a manifest has no name (%s)',
    async subdir => {
      const repo = pluginRepo('temporary')
      const root = path.join(repo, subdir)
      if (subdir) {
        fs.mkdirSync(root, { recursive: true })
        for (const entry of ['desktop', '__init__.py', 'plugin.yaml']) {
          fs.renameSync(path.join(repo, entry), path.join(root, entry))
        }
      }
      fs.writeFileSync(path.join(root, 'plugin.yaml'), 'description: unnamed package\n')
      execFileSync('git', ['add', '.'], { cwd: repo })
      execFileSync(
        'git',
        ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com', 'commit', '-qm', 'unnamed'],
        { cwd: repo }
      )
      const appRoot = mkdtemp('hermes-plugin-unnamed-')
      roots.push(appRoot)
      const identifier = pathToFileURL(repo).href + (subdir ? `#${subdir}` : '')
      const result = await installDesktopPluginFromGit('git', identifier, appRoot, false, {
        ref: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim(),
        catalogName: 'catalog-alias'
      })
      expect(result).toMatchObject({ ok: true, pluginName: subdir ? 'widget' : path.basename(repo) })
      const probe = await probePluginRepo('git', identifier)
      expect(probe).toMatchObject({ ok: true, agentName: result.pluginName })
    }
  )

  it('stamps the package marker on a unified package half and names the folder after the agent package', async () => {
    // Without the marker the Plugins page has no evidence that this copy is the
    // agent row's desktop half: the row sits on "copying…" while the copy shows
    // up as a second, default-enabled standalone row, and reconcile refuses to
    // re-copy the folder ever again.
    const repo = pluginRepo('hermes-talk')
    const appRoot = mkdtemp('hermes-plugin-root-')
    roots.push(appRoot)

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)

    expect(result).toMatchObject({ ok: true, pluginName: 'hermes-talk' })
    const marker = JSON.parse(fs.readFileSync(path.join(appRoot, 'hermes-talk', PACKAGE_MARKER), 'utf8'))
    expect(marker.package).toBe('hermes-talk')
    expect(marker.repo).toBe(pathToFileURL(repo).href)
  })

  it('keeps a git-installed unified half when no local agent package exists', async () => {
    // Remote backends (and a Desktop-only install) never have
    // plugins/<name>/desktop locally. The marker used to name the temp clone,
    // which this function deletes, so the next reconcile ghost-pruned the half.
    const repo = pluginRepo('hermes-talk')
    const home = mkdtemp('hermes-plugin-home-')
    roots.push(home)
    const appRoot = path.join(home, 'desktop-plugins')

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)

    expect(result).toMatchObject({ ok: true, pluginName: 'hermes-talk' })
    const published = path.join(appRoot, 'hermes-talk')
    const marker = JSON.parse(fs.readFileSync(path.join(published, PACKAGE_MARKER), 'utf8'))
    expect(marker.source).toBe(published)
    expect(fs.existsSync(path.join(marker.source, 'plugin.js'))).toBe(true)

    const touched = await reconcileUnifiedDesktopHalves(home, appRoot)

    expect(fs.existsSync(path.join(published, 'plugin.js'))).toBe(true)
    expect(touched).not.toContain(published)
  })

  it('lets a local agent package replace the git-installed half on reconcile', async () => {
    const repo = pluginRepo('hermes-talk')
    const home = mkdtemp('hermes-plugin-home-')
    roots.push(home)
    const appRoot = path.join(home, 'desktop-plugins')

    await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)

    const packageDesktop = path.join(home, 'plugins', 'hermes-talk', 'desktop')
    fs.mkdirSync(packageDesktop, { recursive: true })
    fs.writeFileSync(path.join(packageDesktop, 'plugin.js'), 'from the agent package\n')

    await reconcileUnifiedDesktopHalves(home, appRoot)

    expect(fs.readFileSync(path.join(appRoot, 'hermes-talk', 'plugin.js'), 'utf8')).toBe('from the agent package\n')
    const marker = JSON.parse(fs.readFileSync(path.join(appRoot, 'hermes-talk', PACKAGE_MARKER), 'utf8'))
    expect(marker.source).toBe(packageDesktop)
  })

  it('leaves a desktop-only repo unmarked so it stays a standalone plugin', async () => {
    const repo = pluginRepo(null)
    const appRoot = mkdtemp('hermes-plugin-root-')
    roots.push(appRoot)

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, false, {
      ref: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim(),
      catalogName: 'standalone-catalog'
    })

    expect(result.ok).toBe(true)
    expect(fs.existsSync(path.join(appRoot, String(result.pluginName), 'plugin.js'))).toBe(true)
    expect(fs.existsSync(path.join(appRoot, String(result.pluginName), PACKAGE_MARKER))).toBe(false)
  })
})
