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
  resolveSubdirWithin
} from './desktop-plugin-install'
import { PACKAGE_MARKER, reconcileUnifiedDesktopHalves } from './desktop-plugins-root'
import { mergePluginPackages } from '../src/app/capabilities/plugins/plugin-packages'
import type { PluginRecord } from '../src/contrib/plugins-store'
import type { AgentPluginRow } from '../src/store/agent-plugins'

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

    expect(result).toMatchObject({ ok: true, agent: true, agentName: 'nested-agent' })
  })
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

      for (const ref of [undefined, '0'.repeat(40), 'HEAD', sha.slice(0, 12), `${sha}\n`, '', '--upload-pack=other']) {
        // A missing executable proves malformed refs are rejected before any Git invocation.
        const gitBin = ref === undefined || ref === '0'.repeat(40) ? 'git' : path.join(repo, 'missing-git')
        const failed = await installDesktopPluginFromGit(gitBin, identifier, appRoot, true, { ref, catalogName })
        expect(failed.ok, ref).toBe(false)
        expect(failed.error).toMatch(ref === '0'.repeat(40) ? /git.*failed/i : /40.*SHA/i)
        expect(fs.readFileSync(path.join(target, 'plugin.js'), 'utf8')).toBe(olderBytes)
        expect(fs.readFileSync(markerPath, 'utf8')).toBe(markerBytes)
      }
      for (const unsafeName of ['../escape', '..\\escape', '.', '..', 'C:escape', '', 'trailing.', 'NUL']) {
        const failed = await installDesktopPluginFromGit(path.join(repo, 'missing-git'), identifier, appRoot, true, {
          ref: sha,
          catalogName: unsafeName
        })
        expect(failed.ok, unsafeName).toBe(false)
        expect(failed.error).toMatch(/catalog.*name/i)
        expect(fs.readFileSync(path.join(target, 'plugin.js'), 'utf8')).toBe(olderBytes)
      }

      // The four-argument path still follows HEAD and its original naming rules.
      const unpinned = await installDesktopPluginFromGit('git', identifier, appRoot, true)
      expect(unpinned.ok, unpinned.error).toBe(true)
      expect(fs.readFileSync(path.join(unpinned.path!, 'plugin.js'), 'utf8')).toBe(newerBytes)
      expect(unpinned.pluginName).toBe('manifest-name')
    },
    30_000
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
      const rows = mergePluginPackages(
        [{ id: 'widget', name: 'Widget', kind: 'disk', packageName: marker.package } as PluginRecord],
        [{ name: 'manifest-name', has_desktop_half: true, description: '' } as AgentPluginRow]
      )
      expect(rows).toHaveLength(1)
      expect(rows[0]).toMatchObject({ key: 'manifest-name', desktopMissing: false, agentMissingInProfile: false })
      expect(marker).toMatchObject({ package: 'manifest-name', catalogName: 'catalog-alias' })
      const localDesktop = path.join(home, 'plugins', 'manifest-name', 'desktop')
      fs.mkdirSync(localDesktop, { recursive: true })
      fs.writeFileSync(path.join(localDesktop, 'plugin.js'), 'local agent bytes')
      await reconcileUnifiedDesktopHalves(home, appRoot)
      expect(fs.readdirSync(appRoot)).toEqual(['manifest-name'])
      expect(fs.readFileSync(path.join(appRoot, 'manifest-name', 'plugin.js'), 'utf8')).toBe('local agent bytes')
    }
  )

  it.each(['name: ../escape', 'name: NUL', 'name: 123', 'name: [broken', '- not-a-mapping'])(
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

  it('uses the source folder fallback when a manifest has no name', async () => {
    const repo = pluginRepo('temporary')
    fs.writeFileSync(path.join(repo, 'plugin.yaml'), 'description: unnamed package\n')
    execFileSync('git', ['add', '.'], { cwd: repo })
    execFileSync(
      'git',
      ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com', 'commit', '-qm', 'unnamed'],
      { cwd: repo }
    )
    const appRoot = mkdtemp('hermes-plugin-unnamed-')
    roots.push(appRoot)
    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, false, {
      ref: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim(),
      catalogName: 'catalog-alias'
    })
    expect(result).toMatchObject({ ok: true, pluginName: path.basename(repo) })
  })

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
