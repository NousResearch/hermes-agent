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

  it('detects a manifest-only package — dashboard backend, no __init__.py', async () => {
    const root = mkdtemp('hermes-plugin-manifest-only-')
    roots.push(root)
    fs.writeFileSync(path.join(root, 'plugin.yaml'), 'name: bulletin\n')
    fs.mkdirSync(path.join(root, 'dashboard'), { recursive: true })
    fs.writeFileSync(path.join(root, 'dashboard', 'manifest.json'), '{"name":"bulletin","api":"plugin_api.py"}\n')
    fs.mkdirSync(path.join(root, 'desktop'), { recursive: true })
    fs.writeFileSync(path.join(root, 'desktop', 'plugin.js'), 'export default { id: "bulletin" }')

    // Catalog plugins that ship a dashboard API but no register(ctx) are still packages: the
    // install plan must run the agent half, or the desktop half lands alone and ctx.rest 404s.
    await expect(detectPluginComponents(root)).resolves.toMatchObject({
      agent: true,
      desktop: true,
      agentName: 'bulletin'
    })
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

interface DetectionCase {
  name: string
  root: string
  files: Record<string, string>
  expect: { agent: boolean; desktop: boolean; agent_name: null | string }
}

// Shared with the gateway's detector (tests/hermes_cli/test_plugins_cmd_inspect.py).
const DETECTION_CASES = (
  JSON.parse(
    fs.readFileSync(
      path.resolve(here, '..', '..', '..', 'tests', 'fixtures', 'plugin_component_detection.json'),
      'utf8'
    )
  ) as { cases: DetectionCase[] }
).cases

describe('detectPluginComponents on the shared detection cases', () => {
  const roots: string[] = []

  afterEach(() => {
    for (const root of roots.splice(0)) {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it.each(DETECTION_CASES)('$name', async ({ root, files, expect: expected }) => {
    const parent = mkdtemp('hermes-detect-')
    roots.push(parent)
    const pluginRoot = path.join(parent, root)
    fs.mkdirSync(pluginRoot)

    for (const [rel, text] of Object.entries(files)) {
      fs.mkdirSync(path.dirname(path.join(pluginRoot, rel)), { recursive: true })
      fs.writeFileSync(path.join(pluginRoot, rel), text)
    }

    const detected = await detectPluginComponents(pluginRoot)

    expect({ agent: detected.agent, desktop: detected.desktop, agent_name: detected.agentName }).toEqual(expected)
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
  function pluginRepo(agentName: null | string, desktopFiles: Record<string, string> = {}): string {
    const repo = mkdtemp('hermes-plugin-install-')
    roots.push(repo)
    const git = (...args: string[]) => execFileSync('git', args, { cwd: repo, stdio: 'pipe' })

    fs.mkdirSync(path.join(repo, 'desktop'), { recursive: true })
    fs.writeFileSync(path.join(repo, 'desktop', 'plugin.js'), 'export function register() {}\n')

    for (const [name, text] of Object.entries(desktopFiles)) {
      fs.writeFileSync(path.join(repo, 'desktop', name), text)
    }

    if (agentName) {
      fs.writeFileSync(path.join(repo, 'plugin.yaml'), `name: ${agentName}\n`)
      fs.writeFileSync(path.join(repo, '__init__.py'), 'def register(ctx): pass\n')
    }

    git('init', '-q')
    git('add', '.')
    git('-c', 'user.email=fixture@example.com', '-c', 'user.name=Fixture', 'commit', '-qm', 'init')

    return repo
  }

  it('installs the commit the dialog reviewed after the branch moved, and refuses one the source lacks', async () => {
    const repo = pluginRepo(null, { 'panel.js': 'reviewed\n' })
    const git = (...args: string[]) => execFileSync('git', args, { cwd: repo, stdio: 'pipe' }).toString().trim()
    const appRoot = mkdtemp('hermes-plugin-root-')
    roots.push(appRoot)
    const probe = await probePluginRepo('git', pathToFileURL(repo).href)
    expect(probe.sha).toBe(git('rev-parse', 'HEAD'))
    fs.writeFileSync(path.join(repo, 'desktop', 'panel.js'), 'moved\n')
    git('-c', 'user.email=fixture@example.com', '-c', 'user.name=Fixture', 'commit', '-qam', 'moved')

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, false, probe.sha)

    expect(result).toMatchObject({ ok: true })
    expect(fs.readFileSync(path.join(appRoot, result.pluginName!, 'panel.js'), 'utf8')).toBe('reviewed\n')

    const refused = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, true, 'f'.repeat(40))
    expect(refused).toMatchObject({ ok: false, sourceMoved: true })
    expect(refused.error).toContain('the source may have changed since you reviewed it')
    expect(fs.readFileSync(path.join(appRoot, result.pluginName!, 'panel.js'), 'utf8')).toBe('reviewed\n')
  })

  it('installs a reviewed catalog pin that names an annotated tag at the commit it points at', async () => {
    const repo = pluginRepo(null, { 'panel.js': 'reviewed\n' })
    const git = (...args: string[]) => execFileSync('git', args, { cwd: repo, stdio: 'pipe' }).toString().trim()
    const ident = ['-c', 'user.email=fixture@example.com', '-c', 'user.name=Fixture']
    git(...ident, 'tag', '-a', 'v1.0', '-m', 'v1.0')
    fs.writeFileSync(path.join(repo, 'desktop', 'panel.js'), 'moved\n')
    git(...ident, 'commit', '-qam', 'moved')
    const tagObject = git('rev-parse', 'v1.0')
    expect(tagObject).not.toBe(git('rev-parse', 'v1.0^{commit}'))
    const appRoot = mkdtemp('hermes-plugin-root-')
    roots.push(appRoot)

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, false, tagObject)

    expect(result).toMatchObject({ ok: true })
    expect(fs.readFileSync(path.join(appRoot, result.pluginName!, 'panel.js'), 'utf8')).toBe('reviewed\n')
  })

  it.each([
    [
      'a symlinked file inside it',
      (repo: string) => fs.symlinkSync('/etc/hostname', path.join(repo, 'desktop', 'leak.txt'))
    ],
    [
      'a symlinked desktop folder',
      (repo: string, outside: string) => {
        fs.renameSync(path.join(repo, 'desktop'), path.join(outside, 'desktop'))
        fs.symlinkSync(path.join(outside, 'desktop'), path.join(repo, 'desktop'))
      }
    ]
  ])('refuses a desktop tree with %s and publishes nothing', async (_label, link) => {
    // A symlink in the copied tree would publish whatever it points at on this machine as plugin files.
    const repo = pluginRepo(null)
    const outside = mkdtemp('hermes-plugin-outside-')
    const appRoot = mkdtemp('hermes-plugin-root-')
    roots.push(outside, appRoot)
    link(repo, outside)
    execFileSync('git', ['add', '-A'], { cwd: repo, stdio: 'pipe' })
    execFileSync('git', ['-c', 'user.email=f@example.com', '-c', 'user.name=F', 'commit', '-qm', 'link'], {
      cwd: repo,
      stdio: 'pipe'
    })

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)

    expect(result.ok).toBe(false)
    expect(result.error).toContain('symbolic link')
    expect(fs.readdirSync(appRoot)).toEqual([])
  })

  it('keeps the installed plugin when a forced reinstall refuses the new tree', async () => {
    const repo = pluginRepo(null, { 'panel.js': 'working\n' })
    const git = (...args: string[]) => execFileSync('git', args, { cwd: repo, stdio: 'pipe' })
    const appRoot = mkdtemp('hermes-plugin-root-')
    roots.push(appRoot)
    const installed = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)
    expect(installed).toMatchObject({ ok: true })
    fs.symlinkSync('/etc/hostname', path.join(repo, 'desktop', 'leak.txt'))
    git('add', '-A')
    git('-c', 'user.email=f@example.com', '-c', 'user.name=F', 'commit', '-qm', 'link')

    const refused = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, true)

    expect(refused.ok).toBe(false)
    expect(refused.error).toContain('symbolic link')
    expect(fs.readdirSync(appRoot)).toEqual([installed.pluginName])
    expect(fs.readFileSync(path.join(appRoot, installed.pluginName!, 'panel.js'), 'utf8')).toBe('working\n')
    expect(fs.existsSync(path.join(appRoot, installed.pluginName!, 'plugin.js'))).toBe(true)

    // A plain replacement tree still replaces it.
    fs.rmSync(path.join(repo, 'desktop', 'leak.txt'))
    fs.writeFileSync(path.join(repo, 'desktop', 'panel.js'), 'replaced\n')
    git('add', '-A')
    git('-c', 'user.email=f@example.com', '-c', 'user.name=F', 'commit', '-qm', 'plain')
    expect(await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, true)).toMatchObject({
      ok: true
    })
    expect(fs.readdirSync(appRoot)).toEqual([installed.pluginName])
    expect(fs.readFileSync(path.join(appRoot, installed.pluginName!, 'panel.js'), 'utf8')).toBe('replaced\n')
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

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)

    expect(result.ok).toBe(true)
    expect(fs.existsSync(path.join(appRoot, String(result.pluginName), 'plugin.js'))).toBe(true)
    expect(fs.existsSync(path.join(appRoot, String(result.pluginName), PACKAGE_MARKER))).toBe(false)
  })

  it('drops a package marker a desktop-only repo ships, so it cannot pose as another package', async () => {
    // The marker is what pairs a desktop half with an agent package (and lets
    // it render that package's tool media); only this app may write it.
    const forged = JSON.stringify({ package: 'meshy-ish', source: '/elsewhere' })
    const repo = pluginRepo(null, { [PACKAGE_MARKER]: forged })
    const appRoot = mkdtemp('hermes-plugin-root-')
    roots.push(appRoot)

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)

    expect(result.ok).toBe(true)
    expect(fs.existsSync(path.join(appRoot, String(result.pluginName), 'plugin.js'))).toBe(true)
    expect(fs.existsSync(path.join(appRoot, String(result.pluginName), PACKAGE_MARKER))).toBe(false)
  })
})
