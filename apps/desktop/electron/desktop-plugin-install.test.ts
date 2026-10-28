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
    fs.readFileSync(path.resolve(here, '..', '..', '..', 'tests', 'fixtures', 'plugin_component_detection.json'), 'utf8')
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
