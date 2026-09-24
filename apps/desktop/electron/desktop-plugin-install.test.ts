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
import { PACKAGE_MARKER } from './desktop-plugins-root'

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

// These cases spawn several real Git processes. Windows process startup under
// the full suite can exceed Vitest's 5s unit-test default without a stalled clone.
const GIT_PROCESS_TEST_TIMEOUT = 30_000

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
  }, GIT_PROCESS_TEST_TIMEOUT)
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
  }, GIT_PROCESS_TEST_TIMEOUT)

  it('leaves a desktop-only repo unmarked so it stays a standalone plugin', async () => {
    const repo = pluginRepo(null)
    const appRoot = mkdtemp('hermes-plugin-root-')
    roots.push(appRoot)

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)

    expect(result.ok).toBe(true)
    expect(fs.existsSync(path.join(appRoot, String(result.pluginName), 'plugin.js'))).toBe(true)
    expect(fs.existsSync(path.join(appRoot, String(result.pluginName), PACKAGE_MARKER))).toBe(false)
  }, GIT_PROCESS_TEST_TIMEOUT)
})
