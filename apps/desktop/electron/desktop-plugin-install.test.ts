import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { afterEach, describe, expect, it } from 'vitest'

import {
  desktopPluginFolderName,
  detectPluginComponents,
  findDesktopEntry,
  manifestNameFromYaml,
  repoNameFromUrl,
  resolvePluginGitUrl,
  resolveSubdirWithin
} from './desktop-plugin-install'

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

describe('repoNameFromUrl', () => {
  it('strips .git suffix', () => {
    expect(repoNameFromUrl('https://github.com/o/my-plugin.git')).toBe('my-plugin')
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

  it('names a manifest without `name:` after the caller fallback, not the clone folder', async () => {
    // A probe clone lives in a mkdtemp folder; the backend would install this
    // package under the repo name, so that is the name a caller can look up.
    const root = mkdtemp('hermes-plugin-noname-')
    roots.push(root)
    fs.writeFileSync(path.join(root, 'plugin.yaml'), 'version: 1\n')
    fs.writeFileSync(path.join(root, '__init__.py'), 'def register(ctx): pass\n')

    await expect(detectPluginComponents(root, 'from-repo-url')).resolves.toMatchObject({
      agent: true,
      agentName: 'from-repo-url'
    })
    await expect(detectPluginComponents(root)).resolves.toMatchObject({ agentName: path.basename(root) })
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
      desktopName: 'desktop',
      desktopSourceSubdir: 'desktop'
    })
  })

  it('drops an unquoted trailing YAML comment from the agent name (matches yaml.safe_load)', async () => {
    const root = mkdtemp('hermes-plugin-comment-')
    roots.push(root)
    fs.writeFileSync(path.join(root, 'plugin.yaml'), 'name: dual # the folder the backend creates\n')
    fs.writeFileSync(path.join(root, '__init__.py'), 'def register(ctx): pass\n')

    await expect(detectPluginComponents(root)).resolves.toMatchObject({ agent: true, agentName: 'dual' })
  })

  it('keeps a quoted agent name verbatim, hash included (matches yaml.safe_load)', async () => {
    const root = mkdtemp('hermes-plugin-quoted-')
    roots.push(root)
    fs.writeFileSync(path.join(root, 'plugin.yaml'), 'name: "dual # two"\n')
    fs.writeFileSync(path.join(root, '__init__.py'), 'def register(ctx): pass\n')

    await expect(detectPluginComponents(root)).resolves.toMatchObject({ agent: true, agentName: 'dual # two' })
  })

  it('reports a root plugin.js desktop half as sourceSubdir "."', async () => {
    // The unified plugins/<name>/desktop/ door never serves this shape, so
    // the install modal must keep copying it to desktop-plugins/ (#100412).
    const root = mkdtemp('hermes-plugin-dual-root-')
    roots.push(root)
    fs.writeFileSync(path.join(root, 'plugin.yaml'), 'name: dual-root\n')
    fs.writeFileSync(path.join(root, '__init__.py'), 'def register(ctx): pass\n')
    fs.writeFileSync(path.join(root, 'plugin.js'), 'export default { id: "dual-root-ui" }')

    await expect(detectPluginComponents(root)).resolves.toMatchObject({
      agent: true,
      desktop: true,
      desktopSourceSubdir: '.'
    })
  })
})

describe('manifestNameFromYaml', () => {
  it.each([
    ['name: dual\n', 'dual'],
    ['name: dual # the folder the backend creates\n', 'dual'],
    ['name: "dual" # comment after the quotes\n', 'dual'],
    ["name: 'dual # two'\n", 'dual # two'],
    ['name: foo#bar\n', 'foo#bar'],
    ['version: 1\r\nname: dual # c\r\nfoo: 1\r\n', 'dual'],
    ['name: "dual" # c\r\n', 'dual'],
    ['version: 1\nname:   spaced   \n', 'spaced'],
    ['name: ""\n', null],
    ['version: 1\n', null]
  ])('%j -> %j', (text, expected) => {
    expect(manifestNameFromYaml(text)).toBe(expected)
  })
})
