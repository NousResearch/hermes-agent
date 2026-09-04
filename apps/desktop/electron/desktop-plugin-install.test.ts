import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { afterEach, describe, expect, it } from 'vitest'

import {
  desktopPluginFolderName,
  detectPluginComponents,
  findDesktopEntry,
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
