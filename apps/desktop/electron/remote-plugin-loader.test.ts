import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { loadRemotePlugins, validateRemotePluginModule } from './remote-plugin-loader'

const validRemotePlugin = {
  id: 'fake-remote-plugin',
  scheme: 'fake',
  hint: 'paste a fake ticket',
  connect: async () => ({ baseUrl: 'http://127.0.0.1:59999', teardown: async () => {} })
}

describe('validateRemotePluginModule', () => {
  it('accepts a well-shaped default export and canonicalizes the scheme', () => {
    const { plugin, reason } = validateRemotePluginModule({ default: { ...validRemotePlugin, scheme: 'FAKE' } })

    expect(reason).toBeNull()
    expect(plugin?.scheme).toBe('fake')
    expect(plugin?.id).toBe('fake-remote-plugin')
  })

  it('refuses http/https registration — built-ins stay core', () => {
    expect(validateRemotePluginModule({ default: { ...validRemotePlugin, scheme: 'https' } }).reason).toMatch(/built-in/)
    expect(validateRemotePluginModule({ default: { ...validRemotePlugin, scheme: 'http' } }).reason).toMatch(/built-in/)
  })

  it('refuses malformed shapes with a named reason', () => {
    expect(validateRemotePluginModule(undefined).reason).toBe('no default export')
    expect(validateRemotePluginModule({}).reason).toBe('no default export')
    expect(validateRemotePluginModule({ default: 42 }).reason).toBe('no default export')
    expect(validateRemotePluginModule({ default: { scheme: 'fake', connect: () => {} } }).reason).toBe('missing id')
    expect(validateRemotePluginModule({ default: { id: 'x', connect: () => {} } }).reason).toMatch(/invalid scheme/)
    expect(validateRemotePluginModule({ default: { id: 'x', scheme: 'bad scheme!' } }).reason).toMatch(/invalid scheme/)
    expect(validateRemotePluginModule({ default: { id: 'x', scheme: 'fake' } }).reason).toBe('connect is not a function')
    expect(validateRemotePluginModule({ default: { ...validRemotePlugin, claims: 'nope' } }).reason).toMatch(/claims/)
    expect(validateRemotePluginModule({ default: { ...validRemotePlugin, normalize: 7 } }).reason).toMatch(/normalize/)
    expect(validateRemotePluginModule({ default: { ...validRemotePlugin, pastePattern: '([unclosed' } }).reason).toMatch(/regular expression/)
  })
})

const tmpRoots: string[] = []

function tmpRoot(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-remote-plugins-'))

  tmpRoots.push(dir)

  return dir
}

afterEach(() => {
  while (tmpRoots.length) {
    fs.rmSync(tmpRoots.pop()!, { recursive: true, force: true })
  }
})

function writeRemotePluginFolder(root: string, name: string, source: string, basename = 'remote.mjs'): void {
  fs.mkdirSync(path.join(root, name), { recursive: true })
  fs.writeFileSync(path.join(root, name, basename), source)
}

describe('loadRemotePlugins', () => {
  it('a missing desktop-plugins root is the normal zero-plugin state, not an error', async () => {
    const result = await loadRemotePlugins({ root: '/nonexistent/definitely-not-here' })

    expect(result).toEqual({ plugins: [], issues: [] })
  })

  it('loads a real remote.mjs from disk (real dynamic import)', async () => {
    const root = tmpRoot()

    writeRemotePluginFolder(
      root,
      'fake-plugin',
      `export default {
         id: 'fake-remote-plugin',
         scheme: 'fake',
         hint: 'paste a ticket',
         normalize: url => url.replace(/\\s+/g, ''),
         connect: async () => ({ baseUrl: 'http://127.0.0.1:59999', teardown: async () => {} })
       }`
    )

    const { plugins, issues } = await loadRemotePlugins({ root })

    expect(issues).toEqual([])
    expect(plugins).toHaveLength(1)
    expect(plugins[0].id).toBe('fake-remote-plugin')
    expect(plugins[0].scheme).toBe('fake')
    expect(plugins[0].hint).toBe('paste a ticket')

    const target = await plugins[0].connect('fake://ticket', () => {})

    expect(target.baseUrl).toBe('http://127.0.0.1:59999')
  })

  it('remote.js is the CJS-adjacent alias; .mjs wins when both exist in one folder', async () => {
    const root = tmpRoot()

    writeRemotePluginFolder(root, 'both', `export default { id: 'from-mjs', scheme: 'mjs', connect: async () => ({}) }`)
    writeRemotePluginFolder(root, 'both', `export default { id: 'from-js', scheme: 'js', connect: async () => ({}) }`, 'remote.js')

    const { plugins } = await loadRemotePlugins({ root })

    expect(plugins.map(c => c.id)).toEqual(['from-mjs'])
  })

  it('one bad plugin is an issue, never a brick', async () => {
    const root = tmpRoot()

    writeRemotePluginFolder(root, 'explodes', `throw new Error('top-level kaboom')`)
    writeRemotePluginFolder(root, 'shapeless', `export default { id: 'nope' }`)
    writeRemotePluginFolder(root, 'good', `export default { id: 'fake-remote-plugin', scheme: 'fake', connect: async () => ({ baseUrl: 'http://127.0.0.1:1', teardown: async () => {} }) }`)

    const { plugins, issues } = await loadRemotePlugins({ root })

    expect(plugins.map(c => c.id)).toEqual(['fake-remote-plugin'])
    const folders = issues.map(i => i.folder)

    expect(folders).toContain('explodes')
    expect(folders).toContain('shapeless')
    expect(issues.find(i => i.folder === 'explodes')?.detail).toContain('kaboom')
  })

  it('duplicate scheme claims are first-wins (stable readdir order), loser reported', async () => {
    const root = tmpRoot()

    writeRemotePluginFolder(root, 'aaa', `export default { id: 'first', scheme: 'fake', connect: async () => ({ baseUrl: 'x', teardown: async () => {} }) }`)
    writeRemotePluginFolder(root, 'bbb', `export default { id: 'second', scheme: 'fake', connect: async () => ({ baseUrl: 'y', teardown: async () => {} }) }`)

    const { plugins, issues } = await loadRemotePlugins({ root })

    expect(plugins.map(c => c.id)).toEqual(['first'])
    expect(issues).toEqual([{ folder: 'bbb', detail: "duplicate scheme 'fake' (owned by 'aaa')" }])
  })

  it('isEnabled gates at the plugin-folder level (Settings → Plugins parity)', async () => {
    const root = tmpRoot()

    writeRemotePluginFolder(root, 'off', `export default { id: 'off-plugin', scheme: 'off', connect: async () => ({}) }`)
    writeRemotePluginFolder(root, 'on', `export default { id: 'on-plugin', scheme: 'on', connect: async () => ({}) }`)

    const { plugins, issues } = await loadRemotePlugins({ root, isEnabled: id => id !== 'off' })

    expect(plugins.map(c => c.id)).toEqual(['on-plugin'])
    // Disabled is not an issue — the user chose it.
    expect(issues).toEqual([])
  })

  it('folders without a remote plugin module are invisible to the loader', async () => {
    const root = tmpRoot()

    writeRemotePluginFolder(root, 'renderer-only', '// no remote plugin here', 'plugin.js')

    const { plugins, issues } = await loadRemotePlugins({ root })

    expect(plugins).toEqual([])
    expect(issues).toEqual([])
  })

  it('async isEnabled is awaited', async () => {
    const root = tmpRoot()

    writeRemotePluginFolder(root, 'p', `export default { id: 'p', scheme: 'p', connect: async () => ({}) }`)

    const isEnabled = vi.fn(async () => false)
    const { plugins } = await loadRemotePlugins({ root, isEnabled })

    expect(plugins).toEqual([])
    expect(isEnabled).toHaveBeenCalledWith('p')
  })
})
