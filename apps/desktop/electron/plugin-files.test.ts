/**
 * The folder-scoped plugin file door (`ctx.files`). Every refusal here is the
 * main process's own: the renderer (and the plugin in it) only names a folder
 * handle and a relative path, and this module decides what that may reach.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPluginFiles, type PluginFiles } from './plugin-files'

let scratch: string
let userData: string
let hermesHome: string
let vault: string
let trashed: string[]
let files: PluginFiles

const make = () =>
  createPluginFiles({
    userDataDir: () => userData,
    protectedRoots: () => [hermesHome],
    trashItem: async target => {
      trashed.push(target)
      await fs.promises.rm(target, { recursive: true })
    }
  })

beforeEach(() => {
  scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-plugin-files-')))
  userData = path.join(scratch, 'userData')
  hermesHome = path.join(scratch, 'home', '.hermes')
  vault = path.join(scratch, 'home', 'Vault')
  fs.mkdirSync(userData, { recursive: true })
  fs.mkdirSync(hermesHome, { recursive: true })
  fs.writeFileSync(path.join(hermesHome, '.env'), 'OPENROUTER_API_KEY=sk-secret\n')
  fs.writeFileSync(path.join(hermesHome, 'config.yaml'), 'model: x\n')
  fs.mkdirSync(path.join(vault, 'notes'), { recursive: true })
  fs.writeFileSync(path.join(vault, 'notes', 'a.md'), '# A\n')
  trashed = []
  files = make()
})

afterEach(() => {
  fs.rmSync(scratch, { recursive: true, force: true })
})

async function grantVault(pluginId = 'plexus', folder = vault): Promise<string> {
  const granted = await files.addGrant(pluginId, folder)

  expect(granted.ok).toBe(true)

  return (granted as { folder: { id: string } }).folder.id
}

describe('grants', () => {
  it('persists a user-picked folder per plugin and survives a restart', async () => {
    const id = await grantVault()

    expect(await make().grants('plexus')).toEqual([expect.objectContaining({ id, name: 'Vault', path: vault })])
    expect(await make().grants('other')).toEqual([])
  })

  it('re-picking the same folder reuses the grant', async () => {
    const first = await grantVault()

    expect(await grantVault()).toBe(first)
    expect(await files.grants('plexus')).toHaveLength(1)
  })

  it.each([
    ['the Hermes home', () => hermesHome],
    ['a folder inside the Hermes home', () => path.join(hermesHome, 'skills')],
    ['a folder that contains the Hermes home', () => path.join(scratch, 'home')],
    ['the app data folder', () => userData],
    ['another plugin data folder', () => path.join(userData, 'plugin-files', 'data', 'other')]
  ])('refuses to grant %s', async (_label, folder) => {
    fs.mkdirSync(folder(), { recursive: true })

    expect(await files.addGrant('plexus', folder())).toMatchObject({ ok: false, error: 'protected' })
    expect(await files.grants('plexus')).toEqual([])
  })

  it('refuses to grant a key folder such as ~/.ssh', async () => {
    const ssh = path.join(scratch, 'home', '.ssh')
    fs.mkdirSync(ssh, { recursive: true })

    expect(await files.addGrant('plexus', ssh)).toMatchObject({ ok: false, error: 'protected' })
  })

  it('refuses to grant a symlink that resolves into the Hermes home', async () => {
    const link = path.join(scratch, 'innocent')
    fs.symlinkSync(hermesHome, link)

    expect(await files.addGrant('plexus', link)).toMatchObject({ ok: false, error: 'protected' })
  })

  it('a revoked grant refuses every later operation at once', async () => {
    const id = await grantVault()

    expect(await files.run('plexus', { op: 'readText', folder: id, path: 'notes/a.md' })).toMatchObject({ ok: true })
    expect(await files.revoke('plexus', id)).toEqual({ ok: true, removed: 1 })

    expect(await files.run('plexus', { op: 'readText', folder: id, path: 'notes/a.md' })).toMatchObject({
      ok: false,
      error: 'not-granted'
    })
    expect(await make().grants('plexus')).toEqual([])
  })

  it("one plugin cannot use another plugin's grant id", async () => {
    const id = await grantVault('plexus')

    expect(await files.run('intruder', { op: 'list', folder: id, path: '' })).toMatchObject({
      ok: false,
      error: 'not-granted'
    })
  })

  it('refuses a grant whose folder was swapped for a link into the Hermes home', async () => {
    const id = await grantVault()
    fs.rmSync(vault, { recursive: true })
    fs.symlinkSync(hermesHome, vault)

    expect(await files.run('plexus', { op: 'readText', folder: id, path: '.env' })).toMatchObject({ ok: false })
    expect(await files.run('plexus', { op: 'list', folder: id, path: '' })).toMatchObject({ ok: false })
  })
})

describe('path containment', () => {
  it.each([
    ['parent traversal', '../.hermes/.env'],
    ['nested traversal', 'notes/../../.hermes/.env'],
    ['backslash traversal', 'notes\\..\\..\\.hermes\\.env'],
    ['an absolute POSIX path', '/etc/passwd'],
    ['an absolute path into the Hermes home', ''],
    ['a Windows drive path', 'C:\\Users\\me\\.hermes\\.env'],
    ['a UNC path', '\\\\server\\share\\x'],
    ['a NUL byte', 'notes/a.md\0.png']
  ])('refuses %s', async (label, rel) => {
    const id = await grantVault()
    const target = label === 'an absolute path into the Hermes home' ? path.join(hermesHome, '.env') : rel

    expect(await files.run('plexus', { op: 'readText', folder: id, path: target })).toMatchObject({
      ok: false,
      error: 'invalid-path'
    })
    expect(
      await files.run('plexus', { op: 'writeText', folder: id, path: target, text: 'pwned' })
    ).toMatchObject({ ok: false, error: 'invalid-path' })
    expect(fs.readFileSync(path.join(hermesHome, 'config.yaml'), 'utf8')).toBe('model: x\n')
  })

  it('refuses to read through a symlinked file that points outside the folder', async () => {
    const id = await grantVault()
    fs.symlinkSync(path.join(hermesHome, '.env'), path.join(vault, 'notes', 'leak.md'))

    expect(await files.run('plexus', { op: 'readText', folder: id, path: 'notes/leak.md' })).toMatchObject({
      ok: false,
      error: 'outside-folder'
    })
    expect(await files.run('plexus', { op: 'readDataUrl', folder: id, path: 'notes/leak.md' })).toMatchObject({
      ok: false,
      error: 'outside-folder'
    })
  })

  it('refuses to write through a symlinked file or folder that points outside', async () => {
    const id = await grantVault()
    fs.symlinkSync(path.join(hermesHome, 'config.yaml'), path.join(vault, 'cfg.md'))
    fs.symlinkSync(hermesHome, path.join(vault, 'home'))

    for (const rel of ['cfg.md', 'home/config.yaml', 'home/new.md']) {
      expect(await files.run('plexus', { op: 'writeText', folder: id, path: rel, text: 'pwned' })).toMatchObject({
        ok: false,
        error: 'outside-folder'
      })
    }

    expect(fs.readFileSync(path.join(hermesHome, 'config.yaml'), 'utf8')).toBe('model: x\n')
    expect(fs.existsSync(path.join(hermesHome, 'new.md'))).toBe(false)
  })

  it('refuses a dangling symlink instead of creating its target', async () => {
    const id = await grantVault()
    const outside = path.join(scratch, 'created-outside.md')
    fs.symlinkSync(outside, path.join(vault, 'dangling.md'))

    expect(
      await files.run('plexus', { op: 'writeText', folder: id, path: 'dangling.md', text: 'x' })
    ).toMatchObject({ ok: false })
    expect(fs.existsSync(outside)).toBe(false)
  })

  it('refuses listing, renaming and trashing through a symlinked folder into the Hermes home', async () => {
    const id = await grantVault()
    fs.symlinkSync(hermesHome, path.join(vault, 'home'))

    expect(await files.run('plexus', { op: 'list', folder: id, path: 'home' })).toMatchObject({
      ok: false,
      error: 'outside-folder'
    })
    expect(
      await files.run('plexus', { op: 'rename', folder: id, path: 'home/config.yaml', name: 'x.yaml' })
    ).toMatchObject({ ok: false, error: 'outside-folder' })
    expect(await files.run('plexus', { op: 'trash', folder: id, path: 'home/.env' })).toMatchObject({
      ok: false,
      error: 'outside-folder'
    })
    expect(trashed).toEqual([])
    expect(fs.existsSync(path.join(hermesHome, '.env'))).toBe(true)
  })

  it('a listing hides links that leave the folder and keeps links that stay inside', async () => {
    const id = await grantVault()
    fs.symlinkSync(hermesHome, path.join(vault, 'escape'))
    fs.symlinkSync(path.join(vault, 'notes'), path.join(vault, 'alias'))

    const listed = await files.run('plexus', { op: 'list', folder: id, path: '' })

    expect(listed).toMatchObject({ ok: true })
    expect((listed as unknown as { entries: { name: string }[] }).entries.map(e => e.name).sort()).toEqual(['alias', 'notes'])
  })

  it('rename keeps the item in its folder and refuses a name with a separator or traversal', async () => {
    const id = await grantVault()

    for (const name of ['../../escaped.md', 'sub/x.md', '..', '.', '']) {
      expect(await files.run('plexus', { op: 'rename', folder: id, path: 'notes/a.md', name })).toMatchObject({
        ok: false,
        error: 'invalid-path'
      })
    }

    expect(await files.run('plexus', { op: 'rename', folder: id, path: 'notes/a.md', name: 'b.md' })).toEqual({
      ok: true,
      path: 'notes/b.md'
    })
    expect(fs.existsSync(path.join(vault, 'notes', 'b.md'))).toBe(true)
  })

  it('never trashes or renames the granted folder itself', async () => {
    const id = await grantVault()

    expect(await files.run('plexus', { op: 'trash', folder: id, path: '' })).toMatchObject({
      ok: false,
      error: 'invalid-path'
    })
    expect(await files.run('plexus', { op: 'rename', folder: id, path: '.', name: 'x' })).toMatchObject({
      ok: false,
      error: 'invalid-path'
    })
    expect(fs.existsSync(vault)).toBe(true)
  })
})

describe('plugin data folder', () => {
  it('reads and writes inside its own data folder without any grant', async () => {
    expect(
      await files.run('plexus', { op: 'writeText', folder: 'data', path: 'state.json', text: '{"a":1}' })
    ).toMatchObject({ ok: true, path: 'state.json' })
    expect(await files.run('plexus', { op: 'readText', folder: 'data', path: 'state.json' })).toMatchObject({
      ok: true,
      text: '{"a":1}'
    })
    expect(fs.readFileSync(path.join(userData, 'plugin-files', 'data', 'plexus', 'state.json'), 'utf8')).toBe(
      '{"a":1}'
    )
  })

  it("cannot reach another plugin's data folder or the grants record", async () => {
    await files.run('other', { op: 'writeText', folder: 'data', path: 'secret.json', text: 'theirs' })
    await files.run('plexus', { op: 'writeText', folder: 'data', path: 'mine.json', text: 'mine' })

    for (const rel of ['../other/secret.json', '../../grants.json']) {
      expect(await files.run('plexus', { op: 'readText', folder: 'data', path: rel })).toMatchObject({
        ok: false,
        error: 'invalid-path'
      })
    }

    fs.symlinkSync(
      path.join(userData, 'plugin-files', 'data', 'other'),
      path.join(userData, 'plugin-files', 'data', 'plexus', 'peek')
    )

    expect(await files.run('plexus', { op: 'readText', folder: 'data', path: 'peek/secret.json' })).toMatchObject({
      ok: false,
      error: 'outside-folder'
    })
  })

  it('keeps ids that differ only in case apart on any filesystem', async () => {
    await files.run('plexus', { op: 'writeText', folder: 'data', path: 'who.txt', text: 'lower' })
    await files.run('Plexus', { op: 'writeText', folder: 'data', path: 'who.txt', text: 'upper' })

    expect(await files.run('plexus', { op: 'readText', folder: 'data', path: 'who.txt' })).toMatchObject({
      text: 'lower'
    })
  })

  it.each(['../x', 'a/b', '..', '', 'x'.repeat(200)])('refuses a plugin id that is not a plain slug: %j', async id => {
    expect(await files.run(id, { op: 'list', folder: 'data', path: '' })).toMatchObject({
      ok: false,
      error: 'invalid-plugin'
    })
  })
})

describe('caps and shapes', () => {
  it('refuses oversized writes and reads instead of truncating', async () => {
    const id = await grantVault()
    const big = 'x'.repeat(4 * 1024 * 1024 + 1)

    expect(await files.run('plexus', { op: 'writeText', folder: id, path: 'big.md', text: big })).toMatchObject({
      ok: false,
      error: 'too-large'
    })
    expect(fs.existsSync(path.join(vault, 'big.md'))).toBe(false)

    fs.writeFileSync(path.join(vault, 'big.md'), big)
    expect(await files.run('plexus', { op: 'readText', folder: id, path: 'big.md' })).toMatchObject({
      ok: false,
      error: 'too-large'
    })
  })

  it('reads binary files only as data URLs', async () => {
    const id = await grantVault()
    fs.writeFileSync(path.join(vault, 'pic.png'), Buffer.from([0x89, 0x50, 0x4e, 0x47, 0, 1, 2]))

    expect(await files.run('plexus', { op: 'readText', folder: id, path: 'pic.png' })).toMatchObject({
      ok: false,
      error: 'binary'
    })
    expect(await files.run('plexus', { op: 'readDataUrl', folder: id, path: 'pic.png' })).toEqual({
      ok: true,
      url: 'data:image/png;base64,iVBORwABAg=='
    })
  })

  it('refuses sensitive files inside a granted folder', async () => {
    const id = await grantVault()
    fs.writeFileSync(path.join(vault, '.env'), 'TOKEN=x\n')

    expect(await files.run('plexus', { op: 'readText', folder: id, path: '.env' })).toMatchObject({
      ok: false,
      error: 'sensitive-file'
    })
    expect(
      await files.run('plexus', { op: 'writeText', folder: id, path: 'notes/id_ed25519', text: 'k' })
    ).toMatchObject({ ok: false, error: 'sensitive-file' })
    expect(await files.run('plexus', { op: 'rename', folder: id, path: 'notes/a.md', name: '.env' })).toMatchObject({
      ok: false,
      error: 'sensitive-file'
    })
  })

  it('an exclusive write never replaces an existing file', async () => {
    const id = await grantVault()

    expect(
      await files.run('plexus', { op: 'writeText', folder: id, path: 'notes/a.md', text: 'new', exclusive: true })
    ).toMatchObject({ ok: false, error: 'exists' })
    expect(fs.readFileSync(path.join(vault, 'notes', 'a.md'), 'utf8')).toBe('# A\n')
  })

  it('answers an unknown op as a result, never a throw', async () => {
    const id = await grantVault()

    expect(await files.run('plexus', { op: 'unlink', folder: id, path: 'notes/a.md' } as never)).toMatchObject({
      ok: false,
      error: 'invalid-op'
    })
  })

  it('stat reports a missing entry without failing', async () => {
    const id = await grantVault()

    expect(await files.run('plexus', { op: 'stat', folder: id, path: 'notes/missing.md' })).toEqual({
      ok: true,
      exists: false
    })
    expect(await files.run('plexus', { op: 'stat', folder: id, path: 'notes/a.md' })).toMatchObject({
      ok: true,
      exists: true,
      isDirectory: false,
      size: 4
    })
  })

  it('trash moves only the named entry', async () => {
    const id = await grantVault()

    expect(await files.run('plexus', { op: 'trash', folder: id, path: 'notes/a.md' })).toEqual({ ok: true })
    expect(trashed).toEqual([path.join(vault, 'notes', 'a.md')])
  })
})

describe('protected roots', () => {
  it('an op inside a granted folder never reaches a protected root nested under it', async () => {
    // A grant made before HERMES_HOME moved inside it must not become a door into the new home.
    const id = await grantVault()
    const nestedHome = path.join(vault, 'hermes')
    fs.mkdirSync(nestedHome)
    fs.writeFileSync(path.join(nestedHome, 'config.yaml'), 'model: y\n')
    const roots = vi.fn(() => [hermesHome, nestedHome])

    files = createPluginFiles({ userDataDir: () => userData, protectedRoots: roots, trashItem: async () => undefined })

    expect(await files.run('plexus', { op: 'readText', folder: id, path: 'hermes/config.yaml' })).toMatchObject({
      ok: false,
      error: 'protected'
    })
  })
})
