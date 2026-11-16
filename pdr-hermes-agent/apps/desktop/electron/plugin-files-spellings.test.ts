/**
 * `ctx.files` refusals for a second spelling of a path: case on a
 * case-sensitive macOS volume, Unicode case folding, macOS named forks, WSL
 * shares, loopback admin shares, and kernel pseudo-filesystems mounted
 * somewhere a grant can reach.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPluginFiles, type PluginFiles } from './plugin-files'
import { loopbackShare, onPseudoFilesystem, overlapsByIdentity, wslProtectedPath } from './plugin-files-guards'

const posix = process.platform !== 'win32'

let scratch: string
let userData: string
let hermesHome: string
let vault: string

const make = (platform?: NodeJS.Platform): PluginFiles =>
  createPluginFiles({
    userDataDir: () => userData,
    protectedRoots: () => [hermesHome],
    trashItem: target => fs.promises.rm(target, { recursive: true }),
    platform
  })

beforeEach(() => {
  scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-plugin-files-spell-')))
  userData = path.join(scratch, 'userData')
  hermesHome = path.join(scratch, 'home', '.hermes')
  vault = path.join(scratch, 'home', 'Vault')
  fs.mkdirSync(userData, { recursive: true })
  fs.mkdirSync(hermesHome, { recursive: true })
  fs.mkdirSync(vault, { recursive: true })
  fs.writeFileSync(path.join(vault, 'a.md'), '# A\n')
})

afterEach(() => {
  vi.restoreAllMocks()
  fs.rmSync(scratch, { recursive: true, force: true })
})

async function grant(files: PluginFiles, folder = vault): Promise<string> {
  const granted = await files.addGrant('plexus', folder)

  expect(granted.ok).toBe(true)

  return (granted as { folder: { id: string } }).folder.id
}

// Linux volumes are case-sensitive, so they stand in for a case-sensitive
// macOS volume here: `Vault` and `vault` are two folders.
describe('case on a case-sensitive macOS volume', () => {
  it.runIf(posix)('a link from the granted Vault into a sibling vault is outside the folder', async () => {
    const files = make('darwin')
    const id = await grant(files)
    const sibling = path.join(scratch, 'home', 'vault')
    fs.mkdirSync(sibling)
    fs.writeFileSync(path.join(sibling, 'secret.txt'), 'not granted')
    fs.symlinkSync(path.join(sibling, 'secret.txt'), path.join(vault, 'link.txt'))

    const read = await files.run('plexus', { op: 'readText', folder: id, path: 'link.txt' })

    expect(read).toMatchObject({ ok: false, error: 'outside-folder' })
    expect(JSON.stringify(read)).not.toContain('not granted')
  })

  it.runIf(posix)('a granted Vault replaced by a differently-cased vault must be granted again', async () => {
    const files = make('darwin')
    const id = await grant(files)
    fs.renameSync(vault, path.join(scratch, 'home', 'vault'))
    fs.symlinkSync(path.join(scratch, 'home', 'vault'), vault)

    expect(await files.run('plexus', { op: 'list', folder: id, path: '' })).toMatchObject({
      ok: false,
      error: 'folder-changed'
    })
  })

  it.runIf(posix)('a differently-cased spelling of the Hermes home still cannot be granted', async () => {
    const lookalike = path.join(scratch, 'home', '.HERMES')
    fs.mkdirSync(lookalike)

    expect(await make('darwin').addGrant('plexus', lookalike)).toMatchObject({ ok: false, error: 'protected' })
  })
})

// macOS and Windows match names with Unicode case folding (`ſ` is `s`), so a
// name that folds onto a key file may open that key file there.
describe('sensitive names after Unicode folding', () => {
  // HFS+ also ignores zero-width and other default-ignorable code points.
  it.each(['id_r\u017fa', '.ENV', '\uff0eenv', '.env.LOCAL', '.e\u200dnv', 'id_\u200brsa', '.np\ufeffmrc'])(
    'refuses reading or writing %s',
    async name => {
    const files = make()
    const id = await grant(files)
    fs.writeFileSync(path.join(vault, name), 'SECRET=1\n')

    expect(await files.run('plexus', { op: 'readText', folder: id, path: name })).toMatchObject({
      ok: false,
      error: 'sensitive-file'
    })
    fs.mkdirSync(path.join(vault, 'fresh'))

    expect(await files.run('plexus', { op: 'writeText', folder: id, path: `fresh/${name}`, text: 'x' })).toMatchObject({
      ok: false,
      error: 'sensitive-file'
    })
    expect(fs.existsSync(path.join(vault, 'fresh', name))).toBe(false)
    }
  )

  it('refuses a folded key folder as a path and as a rename target', async () => {
    const files = make()
    const id = await grant(files)
    fs.mkdirSync(path.join(vault, '.\u017f\u017fh'))
    fs.writeFileSync(path.join(vault, '.\u017f\u017fh', 'config'), 'Host x\n')

    expect(await files.run('plexus', { op: 'readText', folder: id, path: '.\u017f\u017fh/config' })).toMatchObject({
      ok: false,
      error: 'sensitive-file'
    })
    expect(await files.run('plexus', { op: 'rename', folder: id, path: 'a.md', name: '.ENV' })).toMatchObject({
      ok: false,
      error: 'sensitive-file'
    })
  })
})

describe('macOS named forks', () => {
  it.each(['.env/..namedfork/data', 'a.md/..namedfork/rsrc', 'x/..NamedFork/data'])('refuses %s', async rel => {
    const files = make()
    const id = await grant(files)
    fs.writeFileSync(path.join(vault, '.env'), 'SECRET=1\n')

    expect(await files.run('plexus', { op: 'readText', folder: id, path: rel })).toMatchObject({
      ok: false,
      error: 'invalid-path'
    })
  })
})

describe('Windows names that open something else', () => {
  it.each(['CON', 'nul.txt', 'COM1', 'lpt\u00b9', 'CONIN$', 'conout$.log', 'CLOCK$', 'a.md:secret', '.env.', '.env ', 'notes\\..\\..'])(
    'refuses the segment %s',
    async rel => {
      const files = make('win32')
      const id = await grant(files)

      expect(await files.run('plexus', { op: 'readText', folder: id, path: rel })).toMatchObject({
        ok: false,
        error: 'invalid-path'
      })
    }
  )
})

describe('Windows second paths to this machine', () => {
  it.each([
    ['\\\\wsl.localhost\\Ubuntu', true],
    ['\\\\wsl$\\Ubuntu\\home', true],
    ['\\\\wsl.localhost\\Ubuntu\\home\\me', true],
    ['\\\\wsl.localhost\\Ubuntu\\root', true],
    ['\\\\wsl.localhost\\Ubuntu\\home\\me\\.hermes\\profiles', true],
    ['\\\\WSL.LOCALHOST\\Debian\\HOME\\me\\.Hermes', true],
    // The distro's own process environments and its view of the Windows drives.
    ['\\\\wsl.localhost\\Ubuntu\\proc', true],
    ['\\\\wsl$\\Ubuntu\\proc\\1\\environ', true],
    ['\\\\wsl.localhost\\Ubuntu\\sys', true],
    ['\\\\wsl.localhost\\Ubuntu\\dev', true],
    ['\\\\wsl.localhost\\Ubuntu\\run\\user', true],
    ['\\\\wsl.localhost\\Ubuntu\\mnt\\c\\Users\\me', true],
    ['\\\\wsl.localhost\\Ubuntu\\home\\me\\Vault', false],
    ['\\\\server\\share\\home', false],
    ['C:\\Users\\me\\Vault', false]
  ])('treats the WSL path %s as protected: %s', (candidate, exposed) => {
    expect(wslProtectedPath(candidate)).toBe(exposed)
  })

  it.each([
    ['\\\\localhost\\C$\\Users\\me', true],
    ['\\\\LOCALHOST\\share', true],
    ['\\\\127.0.0.1\\c$\\Users', true],
    ['\\\\127.1.2.3\\c$', true],
    ['\\\\[::1]\\c$', true],
    ['\\\\0--1.ipv6-literal.net\\c$', true],
    ['\\\\?\\UNC\\localhost\\c$\\Users', true],
    ['\\\\.\\C:\\Users', true],
    ['\\\\?\\C:\\Users', true],
    [`\\\\${os.hostname()}\\c$`, true],
    ['\\\\fileserver\\team\\notes', false],
    ['C:\\Users\\me\\Vault', false]
  ])('treats %s as a second path to this machine: %s', (candidate, loopback) => {
    expect(loopbackShare(candidate)).toBe(loopback)
  })
})

describe('kernel pseudo-filesystems wherever they are mounted', () => {
  it.runIf(process.platform === 'linux')('recognizes procfs and sysfs by filesystem type', async () => {
    expect(await onPseudoFilesystem('/proc/self/environ')).toBe(true)
    expect(await onPseudoFilesystem('/sys/kernel')).toBe(true)
    expect(await onPseudoFilesystem(vault)).toBe(false)
  })

  // Stands in for a procfs mounted inside a granted folder (a container's
  // `/host/proc`, a chroot's `proc`): its files are regular and singly linked.
  it.runIf(process.platform === 'linux')('refuses a file whose filesystem is procfs', async () => {
    const files = make()
    const id = await grant(files)
    const environ = path.join(vault, 'environ')
    fs.writeFileSync(environ, 'OPENAI_API_KEY=sk-live\n')
    const original = fs.promises.statfs.bind(fs.promises)

    vi.spyOn(fs.promises, 'statfs').mockImplementation((async (target: fs.PathLike, options?: never) => {
      const real = await original(target, options)

      return String(target) === environ ? { ...real, type: 0x9fa0 } : real
    }) as never)

    for (const op of ['readText', 'readDataUrl', 'stat'] as const) {
      expect(await files.run('plexus', { op, folder: id, path: 'environ' })).toMatchObject({ ok: false, error: 'protected' })
    }

    expect(await files.run('plexus', { op: 'writeText', folder: id, path: 'environ', text: 'x' })).toMatchObject({
      ok: false,
      error: 'protected'
    })
  })
})

describe('Hermes data reached by another spelling', () => {
  it('compares folders by filesystem identity', async () => {
    expect(await overlapsByIdentity(hermesHome, [hermesHome])).toBe(true)
    expect(await overlapsByIdentity(path.join(scratch, 'home'), [hermesHome])).toBe(true)
    fs.mkdirSync(path.join(hermesHome, 'skills'))
    expect(await overlapsByIdentity(path.join(hermesHome, 'skills'), [hermesHome])).toBe(true)
    expect(await overlapsByIdentity(vault, [hermesHome])).toBe(false)
  })
})
