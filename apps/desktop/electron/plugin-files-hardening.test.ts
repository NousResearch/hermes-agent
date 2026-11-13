/**
 * `ctx.files` refusals beyond plain containment: sensitive files moved out of
 * their names, kernel pseudo-filesystems, FIFOs, hard links, a folder or file
 * swapped between the check and the open, and the grants record itself.
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPluginFiles, type PluginFiles } from './plugin-files'

const posix = process.platform !== 'win32'
const unprivileged = posix && process.getuid?.() !== 0

let scratch: string
let userData: string
let hermesHome: string
let vault: string
let files: PluginFiles

const make = () =>
  createPluginFiles({
    userDataDir: () => userData,
    protectedRoots: () => [hermesHome],
    trashItem: target => fs.promises.rm(target, { recursive: true })
  })

beforeEach(() => {
  scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-plugin-files-hard-')))
  userData = path.join(scratch, 'userData')
  hermesHome = path.join(scratch, 'home', '.hermes')
  vault = path.join(scratch, 'home', 'Vault')
  fs.mkdirSync(userData, { recursive: true })
  fs.mkdirSync(hermesHome, { recursive: true })
  fs.writeFileSync(path.join(hermesHome, 'config.yaml'), 'model: secret-config\n')
  fs.mkdirSync(path.join(vault, 'notes'), { recursive: true })
  fs.writeFileSync(path.join(vault, 'notes', 'a.md'), '# A\n')
  files = make()
})

afterEach(() => {
  vi.restoreAllMocks()
  fs.chmodSync(scratch, 0o700)
  fs.rmSync(scratch, { recursive: true, force: true })
})

async function grantVault(): Promise<string> {
  const granted = await files.addGrant('plexus', vault)

  expect(granted.ok).toBe(true)

  return (granted as { folder: { id: string } }).folder.id
}

describe('sensitive files cannot be moved out of their names', () => {
  it.each([
    ['.env', 'notes.txt'],
    ['server.pem', 'p.txt'],
    ['id_ed25519', 'key.txt']
  ])('refuses renaming %s to %s, and trashing it', async (name, renamed) => {
    const id = await grantVault()
    fs.writeFileSync(path.join(vault, name), 'SECRET=sk-live\n')

    expect(await files.run('plexus', { op: 'rename', folder: id, path: name, name: renamed })).toMatchObject({
      ok: false,
      error: 'sensitive-file'
    })
    expect(await files.run('plexus', { op: 'trash', folder: id, path: name })).toMatchObject({
      ok: false,
      error: 'sensitive-file'
    })
    expect(fs.existsSync(path.join(vault, name))).toBe(true)
    expect(fs.existsSync(path.join(vault, renamed))).toBe(false)
  })

  it('refuses renaming a key folder so its files lose the rule', async () => {
    const id = await grantVault()
    fs.mkdirSync(path.join(vault, '.ssh'))
    fs.writeFileSync(path.join(vault, '.ssh', 'config'), 'Host x\n')

    expect(await files.run('plexus', { op: 'rename', folder: id, path: '.ssh', name: 'ssh' })).toMatchObject({
      ok: false,
      error: 'sensitive-file'
    })
    expect(await files.run('plexus', { op: 'readText', folder: id, path: '.ssh/config' })).toMatchObject({
      ok: false,
      error: 'sensitive-file'
    })
  })
})

describe('only plain files', () => {
  it.runIf(process.platform === 'linux').each(['/proc', '/sys', '/dev', '/proc/self'])(
    'refuses to grant the kernel pseudo-filesystem %s',
    async folder => {
      expect(await files.addGrant('plexus', folder)).toMatchObject({
        ok: false,
        error: 'protected',
        message: expect.stringContaining('system folder')
      })
    }
  )

  it.runIf(posix)('refuses a FIFO instead of blocking on it', async () => {
    const id = await grantVault()
    execFileSync('mkfifo', [path.join(vault, 'pipe')])

    const result = await Promise.race([
      files.run('plexus', { op: 'readText', folder: id, path: 'pipe' }),
      new Promise(resolve => setTimeout(() => resolve('blocked'), 1500))
    ])

    expect(result).toMatchObject({ ok: false, error: 'not-a-file' })
    expect(await files.run('plexus', { op: 'writeText', folder: id, path: 'pipe', text: 'x' })).toMatchObject({
      ok: false,
      error: 'not-a-file'
    })
  })

  it.runIf(posix)('refuses a hard link, which may be a second name for Hermes data', async () => {
    const id = await grantVault()
    fs.linkSync(path.join(hermesHome, 'config.yaml'), path.join(vault, 'notes', 'c.md'))

    expect(await files.run('plexus', { op: 'readText', folder: id, path: 'notes/c.md' })).toMatchObject({
      ok: false,
      error: 'protected'
    })
    expect(await files.run('plexus', { op: 'writeText', folder: id, path: 'notes/c.md', text: 'pwned' })).toMatchObject(
      { ok: false, error: 'protected' }
    )
    expect(fs.readFileSync(path.join(hermesHome, 'config.yaml'), 'utf8')).toBe('model: secret-config\n')
  })
})

describe('a swap between the check and the open is refused', () => {
  // Stands in for another process changing the folder right after `resolve`
  // looked at `checked` (with lstat, or with stat for its last look).
  const swapAfterCheck = (method: 'lstat' | 'stat', checked: string, swap: () => void) => {
    const original = fs.promises[method].bind(fs.promises) as (target: fs.PathLike) => Promise<fs.Stats>
    let done = false

    vi.spyOn(fs.promises, method).mockImplementation((async (target: fs.PathLike) => {
      try {
        return await original(target)
      } finally {
        if (!done && String(target) === checked) {
          done = true
          swap()
        }
      }
    }) as never)
  }

  it.runIf(posix)('a parent folder swapped for a link into the Hermes home: no file lands there', async () => {
    const id = await grantVault()
    const sub = path.join(vault, 'sub')
    fs.mkdirSync(sub)

    swapAfterCheck('lstat', path.join(sub, 'pwn.txt'), () => {
      fs.renameSync(sub, path.join(vault, 'sub-old'))
      fs.symlinkSync(hermesHome, sub)
    })

    // Refused either as outside the folder or, when the swap lands before the
    // identity check, as Hermes data.
    expect(
      (
        (await files.run('plexus', { op: 'writeText', folder: id, path: 'sub/pwn.txt', text: 'pwned' })) as {
          error?: string
        }
      ).error
    ).toMatch(/^(outside-folder|protected)$/)
    expect(fs.existsSync(path.join(hermesHome, 'pwn.txt'))).toBe(false)
  })

  it.runIf(posix)('a checked file swapped for a link to Hermes data is not read or written', async () => {
    const id = await grantVault()
    const note = path.join(vault, 'notes', 'a.md')

    const swap = () => {
      fs.rmSync(note)
      fs.symlinkSync(path.join(hermesHome, 'config.yaml'), note)
    }

    swapAfterCheck('stat', note, swap)
    const read = await files.run('plexus', { op: 'readText', folder: id, path: 'notes/a.md' })

    expect(read).toMatchObject({ ok: false })
    expect(JSON.stringify(read)).not.toContain('secret-config')

    vi.restoreAllMocks()
    fs.rmSync(note)
    fs.writeFileSync(note, '# A\n')
    swapAfterCheck('stat', note, swap)

    expect(await files.run('plexus', { op: 'writeText', folder: id, path: 'notes/a.md', text: 'pwned' })).toMatchObject(
      { ok: false }
    )
    expect(fs.readFileSync(path.join(hermesHome, 'config.yaml'), 'utf8')).toBe('model: secret-config\n')
  })
})

describe('a swap before the final look is refused too', () => {
  // Stands in for another process acting right after `resolve` realpath'd the
  // target, before it took the identity the open is compared against.
  const afterRealpath = (checked: string, swap: () => void) => {
    const original = fs.promises.realpath.bind(fs.promises) as (target: fs.PathLike) => Promise<string>
    let done = false

    vi.spyOn(fs.promises, 'realpath').mockImplementation((async (target: fs.PathLike) => {
      try {
        return await original(target)
      } finally {
        if (!done && String(target) === checked) {
          done = true
          swap()
        }
      }
    }) as never)
  }

  it.runIf(posix)('a checked file swapped for a link to Hermes data: not read, not written', async () => {
    const id = await grantVault()
    const note = path.join(vault, 'notes', 'a.md')

    const swap = () => {
      fs.rmSync(note)
      fs.symlinkSync(path.join(hermesHome, 'config.yaml'), note)
    }

    afterRealpath(note, swap)
    const read = await files.run('plexus', { op: 'readText', folder: id, path: 'notes/a.md' })

    expect(read).toMatchObject({ ok: false })
    expect(JSON.stringify(read)).not.toContain('secret-config')

    vi.restoreAllMocks()
    fs.rmSync(note)
    fs.writeFileSync(note, '# A\n')
    afterRealpath(note, swap)

    expect(await files.run('plexus', { op: 'writeText', folder: id, path: 'notes/a.md', text: 'pwned' })).toMatchObject(
      {
        ok: false
      }
    )
    expect(fs.readFileSync(path.join(hermesHome, 'config.yaml'), 'utf8')).toBe('model: secret-config\n')
  })

  it.runIf(posix)('a parent folder swapped for a link into the Hermes home: not read, not written', async () => {
    const id = await grantVault()
    fs.writeFileSync(path.join(hermesHome, 'a.md'), 'hermes-secret\n')
    const notes = path.join(vault, 'notes')
    const note = path.join(notes, 'a.md')

    const swap = () => {
      fs.renameSync(notes, path.join(vault, 'notes-old'))
      fs.symlinkSync(hermesHome, notes)
    }

    afterRealpath(note, swap)
    const read = await files.run('plexus', { op: 'readText', folder: id, path: 'notes/a.md' })

    expect(read).toMatchObject({ ok: false })
    expect(JSON.stringify(read)).not.toContain('hermes-secret')

    vi.restoreAllMocks()
    fs.rmSync(notes)
    fs.renameSync(path.join(vault, 'notes-old'), notes)
    afterRealpath(note, swap)

    expect(await files.run('plexus', { op: 'writeText', folder: id, path: 'notes/a.md', text: 'pwned' })).toMatchObject(
      {
        ok: false
      }
    )
    expect(fs.readFileSync(path.join(hermesHome, 'a.md'), 'utf8')).toBe('hermes-secret\n')
  })
})

describe('rename, trash and list after a parent swap', () => {
  // Another process swaps `notes` for a link into the Hermes home right after
  // `resolve` finished (its last look is the second lstat of the entry): the
  // op must not act on the Hermes home.
  const swapParentAfterResolve = (entry: string, notes: string) => {
    const original = fs.promises.lstat.bind(fs.promises) as (target: fs.PathLike) => Promise<fs.Stats>
    let seen = 0

    vi.spyOn(fs.promises, 'lstat').mockImplementation((async (target: fs.PathLike) => {
      try {
        return await original(target)
      } finally {
        if (String(target) === entry && ++seen === 2) {
          fs.renameSync(notes, `${notes}-old`)
          fs.symlinkSync(hermesHome, notes)
        }
      }
    }) as never)
  }

  beforeEach(() => {
    fs.writeFileSync(path.join(hermesHome, 'a.md'), 'hermes-file\n')
  })

  it.runIf(posix)('rename does not move a Hermes file', async () => {
    const id = await grantVault()
    const notes = path.join(vault, 'notes')
    swapParentAfterResolve(path.join(notes, 'a.md'), notes)

    expect(await files.run('plexus', { op: 'rename', folder: id, path: 'notes/a.md', name: 'b.md' })).toMatchObject({
      ok: false
    })
    expect(fs.readFileSync(path.join(hermesHome, 'a.md'), 'utf8')).toBe('hermes-file\n')
    expect(fs.existsSync(path.join(hermesHome, 'b.md'))).toBe(false)
  })

  it.runIf(posix)('a swap in the last instant before the rename is undone', async () => {
    const id = await grantVault()
    const notes = path.join(vault, 'notes')
    const original = fs.promises.rename.bind(fs.promises) as (from: fs.PathLike, to: fs.PathLike) => Promise<void>
    let swapped = false

    vi.spyOn(fs.promises, 'rename').mockImplementation((async (from: fs.PathLike, to: fs.PathLike) => {
      if (!swapped && String(from) === path.join(notes, 'a.md')) {
        swapped = true
        fs.renameSync(notes, `${notes}-old`)
        fs.symlinkSync(hermesHome, notes)
      }

      return original(from, to)
    }) as never)

    expect(await files.run('plexus', { op: 'rename', folder: id, path: 'notes/a.md', name: 'b.md' })).toMatchObject({
      ok: false,
      error: 'outside-folder'
    })
    expect(fs.readFileSync(path.join(hermesHome, 'a.md'), 'utf8')).toBe('hermes-file\n')
    expect(fs.existsSync(path.join(hermesHome, 'b.md'))).toBe(false)
  })

  it.runIf(posix)('the undo never overwrites a file recreated meanwhile', async () => {
    const id = await grantVault()
    const notes = path.join(vault, 'notes')
    const original = fs.promises.rename.bind(fs.promises) as (from: fs.PathLike, to: fs.PathLike) => Promise<void>
    let calls = 0

    vi.spyOn(fs.promises, 'rename').mockImplementation((async (from: fs.PathLike, to: fs.PathLike) => {
      if (++calls === 1) {
        fs.renameSync(notes, `${notes}-old`)
        fs.symlinkSync(hermesHome, notes)
        await original(from, to)
        // Another app writes a new a.md where the moved one was.
        fs.writeFileSync(path.join(hermesHome, 'a.md'), 'newer\n')

        return
      }

      return original(from, to)
    }) as never)

    expect(await files.run('plexus', { op: 'rename', folder: id, path: 'notes/a.md', name: 'b.md' })).toMatchObject({
      ok: false,
      error: 'outside-folder'
    })
    expect(fs.readFileSync(path.join(hermesHome, 'a.md'), 'utf8')).toBe('newer\n')
    expect(fs.readFileSync(path.join(hermesHome, 'b.md'), 'utf8')).toBe('hermes-file\n')
  })

  it.runIf(posix)('trash does not remove a Hermes file', async () => {
    const id = await grantVault()
    const notes = path.join(vault, 'notes')
    swapParentAfterResolve(path.join(notes, 'a.md'), notes)

    expect(await files.run('plexus', { op: 'trash', folder: id, path: 'notes/a.md' })).toMatchObject({ ok: false })
    expect(fs.readFileSync(path.join(hermesHome, 'a.md'), 'utf8')).toBe('hermes-file\n')
  })

  it.runIf(posix)('list does not return the Hermes home', async () => {
    const id = await grantVault()
    const sub = path.join(vault, 'notes', 'sub')
    fs.mkdirSync(sub)
    fs.mkdirSync(path.join(hermesHome, 'sub'))
    fs.writeFileSync(path.join(hermesHome, 'sub', 'config.yaml'), 'x\n')
    swapParentAfterResolve(sub, path.join(vault, 'notes'))

    const listed = await files.run('plexus', { op: 'list', folder: id, path: 'notes/sub' })

    expect(listed).toMatchObject({ ok: false })
    expect(JSON.stringify(listed)).not.toContain('config.yaml')
  })
})

describe('Hermes data mounted inside a granted folder', () => {
  // Stands in for a bind mount of the Hermes home at Vault/mounted (present
  // before the grant or added after): same filesystem identity, inside path.
  it('refuses every op under it', async () => {
    const mounted = path.join(vault, 'mounted')
    fs.mkdirSync(mounted)
    fs.writeFileSync(path.join(mounted, 'config.yaml'), 'mirror\n')
    const id = await grantVault()
    const home = fs.statSync(hermesHome)
    const original = fs.promises.stat.bind(fs.promises) as (target: fs.PathLike, options?: never) => Promise<fs.Stats>

    vi.spyOn(fs.promises, 'stat').mockImplementation((async (target: fs.PathLike, options?: never) => {
      const real = await original(target, options)

      return String(target) === mounted
        ? Object.assign(Object.create(Object.getPrototypeOf(real)), real, { dev: home.dev, ino: home.ino })
        : real
    }) as never)

    for (const request of [
      { op: 'readText', path: 'mounted/config.yaml' },
      { op: 'list', path: 'mounted' },
      { op: 'stat', path: 'mounted/config.yaml' },
      { op: 'writeText', path: 'mounted/new.md', text: 'x' }
    ] as const) {
      expect(await files.run('plexus', { ...request, folder: id } as never)).toMatchObject({
        ok: false,
        error: 'protected'
      })
    }

    expect(fs.existsSync(path.join(mounted, 'new.md'))).toBe(false)
  })
})

describe('the grants record', () => {
  it.runIf(unprivileged)('a record that cannot be read is not overwritten with nothing', async () => {
    await grantVault()
    const record = path.join(userData, 'plugin-files', 'grants.json')
    const before = fs.readFileSync(record, 'utf8')
    fs.chmodSync(record, 0o000)
    const other = path.join(scratch, 'Other')
    fs.mkdirSync(other)

    expect(await make().addGrant('plexus', other)).toMatchObject({ ok: false, error: 'io-error' })
    fs.chmodSync(record, 0o600)
    expect(fs.readFileSync(record, 'utf8')).toBe(before)
  })

  it.runIf(unprivileged)('a revoke that cannot be saved does not take effect now or after restart', async () => {
    const id = await grantVault()
    const base = path.join(userData, 'plugin-files')
    fs.chmodSync(base, 0o500)

    await expect(files.revoke('plexus', id)).rejects.toThrow()
    fs.chmodSync(base, 0o700)

    expect((await files.grants('plexus')).map(row => row.id)).toEqual([id])
    expect((await make().grants('plexus')).map(row => row.id)).toEqual([id])
  })

  it('a first read still in flight cannot undo a revoke made meanwhile', async () => {
    const id = await grantVault()
    const fresh = make()
    const original = fs.promises.readFile.bind(fs.promises) as (...args: unknown[]) => Promise<unknown>

    let releaseFirst = () => {}
    const firstHeld = new Promise<void>(resolve => (releaseFirst = resolve))
    let calls = 0

    // The first read of the record finishes only after the revoke has saved.
    vi.spyOn(fs.promises, 'readFile').mockImplementation((async (...args: unknown[]) => {
      const data = await original(...args)

      if (String(args[0]).endsWith('grants.json') && calls++ === 0) {
        await firstHeld
      }

      return data
    }) as never)

    const reading = fresh.grants('plexus')
    const revoking = fresh.revoke('plexus', id)
    await new Promise(resolve => setTimeout(resolve, 200))
    releaseFirst()

    expect((await Promise.all([reading, revoking]))[1]).toEqual({ ok: true, removed: 1 })
    expect(await fresh.grants('plexus')).toEqual([])
    expect(await fresh.run('plexus', { op: 'list', folder: id, path: '' })).toMatchObject({
      ok: false,
      error: 'not-granted'
    })
  })

  it('treats an id such as `constructor` as a plain key', async () => {
    expect(await files.addGrant('constructor', vault)).toMatchObject({ ok: true })
    expect(await files.grants('toString')).toEqual([])
  })

  it.each(['foo.', 'con', 'NUL.txt', 'com1'])('refuses the plugin id %s, which Windows would alias', async id => {
    expect(await files.run(id, { op: 'list', folder: 'data', path: '' })).toMatchObject({
      ok: false,
      error: 'invalid-plugin'
    })
  })
})
