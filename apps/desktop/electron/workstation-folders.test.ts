import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createWorkstationFoldersHandler, type WorkstationFoldersHandler } from './workstation-folders'

let home: string
let trashed: string[]
let handler: WorkstationFoldersHandler

beforeEach(() => {
  home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-workstation-')))

  for (const name of ['Desktop', 'Documents', 'Downloads']) {
    fs.mkdirSync(path.join(home, name))
  }

  trashed = []
  handler = createWorkstationFoldersHandler({
    home,
    trashItem: async targetPath => {
      trashed.push(targetPath)
      await fs.promises.rm(targetPath, { force: true, recursive: true })
    }
  })
})

afterEach(() => fs.rmSync(home, { force: true, recursive: true }))

describe('workstation folder boundary', () => {
  it('enumerates roots and creates, reads, writes, renames, lists, and trashes entries', async () => {
    const rootResult = await handler('roots')
    expect(rootResult).toMatchObject({
      ok: true,
      roots: {
        desktop: path.join(home, 'Desktop'),
        documents: path.join(home, 'Documents'),
        downloads: path.join(home, 'Downloads')
      }
    })

    const directoryPath = path.join(home, 'Downloads', 'project')
    const sourcePath = path.join(directoryPath, 'draft.bin')
    const destinationPath = path.join(directoryPath, 'final.bin')
    expect(await handler('mkdir', { path: directoryPath })).toMatchObject({
      ok: true,
      entry: { path: directoryPath, isDirectory: true }
    })

    const created = await handler('create', {
      path: sourcePath,
      contentBase64: Buffer.from('one').toString('base64')
    })
    expect(created).toMatchObject({ ok: true, source: { path: sourcePath } })

    const read = await handler('read', { path: sourcePath })
    expect(read).toMatchObject({
      ok: true,
      contentBase64: Buffer.from('one').toString('base64')
    })
    expect(read.source).toMatchObject({
      dev: expect.any(Number),
      ino: expect.any(Number),
      mode: expect.any(Number),
      mtimeMs: expect.any(Number),
      size: 3,
      sha256: expect.any(String)
    })

    const written = await handler('write', {
      source: read.source,
      contentBase64: Buffer.from('two').toString('base64')
    })
    expect(written).toMatchObject({ ok: true, source: { path: sourcePath } })
    expect(fs.readFileSync(sourcePath, 'utf8')).toBe('two')

    const metadata = await handler('stat', { path: sourcePath })
    const renamed = await handler('rename', {
      sourcePath,
      destinationPath,
      source: metadata.entry
    })
    expect(renamed).toMatchObject({
      ok: true,
      entry: { path: destinationPath, isDirectory: false }
    })

    const listed = await handler('list', { path: directoryPath })
    expect(listed).toMatchObject({
      ok: true,
      entries: [{ name: 'final.bin', path: destinationPath, isDirectory: false }]
    })

    expect(await handler('trash', { path: directoryPath })).toEqual({
      ok: true,
      originalPath: directoryPath
    })
    expect(trashed).toEqual([directoryPath])
  })

  it('cleans a partial create and keeps the destination publish atomic', async () => {
    const target = path.join(home, 'Documents', 'broken.txt')
    const failingFs = {
      ...fs,
      promises: {
        ...fs.promises,
        open: async (targetPath: string, flags: number, mode?: number) => {
          const handle = await fs.promises.open(targetPath, flags, mode)

          if (!targetPath.endsWith('.tmp')) {
            return handle
          }

          return {
            close: () => handle.close(),
            stat: () => handle.stat(),
            sync: () => handle.sync(),
            writeFile: async () => {
              throw new Error('write-failed')
            }
          }
        }
      }
    } as unknown as typeof fs
    const failingHandler = createWorkstationFoldersHandler({ fs: failingFs, home })

    expect(
      await failingHandler('create', {
        path: target,
        contentBase64: Buffer.from('partial').toString('base64')
      })
    ).toEqual({ ok: false, error: 'filesystem-error' })
    expect(fs.existsSync(target)).toBe(false)
    expect(fs.readdirSync(path.dirname(target)).some(name => name.includes('.broken.txt.'))).toBe(false)
  })

  it('rejects stale source receipts before write or rename', async () => {
    const sourcePath = path.join(home, 'Documents', 'source.txt')
    fs.writeFileSync(sourcePath, 'original')
    const read = await handler('read', { path: sourcePath })
    const metadata = await handler('stat', { path: sourcePath })
    fs.writeFileSync(sourcePath, 'changed elsewhere')

    expect(
      await handler('write', {
        source: read.source,
        contentBase64: Buffer.from('mine').toString('base64')
      })
    ).toEqual({ ok: false, error: 'source-conflict' })
    expect(
      await handler('rename', {
        sourcePath,
        destinationPath: path.join(home, 'Documents', 'moved.txt'),
        source: metadata.entry
      })
    ).toEqual({ ok: false, error: 'source-conflict' })
    expect(fs.readFileSync(sourcePath, 'utf8')).toBe('changed elsewhere')
  })

  it('reports each known root independently when one is unavailable', async () => {
    const unavailable = path.join(home, 'Documents', 'not-mounted')
    const partialHandler = createWorkstationFoldersHandler({
      home,
      roots: {
        desktop: path.join(home, 'Desktop'),
        documents: unavailable,
        downloads: path.join(home, 'Downloads')
      }
    })

    expect(await partialHandler('roots')).toEqual({
      ok: true,
      roots: {
        desktop: path.join(home, 'Desktop'),
        downloads: path.join(home, 'Downloads')
      }
    })
  })

  it('bounds directory listings and supports exact-name discovery', async () => {
    const downloads = path.join(home, 'Downloads')

    for (let index = 0; index < 205; index += 1) {
      fs.writeFileSync(path.join(downloads, `entry-${index}.txt`), 'x')
    }

    const bounded = await handler('list', { path: downloads })
    expect(bounded).toMatchObject({
      ok: true,
      totalEntries: 205,
      truncated: true
    })
    expect(bounded.entries).toHaveLength(200)

    const remainder = await handler('list', {
      path: downloads,
      offset: bounded.nextOffset,
      limit: 200
    })
    expect(remainder).toMatchObject({
      ok: true,
      nextOffset: 205,
      truncated: false
    })
    expect(remainder.entries).toHaveLength(5)

    const exact = await handler('list', {
      path: downloads,
      query: 'entry-204.txt',
      exact: true
    })
    expect(exact).toMatchObject({
      ok: true,
      totalEntries: 1,
      truncated: false,
      entries: [{ name: 'entry-204.txt', isDirectory: false, size: 1 }]
    })
  })

  it('bounds directory scanning before materializing an unbounded listing', async () => {
    const downloads = path.join(home, 'Downloads')
    const boundedFs = {
      ...fs,
      promises: {
        ...fs.promises,
        opendir: async () => ({
          close: async () => undefined,
          async *[Symbol.asyncIterator]() {
            for (let index = 0; index <= 10_000; index += 1) {
              yield { name: `entry-${index}.txt` }
            }
          }
        })
      }
    } as unknown as typeof fs
    const boundedHandler = createWorkstationFoldersHandler({ fs: boundedFs, home })

    expect(await boundedHandler('list', { path: downloads })).toEqual({
      ok: false,
      error: 'directory-too-large'
    })
  })

  it('cleans the staged temporary file when a write fails before publish', async () => {
    const target = path.join(home, 'Documents', 'write-failure.txt')
    fs.writeFileSync(target, 'original')
    const source = (await handler('read', { path: target })).source
    const failingFs = {
      ...fs,
      promises: {
        ...fs.promises,
        open: async (targetPath: string, flags: number, mode?: number) => {
          const handle = await fs.promises.open(targetPath, flags, mode)

          if (!targetPath.endsWith('.tmp')) {
            return handle
          }

          return {
            close: () => handle.close(),
            sync: () => handle.sync(),
            writeFile: async () => {
              throw new Error('write-failed')
            }
          }
        }
      }
    } as unknown as typeof fs
    const failingHandler = createWorkstationFoldersHandler({ fs: failingFs, home })

    expect(
      await failingHandler('write', {
        source,
        contentBase64: Buffer.from('replacement').toString('base64')
      })
    ).toEqual({ ok: false, error: 'filesystem-error' })
    expect(fs.readFileSync(target, 'utf8')).toBe('original')
    expect(fs.readdirSync(path.dirname(target)).some(name => name.includes('.write-failure.txt.'))).toBe(false)
  })

  it('rejects paths outside approved roots, symlinks, and root mutation', async () => {
    const outside = path.join(home, 'outside.txt')
    fs.writeFileSync(outside, 'private')
    const link = path.join(home, 'Desktop', 'linked.txt')
    fs.symlinkSync(outside, link)

    expect(await handler('read', { path: outside })).toEqual({ ok: false, error: 'unapproved-root' })
    expect(await handler('read', { path: link })).toEqual({ ok: false, error: 'symlink-not-allowed' })
    expect(await handler('trash', { path: path.join(home, 'Desktop') })).toEqual({
      ok: false,
      error: 'root-mutation-forbidden'
    })

    const rootMetadata = await handler('stat', { path: path.join(home, 'Desktop') })
    expect(await handler('rename', {
      sourcePath: path.join(home, 'Desktop'),
      destinationPath: path.join(home, 'Downloads', 'Desktop'),
      source: rootMetadata.entry
    })).toEqual({ ok: false, error: 'root-mutation-forbidden' })
    expect(trashed).toEqual([])
  })

  it('rejects invalid content, oversized files, and unsupported actions', async () => {
    const target = path.join(home, 'Documents', 'note.txt')
    fs.writeFileSync(target, 'note')

    expect(await handler('create', { path: target, contentBase64: 'not base64' })).toEqual({
      ok: false,
      error: 'invalid-content'
    })
    expect(await handler('unknown', {})).toEqual({ ok: false, error: 'unsupported-action' })

    const source = (await handler('read', { path: target })).source
    expect(await handler('write', {
      source,
      contentBase64: Buffer.alloc(10 * 1024 * 1024 + 1).toString('base64')
    })).toEqual({ ok: false, error: 'file-too-large' })
  })

  it('enforces the same root boundary with Windows path semantics', async () => {
    const desktop = String.raw`C:\Users\Tristan\Desktop`
    const filePath = path.win32.join(desktop, 'report.txt')
    const fileStat = {
      dev: 1,
      ino: 2,
      isDirectory: () => false,
      isFile: () => true,
      isSymbolicLink: () => false,
      mode: 0o100600,
      mtimeMs: 3,
      size: 4
    }
    const windowsFs = {
      promises: {
        lstat: async (target: string) => {
          if (target === filePath) return fileStat
          throw Object.assign(new Error('missing'), { code: 'ENOENT' })
        },
        realpath: async (target: string) => target
      }
    } as unknown as typeof fs
    const windowsHandler = createWorkstationFoldersHandler({
      fs: windowsFs,
      path: path.win32,
      roots: {
        desktop,
        documents: String.raw`C:\Users\Tristan\Documents`,
        downloads: String.raw`C:\Users\Tristan\Downloads`
      }
    })

    expect(await windowsHandler('stat', { path: filePath })).toMatchObject({
      ok: true,
      entry: { name: 'report.txt', path: filePath }
    })
    expect(await windowsHandler('stat', {
      path: String.raw`C:\Users\Tristan\Desktop-private\report.txt`
    })).toEqual({ ok: false, error: 'unapproved-root' })
    expect(await windowsHandler('stat', {
      path: String.raw`C:\Windows\System32\drivers\etc\hosts`
    })).toEqual({ ok: false, error: 'unapproved-root' })
  })
})
