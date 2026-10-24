import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registerDesktopStorageIpc, scanStorage } from './desktop-storage'

let root: string

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-storage-'))
})

afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true })
})

describe('scanStorage', () => {
  it('sums nested files per top-level entry, largest first', async () => {
    fs.mkdirSync(path.join(root, 'logs', 'old'), { recursive: true })
    fs.writeFileSync(path.join(root, 'logs', 'a.log'), Buffer.alloc(100))
    fs.writeFileSync(path.join(root, 'logs', 'old', 'b.log'), Buffer.alloc(50))
    fs.writeFileSync(path.join(root, 'config.yaml'), Buffer.alloc(10))

    const summary = await scanStorage(root)

    expect(summary.root).toBe(root)
    expect(summary.entries).toEqual([
      { name: 'logs', bytes: 150, kind: 'dir' },
      { name: 'config.yaml', bytes: 10, kind: 'file' }
    ])
    expect(summary.totalBytes).toBe(160)
    expect(summary.truncated).toBe(false)
  })

  it('does not follow symlinks out of the root', async () => {
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-outside-'))

    try {
      fs.writeFileSync(path.join(outside, 'big.bin'), Buffer.alloc(1000))
      fs.symlinkSync(outside, path.join(root, 'link'), 'junction')

      const summary = await scanStorage(root)

      expect(summary.totalBytes).toBe(0)
    } finally {
      fs.rmSync(outside, { recursive: true, force: true })
    }
  })

  it('reports truncation when the entry budget is exhausted', async () => {
    // A chain of nested directories leaves work on the stack when the budget runs out.
    fs.mkdirSync(path.join(root, 'deep', 'a', 'b', 'c', 'd', 'e', 'f'), { recursive: true })

    const summary = await scanStorage(root, { maxEntries: 2 })

    expect(summary.truncated).toBe(true)
  })
})

describe('registerDesktopStorageIpc', () => {
  type Handler = (event: unknown, payload?: unknown) => Promise<unknown>

  function setup() {
    const handlers = new Map<string, Handler>()
    const showItemInFolder = vi.fn()

    registerDesktopStorageIpc({
      ipcMain: { handle: (channel, handler) => void handlers.set(channel, handler) },
      hermesHome: () => root,
      showItemInFolder
    })

    return { handlers, showItemInFolder }
  }

  it('scans the home, or the folder named by the trail', async () => {
    fs.mkdirSync(path.join(root, 'logs', 'old'), { recursive: true })
    fs.writeFileSync(path.join(root, 'logs', 'a.log'), Buffer.alloc(7))
    fs.writeFileSync(path.join(root, 'logs', 'old', 'b.log'), Buffer.alloc(3))

    const summary = setup().handlers.get('hermes:storage:summary') as Handler

    expect(((await summary({}, { trail: [] })) as { totalBytes: number }).totalBytes).toBe(10)

    const inner = (await summary({}, { trail: ['logs'] })) as { root: string; entries: { name: string }[] }

    expect(inner.root).toBe(path.join(root, 'logs'))
    expect(inner.entries.map(e => e.name)).toEqual(['a.log', 'old'])
  })

  it('shows an entry in its folder, resolved under the trail', async () => {
    fs.mkdirSync(path.join(root, 'logs'))
    fs.writeFileSync(path.join(root, 'logs', 'a.log'), 'x')

    const { handlers, showItemInFolder } = setup()
    const reveal = handlers.get('hermes:storage:reveal') as Handler

    expect(await reveal({}, { trail: ['logs'], name: 'a.log' })).toEqual({ ok: true })
    expect(showItemInFolder).toHaveBeenCalledWith(path.join(root, 'logs', 'a.log'))
  })

  it('refuses trails and names that escape the Hermes home', async () => {
    const { handlers, showItemInFolder } = setup()
    const reveal = handlers.get('hermes:storage:reveal') as Handler
    const summary = handlers.get('hermes:storage:summary') as Handler

    for (const name of ['..', '../x', 'a/b', 'a\b', '', '.', 42, undefined, 'missing']) {
      expect(await reveal({}, { trail: [], name })).toEqual({ ok: false })
      await expect(summary({}, { trail: [name] })).rejects.toThrow()
    }

    expect(await reveal({}, { trail: 'logs', name: 'x' })).toEqual({ ok: false })
    expect(showItemInFolder).not.toHaveBeenCalled()
  })

  it('refuses a trail through a symlink', async () => {
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-outside-'))

    try {
      fs.symlinkSync(outside, path.join(root, 'link'), 'junction')

      await expect((setup().handlers.get('hermes:storage:summary') as Handler)({}, { trail: ['link'] })).rejects.toThrow()
    } finally {
      fs.rmSync(outside, { recursive: true, force: true })
    }
  })
})
