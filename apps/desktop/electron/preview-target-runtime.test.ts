import { statSync } from 'node:fs'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { createPreviewTargetRuntime, registerPreviewTargetIpc } from './preview-target-runtime'

const disposableDirectories: string[] = []
const runtimes: ReturnType<typeof createPreviewTargetRuntime>[] = []
const disposableRoot = tmpdir()

async function disposableDirectory() {
  const dir = await mkdtemp(path.join(disposableRoot, 'preview-target-e02-'))

  disposableDirectories.push(dir)

  return dir
}

function runtime(home: string) {
  const send = vi.fn()

  const isType = (filePath: string, type: 'file' | 'directory') => {
    try {
      const stat = statSync(filePath)

      return type === 'file' ? stat.isFile() : stat.isDirectory()
    } catch {
      return false
    }
  }

  const target = createPreviewTargetRuntime({
    app: { getPath: () => home } as Parameters<typeof createPreviewTargetRuntime>[0]['app'],
    directoryExists: filePath => isType(filePath, 'directory'),
    fileExists: filePath => isType(filePath, 'file'),
    getMainWindow: () =>
      ({ isDestroyed: () => false, webContents: { isDestroyed: () => false, send } }) as unknown as ReturnType<
        Parameters<typeof createPreviewTargetRuntime>[0]['getMainWindow']
      >,
    hermesHome: home,
    mimeTypeForPath: filePath => (path.extname(filePath) === '.png' ? 'image/png' : 'application/octet-stream'),
    resolveHermesCwd: () => home
  })

  runtimes.push(target)

  return { send, target }
}

afterEach(async () => {
  for (const target of runtimes.splice(0)) {
    target.closePreviewWatchers()
  }

  for (const dir of disposableDirectories.splice(0)) {
    const resolved = path.resolve(dir)
    const expectedParent = path.resolve(disposableRoot)

    if (path.dirname(resolved) !== expectedParent || !path.basename(resolved).startsWith('preview-target-e02-')) {
      throw new Error(`Refusing to remove unexpected temporary directory: ${resolved}`)
    }

    await rm(dir, { force: true, recursive: true })
  }
})

describe('preview target runtime', () => {
  it('normalizes a real local file and refuses sensitive files and remote hosts', async () => {
    const dir = await disposableDirectory()
    const image = path.join(dir, 'image.png')
    const secret = path.join(dir, 'private.pem')
    await writeFile(image, Buffer.from([0x89, 0x50, 0x4e, 0x47]))
    await writeFile(secret, 'private test data')
    const { target } = runtime(dir)

    await expect(target.normalizePreviewTarget(image, dir)).resolves.toMatchObject({
      kind: 'file',
      label: 'image.png',
      mimeType: 'image/png',
      path: image,
      previewKind: 'image',
      url: pathToFileURL(image).toString()
    })
    await expect(target.normalizePreviewTarget(secret, dir)).resolves.toBeNull()
    await expect(target.watchPreviewFile(pathToFileURL(secret).toString())).rejects.toThrow(/sensitive file/i)
    await expect(target.normalizePreviewTarget('https://example.com/', dir)).resolves.toBeNull()
    await expect(target.normalizePreviewTarget('http://0.0.0.0:5173/app', dir)).resolves.toMatchObject({
      kind: 'url',
      url: 'http://127.0.0.1:5173/app'
    })
  })

  it('watches the requested file and closes its watch by id', async () => {
    const dir = await disposableDirectory()
    const file = path.join(dir, 'watched.txt')
    await writeFile(file, 'first')
    const { send, target } = runtime(dir)
    const watch = await target.watchPreviewFile(pathToFileURL(file).toString())

    await writeFile(file, 'second')
    await vi.waitFor(
      () =>
        expect(send).toHaveBeenCalledWith('hermes:preview-file-changed', {
          id: watch.id,
          path: file,
          url: pathToFileURL(file).toString()
        }),
      { timeout: 3000 }
    )

    expect(target.stopPreviewFileWatch(watch.id)).toBe(true)
    expect(target.stopPreviewFileWatch(watch.id)).toBe(false)
    send.mockClear()
    await writeFile(file, 'third')
    await new Promise(resolve => setTimeout(resolve, 250))
    expect(send).not.toHaveBeenCalled()
  })

  it('watches directory entry churn through the same registry and closes all watches', async () => {
    const dir = await disposableDirectory()
    const existingFile = path.join(dir, 'existing.txt')
    await writeFile(existingFile, 'existing')
    const { send, target } = runtime(dir)
    const fileWatch = await target.watchPreviewFile(pathToFileURL(existingFile).toString())
    const watch = target.watchDirectory(dir)

    await writeFile(path.join(dir, 'new-entry.txt'), 'new')
    await vi.waitFor(
      () =>
        expect(send).toHaveBeenCalledWith('hermes:preview-file-changed', {
          id: watch.id,
          path: dir,
          url: pathToFileURL(dir).toString()
        }),
      { timeout: 3000 }
    )

    target.closePreviewWatchers()
    expect(target.stopPreviewFileWatch(watch.id)).toBe(false)
    expect(target.stopPreviewFileWatch(fileWatch.id)).toBe(false)
  })

  it('registers the original IPC names and string coercion against a real temporary file', async () => {
    const dir = await disposableDirectory()
    const file = path.join(dir, 'from-ipc.txt')
    await writeFile(file, 'text')
    const { target } = runtime(dir)
    const handlers = new Map<string, (...args: unknown[]) => unknown>()
    registerPreviewTargetIpc(
      {
        handle: (name, handler) => {
          handlers.set(name, handler)
        }
      } as Parameters<typeof registerPreviewTargetIpc>[0],
      target
    )

    expect([...handlers.keys()]).toEqual([
      'hermes:normalizePreviewTarget',
      'hermes:watchPreviewFile',
      'hermes:watchDirectory',
      'hermes:stopPreviewFileWatch'
    ])
    await expect(handlers.get('hermes:normalizePreviewTarget')?.(undefined, file, dir)).resolves.toMatchObject({
      path: file
    })

    const watch = (await handlers.get('hermes:watchPreviewFile')?.(undefined, pathToFileURL(file).toString())) as {
      id: string
      path: string
    }

    expect(watch).toMatchObject({ path: file })
    expect(handlers.get('hermes:stopPreviewFileWatch')?.(undefined, watch.id)).toBe(true)

    const directoryWatch = handlers.get('hermes:watchDirectory')?.(undefined, dir) as { id: string; path: string }
    expect(directoryWatch.path).toBe(dir)
    expect(handlers.get('hermes:stopPreviewFileWatch')?.(undefined, directoryWatch.id)).toBe(true)
  })
})
