import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $composerAttachments, type ComposerAttachment, updateComposerAttachment } from '@/store/composer'
import { $connection } from '@/store/session'

import {
  attachmentPreviewDataUrl,
  type DroppedFile,
  extractDroppedFiles,
  HERMES_PATHS_MIME,
  partitionDroppedFiles,
  useComposerActions
} from './use-composer-actions'

// A Finder/Explorer drop carries a native File handle; an in-app drag (project
// tree, gutter line ref) is path-only. The split decides whether a drop becomes
// an inline @file: ref (in-app, workspace-relative, gateway-resolvable) or goes
// through the upload pipeline (OS drop — absolute local path a remote gateway
// can't read, plus image bytes for vision).
const osDrop = (path: string): DroppedFile => ({ file: new File(['x'], path.split('/').pop() || 'f'), path })
const inAppRef = (path: string, extra: Partial<DroppedFile> = {}): DroppedFile => ({ path, ...extra })

describe('partitionDroppedFiles', () => {
  it('routes File-bearing OS drops to osDrops and path-only in-app drags to inAppRefs', () => {
    const finderPdf = osDrop('/Users/mahmoud/Downloads/DEVIS_signed.pdf')
    const projectFile = inAppRef('src/index.ts')

    const { inAppRefs, osDrops } = partitionDroppedFiles([finderPdf, projectFile])

    expect(osDrops).toEqual([finderPdf])
    expect(inAppRefs).toEqual([projectFile])
  })

  it('treats an OS screenshot drop as an upload target (so it gets byte upload + vision)', () => {
    const screenshot = osDrop('/var/folders/tmp/Screenshot 2026-06-09.png')

    const { inAppRefs, osDrops } = partitionDroppedFiles([screenshot])

    expect(osDrops).toEqual([screenshot])
    expect(inAppRefs).toEqual([])
  })

  it('keeps gutter line-range drags inline (no File handle)', () => {
    const lineRef = inAppRef('src/app.ts', { line: 10, lineEnd: 20 })

    const { inAppRefs, osDrops } = partitionDroppedFiles([lineRef])

    expect(osDrops).toEqual([])
    expect(inAppRefs).toEqual([lineRef])
  })

  it('routes an OS folder drop (path-only, isDirectory) to inAppRefs, not the upload pipeline', () => {
    // extractDroppedFiles emits a dropped directory as a path-only entry so it
    // stays a @folder: ref instead of hitting file.attach, which can't stage a
    // directory ("file not found on gateway and no data_url provided").
    const folder = inAppRef('/Users/jeff/projects/hermes', { isDirectory: true })

    const { inAppRefs, osDrops } = partitionDroppedFiles([folder])

    expect(osDrops).toEqual([])
    expect(inAppRefs).toEqual([folder])
  })

  it('splits a mixed drop and preserves order within each group', () => {
    const a = inAppRef('a.ts')
    const b = osDrop('/abs/b.pdf')
    const c = inAppRef('c.ts')
    const d = osDrop('/abs/d.png')

    const { inAppRefs, osDrops } = partitionDroppedFiles([a, b, c, d])

    expect(inAppRefs).toEqual([a, c])
    expect(osDrops).toEqual([b, d])
  })

  it('returns empty groups for an empty drop', () => {
    expect(partitionDroppedFiles([])).toEqual({ inAppRefs: [], osDrops: [] })
  })
})

// Minimal DataTransfer stand-in. A real OS drop populates BOTH `items` (which
// alone carries webkitGetAsEntry for folder detection) and `files`; the mock
// mirrors that so the dedup path is exercised too.
interface StubEntry {
  path: string
  isDirectory: boolean
}

function stubTransfer(entries: StubEntry[], internalRaw = ''): DataTransfer & { _pathByFile: Map<File, string> } {
  const files = entries.map(entry => new File(['x'], entry.path.split('/').pop() || 'f'))
  const pathByFile = new Map(files.map((file, i) => [file, entries[i].path]))

  const items: Record<number | string, unknown> = { length: entries.length }
  entries.forEach((entry, i) => {
    items[i] = {
      kind: 'file' as const,
      getAsFile: () => files[i],
      webkitGetAsEntry: () => ({ isDirectory: entry.isDirectory, isFile: !entry.isDirectory })
    }
  })

  return {
    getData: (mime: string) => (mime === HERMES_PATHS_MIME ? internalRaw : ''),
    files: {
      length: files.length,
      item: (i: number) => files[i] ?? null
    },
    items,
    _pathByFile: pathByFile
  } as unknown as DataTransfer & { _pathByFile: Map<File, string> }
}

describe('extractDroppedFiles', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  const stubBridge = (transfer: DataTransfer & { _pathByFile: Map<File, string> }) => {
    vi.stubGlobal('window', {
      hermesDesktop: {
        getPathForFile: (file: File) => transfer._pathByFile.get(file) ?? ''
      }
    })
  }

  it('emits a dropped directory as a path-only entry with isDirectory (no File to upload)', () => {
    const transfer = stubTransfer([{ path: '/Users/jeff/projects/hermes', isDirectory: true }]) as DataTransfer & {
      _pathByFile: Map<File, string>
    }

    stubBridge(transfer)

    const result = extractDroppedFiles(transfer)

    expect(result).toHaveLength(1)
    expect(result[0]?.isDirectory).toBe(true)
    expect(result[0]?.path).toBe('/Users/jeff/projects/hermes')
    // A directory carries no bytes — it must NOT ride the File/upload pipeline.
    expect(result[0]?.file).toBeUndefined()
    // And it partitions as an in-app ref (→ @folder:), never an OS upload drop.
    expect(partitionDroppedFiles(result).osDrops).toEqual([])
  })

  it('still emits a dropped file with its native File handle for the upload pipeline', () => {
    const transfer = stubTransfer([
      { path: '/Users/jeff/Downloads/report.pdf', isDirectory: false }
    ]) as DataTransfer & { _pathByFile: Map<File, string> }

    stubBridge(transfer)

    const result = extractDroppedFiles(transfer)

    expect(result).toHaveLength(1)
    expect(result[0]?.isDirectory).toBeFalsy()
    expect(result[0]?.path).toBe('/Users/jeff/Downloads/report.pdf')
    expect(result[0]?.file).toBeInstanceOf(File)
    expect(partitionDroppedFiles(result).osDrops).toHaveLength(1)
  })

  it('classifies a mixed folder+file drop independently', () => {
    const transfer = stubTransfer([
      { path: '/abs/src', isDirectory: true },
      { path: '/abs/notes.txt', isDirectory: false }
    ]) as DataTransfer & { _pathByFile: Map<File, string> }

    stubBridge(transfer)

    const result = extractDroppedFiles(transfer)
    const { inAppRefs, osDrops } = partitionDroppedFiles(result)

    expect(inAppRefs.map(entry => entry.path)).toEqual(['/abs/src'])
    expect(inAppRefs[0]?.isDirectory).toBe(true)
    expect(osDrops.map(entry => entry.path)).toEqual(['/abs/notes.txt'])
  })

  it('does not duplicate a folder that appears in both items and files', () => {
    // Chromium lists a dropped folder in transfer.files too (as a size-0 File);
    // the items pass claims its path first so the files fallback skips it.
    const transfer = stubTransfer([{ path: '/abs/project', isDirectory: true }]) as DataTransfer & {
      _pathByFile: Map<File, string>
    }

    stubBridge(transfer)

    const result = extractDroppedFiles(transfer)

    expect(result).toHaveLength(1)
    expect(result[0]?.isDirectory).toBe(true)
  })
})

describe('attachmentPreviewDataUrl', () => {
  const LOCAL_PREVIEW = 'data:image/png;base64,bG9jYWw='
  const REMOTE_PREVIEW = 'data:image/png;base64,cmVtb3Rl'

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    $connection.set(null)
  })

  it('reads a local path via the local bridge even in remote mode (paperclip/paste/OS drop)', async () => {
    const readFileDataUrl = vi.fn(async () => LOCAL_PREVIEW)
    const api = vi.fn()

    vi.stubGlobal('window', { hermesDesktop: { api, readFileDataUrl } })
    $connection.set({ mode: 'remote' } as never)

    await expect(attachmentPreviewDataUrl('/Users/me/Pictures/pic.png')).resolves.toBe(LOCAL_PREVIEW)

    expect(readFileDataUrl).toHaveBeenCalledWith('/Users/me/Pictures/pic.png')
    expect(api).not.toHaveBeenCalled()
  })

  it('falls back to the remote fs bridge when the path is not on this machine (project-tree drag)', async () => {
    const readFileDataUrl = vi.fn(async () => {
      throw new Error('ENOENT')
    })

    const api = vi.fn(async ({ path }: { path: string }) => {
      if (path.startsWith('/api/fs/read-data-url?')) {
        return { dataUrl: REMOTE_PREVIEW }
      }

      throw new Error(`unexpected path ${path}`)
    })

    vi.stubGlobal('window', { hermesDesktop: { api, readFileDataUrl } })
    $connection.set({ mode: 'remote' } as never)

    await expect(attachmentPreviewDataUrl('/home/gateway/shot.png')).resolves.toBe(REMOTE_PREVIEW)

    expect(api).toHaveBeenCalledWith({
      path: '/api/fs/read-data-url?path=%2Fhome%2Fgateway%2Fshot.png'
    })
  })

  it('falls back when the local bridge returns an empty read', async () => {
    const readFileDataUrl = vi.fn(async () => '')

    const api = vi.fn(async () => ({ dataUrl: REMOTE_PREVIEW }))

    vi.stubGlobal('window', { hermesDesktop: { api, readFileDataUrl } })
    $connection.set({ mode: 'remote' } as never)

    await expect(attachmentPreviewDataUrl('/home/gateway/shot.png')).resolves.toBe(REMOTE_PREVIEW)
  })
})

describe('useComposerActions native image drops', () => {
  afterEach(() => {
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('copies dropped screenshot bytes before trusting a transient macOS path', async () => {
    const transientPath =
      '/var/folders/x7/example/T/TemporaryItems/NSIRD_screencaptureui_4roSuW/Screen Shot 2026-08-11.png'

    const durablePath = '/Users/test/Library/Application Support/Hermes/composer-images/composer_saved.png'
    const previewUrl = 'data:image/png;base64,c2NyZWVuc2hvdA=='

    const screenshot = new File([new Uint8Array([1, 2, 3])], 'Screen Shot 2026-08-11.png', {
      type: 'image/png'
    })

    const saveImageBuffer = vi.fn(async () => durablePath)

    const readFileDataUrl = vi.fn(async (path: string) => {
      if (path === transientPath) {
        throw new Error('temporary screenshot path disappeared')
      }

      return previewUrl
    })

    const add = vi.fn<(attachment: ComposerAttachment) => void>()

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        readFileDataUrl,
        saveImageBuffer
      }
    })

    const { result } = renderHook(() =>
      useComposerActions({
        activeSessionId: null,
        currentCwd: '/Users/test/project',
        requestGateway: vi.fn(),
        scope: {
          add,
          remove: vi.fn(() => null),
          target: 'test-composer',
          update: vi.fn(() => true),
          updateIfCurrent: vi.fn(() => true)
        }
      })
    )

    let attached = false

    await act(async () => {
      attached = await result.current.attachDroppedItems([{ file: screenshot, path: transientPath }])
    })

    expect(attached).toBe(true)
    expect(saveImageBuffer).toHaveBeenCalledOnce()
    expect(readFileDataUrl).toHaveBeenCalledWith(durablePath)
    expect(readFileDataUrl).not.toHaveBeenCalledWith(transientPath)
    // The bounded-preview pipeline no longer retains the full-resolution data
    // URL on the attachment (`previewUrl`); the durable path is the authority
    // and the display thumbnail resolves asynchronously. What matters here is
    // that the attachment is keyed to the DURABLE path, not the transient one.
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'image',
        path: durablePath
      })
    )
  })
})

describe('attachImagePath thumbnail separation', () => {
  // Full-resolution data URL the local bridge returns for a pasted screenshot.
  // Content is irrelevant — the mock bitmap below reports 4000×3000 so the
  // real downscale path runs (the data URL itself is never decoded in jsdom).
  const FULL_RES =
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+GkZcAAAAASUVORK5CYII='

  afterEach(() => {
    vi.unstubAllGlobals()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    $composerAttachments.set([])
    $connection.set(null)
  })

  it('does not resurrect an attachment removed while its queued resize is in flight', async () => {
    const readFileDataUrl = vi.fn(async () => FULL_RES)

    ;(
      window as unknown as {
        hermesDesktop: { readFileDataUrl: typeof readFileDataUrl }
      }
    ).hermesDesktop = { readFileDataUrl }

    let resolveBitmap!: (bitmap: { close: () => void; height: number; width: number }) => void

    const createBitmap = vi.fn(
      () =>
        new Promise<{ close: () => void; height: number; width: number }>(resolve => {
          resolveBitmap = resolve
        })
    )

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ blob: async () => new Blob([new Uint8Array([0])], { type: 'image/png' }) }))
    )
    vi.stubGlobal('createImageBitmap', createBitmap)

    class MockOffscreenCanvas {
      getContext = () => ({ drawImage: vi.fn() })
      convertToBlob = vi.fn(async () => new Blob(['x'], { type: 'image/png' }))
      constructor(_width: number, _height: number) {}
    }

    vi.stubGlobal('OffscreenCanvas', MockOffscreenCanvas)

    const { result } = renderHook(() =>
      useComposerActions({ activeSessionId: null, currentCwd: '', requestGateway: vi.fn() })
    )

    let pending!: Promise<boolean>

    act(() => {
      pending = result.current.attachImagePath('/tmp/shot.png')
    })

    await waitFor(() => expect(createBitmap).toHaveBeenCalledOnce())
    expect($composerAttachments.get()).toHaveLength(1)

    await act(async () => {
      await result.current.removeAttachment('image:/tmp/shot.png')
    })
    expect($composerAttachments.get()).toEqual([])

    resolveBitmap({ close: vi.fn(), height: 3000, width: 4000 })

    await act(async () => {
      await pending
    })

    expect($composerAttachments.get()).toEqual([])
  })

  it('does not apply a removed occurrence thumbnail to the same path when reattached', async () => {
    let resolveSecondRead!: (value: string) => void

    const secondRead = new Promise<string>(resolve => {
      resolveSecondRead = resolve
    })

    const readFileDataUrl = vi.fn().mockResolvedValueOnce(FULL_RES).mockReturnValueOnce(secondRead)

    ;(
      window as unknown as {
        hermesDesktop: { readFileDataUrl: typeof readFileDataUrl }
      }
    ).hermesDesktop = { readFileDataUrl }

    let resolveBitmap!: (bitmap: { close: () => void; height: number; width: number }) => void

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ blob: async () => new Blob([new Uint8Array([0])], { type: 'image/png' }) }))
    )
    vi.stubGlobal(
      'createImageBitmap',
      vi.fn(
        () =>
          new Promise<{ close: () => void; height: number; width: number }>(resolve => {
            resolveBitmap = resolve
          })
      )
    )

    class MockOffscreenCanvas {
      getContext = () => ({ drawImage: vi.fn() })
      convertToBlob = vi.fn(async () => new Blob(['first'], { type: 'image/png' }))
      constructor(_width: number, _height: number) {}
    }

    vi.stubGlobal('OffscreenCanvas', MockOffscreenCanvas)

    const { result } = renderHook(() =>
      useComposerActions({ activeSessionId: null, currentCwd: '', requestGateway: vi.fn() })
    )

    let first!: Promise<boolean>
    let replacement!: Promise<boolean>

    act(() => {
      first = result.current.attachImagePath('/tmp/shot.png')
    })

    await waitFor(() => expect(createImageBitmap).toHaveBeenCalledOnce())

    await act(async () => {
      await result.current.removeAttachment('image:/tmp/shot.png')
    })

    act(() => {
      replacement = result.current.attachImagePath('/tmp/shot.png')
    })

    resolveBitmap({ close: vi.fn(), height: 3000, width: 4000 })
    await waitFor(() => expect(readFileDataUrl).toHaveBeenCalledTimes(2))

    const afterRemovedOccurrenceResolved = $composerAttachments.get()[0]

    resolveSecondRead('data:text/plain;base64,c2Vjb25k')

    await act(async () => {
      await Promise.all([first, replacement])
    })

    expect(afterRemovedOccurrenceResolved?.thumbnailUrl).toBeUndefined()
    expect($composerAttachments.get()[0]?.previewUrl).toBe('data:text/plain;base64,c2Vjb25k')
  })

  it('merges a delayed thumbnail into the latest staged state of the same occurrence', async () => {
    const readFileDataUrl = vi.fn(async () => FULL_RES)

    ;(
      window as unknown as {
        hermesDesktop: { readFileDataUrl: typeof readFileDataUrl }
      }
    ).hermesDesktop = { readFileDataUrl }

    let resolveBitmap!: (bitmap: { close: () => void; height: number; width: number }) => void

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ blob: async () => new Blob([new Uint8Array([0])], { type: 'image/png' }) }))
    )
    vi.stubGlobal(
      'createImageBitmap',
      vi.fn(
        () =>
          new Promise<{ close: () => void; height: number; width: number }>(resolve => {
            resolveBitmap = resolve
          })
      )
    )

    class MockOffscreenCanvas {
      getContext = () => ({ drawImage: vi.fn() })
      convertToBlob = vi.fn(async () => new Blob(['thumbnail'], { type: 'image/png' }))
      constructor(_width: number, _height: number) {}
    }

    vi.stubGlobal('OffscreenCanvas', MockOffscreenCanvas)

    const { result } = renderHook(() =>
      useComposerActions({ activeSessionId: null, currentCwd: '', requestGateway: vi.fn() })
    )

    let pending!: Promise<boolean>

    act(() => {
      pending = result.current.attachImagePath('C:\\Users\\alice\\Pictures\\photo.png')
    })

    await waitFor(() => expect(createImageBitmap).toHaveBeenCalledOnce())

    const original = $composerAttachments.get()[0]!
    updateComposerAttachment({
      ...original,
      attachedSessionId: 'session-1',
      label: 'photo.png',
      path: '/root/.hermes/attachments/photo.png',
      uploadState: undefined
    })

    resolveBitmap({ close: vi.fn(), height: 3000, width: 4000 })

    await act(async () => {
      await pending
    })

    expect($composerAttachments.get()[0]).toMatchObject({
      attachedSessionId: 'session-1',
      path: '/root/.hermes/attachments/photo.png',
      thumbnailUrl: expect.stringMatching(/^data:image\/png;base64,/)
    })
  })

  it('retains only the bounded thumbnail after creating a composer image preview', async () => {
    const readFileDataUrl = vi.fn(async () => FULL_RES)

    ;(
      window as unknown as {
        hermesDesktop: { readFileDataUrl: typeof readFileDataUrl }
      }
    ).hermesDesktop = { readFileDataUrl }

    // Exercise the real downscale path: 4000×3000 bitmap → 512×384 canvas.
    const drawImage = vi.fn()
    const close = vi.fn()

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        blob: async () => new Blob([new Uint8Array([0])], { type: 'image/png' })
      }))
    )
    vi.stubGlobal(
      'createImageBitmap',
      vi.fn(async () => ({ width: 4000, height: 3000, close }))
    )

    class MockOffscreenCanvas {
      getContext = () => ({ drawImage })
      convertToBlob = vi.fn(async () => new Blob(['x'], { type: 'image/png' }))
      constructor(_width: number, _height: number) {}
    }

    vi.stubGlobal('OffscreenCanvas', MockOffscreenCanvas)

    const { result } = renderHook(() =>
      useComposerActions({ activeSessionId: null, currentCwd: '', requestGateway: vi.fn() })
    )

    await act(async () => {
      await result.current.attachImagePath('/tmp/shot.png')
    })

    const attachment = $composerAttachments.get()[0]

    expect(attachment).toBeDefined()
    // The composer retains only the bounded value; full bytes are re-read from
    // `path` on demand by the lightbox and independently by submit/upload.
    expect(attachment?.previewUrl).toBeUndefined()
    expect(attachment?.thumbnailUrl).toBeDefined()
    expect(attachment?.thumbnailUrl).not.toBe(FULL_RES)
    expect(JSON.stringify(attachment)).not.toContain(FULL_RES)
    expect(drawImage).toHaveBeenCalledWith(expect.anything(), 0, 0, 512, 384)
    expect(close).toHaveBeenCalled()
  })

  it('processes 72 image reads one at a time and retains only bounded thumbnails', async () => {
    let activeReads = 0
    let maxActiveReads = 0

    const readFileDataUrl = vi.fn(async () => {
      activeReads += 1
      maxActiveReads = Math.max(maxActiveReads, activeReads)
      await Promise.resolve()
      activeReads -= 1

      return FULL_RES
    })

    ;(
      window as unknown as {
        hermesDesktop: { readFileDataUrl: typeof readFileDataUrl }
      }
    ).hermesDesktop = { readFileDataUrl }

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ blob: async () => new Blob([new Uint8Array([0])], { type: 'image/png' }) }))
    )
    vi.stubGlobal(
      'createImageBitmap',
      vi.fn(async () => ({ close: vi.fn(), height: 6000, width: 6000 }))
    )

    class MockOffscreenCanvas {
      getContext = () => ({ drawImage: vi.fn() })
      convertToBlob = vi.fn(async () => new Blob(['x'], { type: 'image/png' }))
      constructor(_width: number, _height: number) {}
    }

    vi.stubGlobal('OffscreenCanvas', MockOffscreenCanvas)

    const { result } = renderHook(() =>
      useComposerActions({ activeSessionId: null, currentCwd: '', requestGateway: vi.fn() })
    )

    await act(async () => {
      await Promise.all(Array.from({ length: 72 }, (_, index) => result.current.attachImagePath(`/tmp/${index}.png`)))
    })

    const attachments = $composerAttachments.get()

    expect(readFileDataUrl).toHaveBeenCalledTimes(72)
    expect(maxActiveReads).toBe(1)
    expect(attachments).toHaveLength(72)
    expect(attachments.every(attachment => attachment.thumbnailUrl?.startsWith('data:image/'))).toBe(true)
    expect(attachments.every(attachment => attachment.previewUrl === undefined)).toBe(true)
    expect(JSON.stringify(attachments)).not.toContain(FULL_RES)
  })

  it('leaves the thumbnail undefined when the preview is not an image', async () => {
    const readFileDataUrl = vi.fn(async () => 'data:text/plain;base64,aGVsbG8=')

    ;(
      window as unknown as {
        hermesDesktop: { readFileDataUrl: typeof readFileDataUrl }
      }
    ).hermesDesktop = { readFileDataUrl }

    const { result } = renderHook(() =>
      useComposerActions({ activeSessionId: null, currentCwd: '', requestGateway: vi.fn() })
    )

    await act(async () => {
      await result.current.attachImagePath('/tmp/shot.txt')
    })

    const attachment = $composerAttachments.get()[0]

    expect(attachment?.previewUrl).toBe('data:text/plain;base64,aGVsbG8=')
    expect(attachment?.thumbnailUrl).toBeUndefined()
  })
})
