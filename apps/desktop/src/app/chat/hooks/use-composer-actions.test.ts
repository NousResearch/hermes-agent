import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import {
  attachmentPreviewDataUrl,
  createFileAttachmentFromBlob,
  createImageAttachmentFromBlob,
  type DroppedFile,
  extractDroppedFiles,
  HERMES_PATHS_MIME,
  normalFilePickerOptions,
  partitionDroppedFiles
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



describe('normal file picker attachments', () => {
  it('uses an unrestricted multi-file OS picker by default', () => {
    expect(normalFilePickerOptions('/workspace')).toEqual({
      defaultPath: '/workspace',
      directories: false,
      multiple: true,
      title: 'Attach files'
    })
    expect(normalFilePickerOptions('/workspace')).not.toHaveProperty('filters')
  })

  it('keeps browser/mobile selected non-image files in memory for file.attach upload', async () => {
    const attachment = await createFileAttachmentFromBlob(
      new File(['hello'], 'notes.weird-extension', { type: 'application/x-test' })
    )

    expect(attachment.kind).toBe('file')
    expect(attachment.label).toBe('notes.weird-extension')
    expect(attachment.path).toBeUndefined()
    expect(attachment.dataUrl).toMatch(/^data:application\/x-test;base64,/)
  })
})

describe('createImageAttachmentFromBlob', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('keeps pasted screenshots in memory instead of requiring saveImageBuffer disk staging', async () => {
    const saveImageBuffer = vi.fn(async () => {
      throw new Error('disk write unavailable')
    })

    vi.stubGlobal('window', { hermesDesktop: { saveImageBuffer } })

    const attachment = await createImageAttachmentFromBlob(new File(['hello'], 'screen.png', { type: 'image/png' }), 'screen.png')

    expect(saveImageBuffer).not.toHaveBeenCalled()
    expect(attachment.kind).toBe('image')
    expect(attachment.label).toBe('screen.png')
    expect(attachment.path).toBeUndefined()
    expect(attachment.previewUrl).toMatch(/^data:image\/png;base64,/)
  })
})
