import { useCallback } from 'react'

import { requestComposerFocus, requestComposerInsert, requestComposerInsertRefs } from '@/app/chat/composer/focus'
import { droppedFileInlineRef } from '@/app/chat/composer/inline-refs'
import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { useI18n } from '@/i18n'
import { attachmentId, contextPath, pathLabel } from '@/lib/chat-runtime'
import {
  addComposerAttachment,
  type ComposerAttachment,
  removeComposerAttachment,
  setComposerTerminalSelection,
  updateComposerAttachment
} from '@/store/composer'
import { notify, notifyError } from '@/store/notifications'

import type { ImageDetachResponse } from '../../types'

const IMAGE_EXTENSION_PATTERN = /\.(png|jpe?g|gif|webp|bmp|tiff?|svg|ico)$/i

const BLOB_MIME_EXTENSION: Record<string, string> = {
  'image/bmp': '.bmp',
  'image/gif': '.gif',
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/svg+xml': '.svg',
  'image/tiff': '.tiff',
  'image/webp': '.webp',
  'image/x-icon': '.ico'
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.addEventListener('load', () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
      } else {
        reject(new Error('Could not read file'))
      }
    })
    reader.addEventListener('error', () => reject(reader.error || new Error('Could not read file')))
    reader.readAsDataURL(blob)
  })
}

function blobExtension(blob: Blob): string {
  const mime = blob.type.split(';')[0]?.trim().toLowerCase()

  return (mime && BLOB_MIME_EXTENSION[mime]) || '.png'
}

function browserFileId(kind: ComposerAttachment['kind'], file: File, path = ''): string {
  const key = path || `${file.name}:${file.size}:${file.lastModified}`
  return attachmentId(kind, key)
}

function fileDetail(file: File, path = ''): string {
  if (path) {
    return path
  }

  const size = file.size ? `${Math.ceil(file.size / 1024)} KB` : ''
  const type = file.type || 'file'

  return [type, size].filter(Boolean).join(' · ')
}

function pickBrowserFiles(options: { accept?: string; directory?: boolean } = {}): Promise<File[]> {
  return new Promise(resolve => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    if (options.accept) {
      input.accept = options.accept
    }
    if (options.directory) {
      input.setAttribute('webkitdirectory', '')
    }
    input.style.position = 'fixed'
    input.style.left = '-10000px'
    input.style.top = '-10000px'
    input.addEventListener(
      'change',
      () => {
        const files = Array.from(input.files || [])
        input.remove()
        resolve(files)
      },
      { once: true }
    )
    input.addEventListener(
      'cancel',
      () => {
        input.remove()
        resolve([])
      },
      { once: true }
    )
    document.body.appendChild(input)
    input.click()
  })
}

export function isImagePath(filePath: string): boolean {
  return IMAGE_EXTENSION_PATTERN.test(filePath)
}

export interface DroppedFile {
  /** Browser-native File handle. Absent for in-app drags (e.g. project tree). */
  file?: File
  /** Absolute filesystem path. Empty when an OS drop didn't carry one. */
  path: string
  /** True if the entry is a directory. Currently only set by in-app drags. */
  isDirectory?: boolean
  /** First line number for in-app line-ref drags (source view gutter). */
  line?: number
  /** Last line number for line-range drags (`line..lineEnd` inclusive). */
  lineEnd?: number
}

/** MIME emitted by in-app drag sources (project tree, gutter line numbers).
 * Payload is JSON `{ path; isDirectory?; line?; lineEnd? }[]`. */
export const HERMES_PATHS_MIME = 'application/x-hermes-paths'

/**
 * Eagerly resolve files from a drop event into [File?, path, isDirectory?]
 * triples. Internal Hermes sources (e.g. the project tree) ride on a custom
 * MIME and produce path-only entries; OS drops produce File-bearing entries.
 *
 * Must be called synchronously from inside the drop handler — `DataTransfer`
 * items are detached as soon as the handler returns, and `webUtils.getPathForFile`
 * also requires the original (non-cloned) File reference.
 */
export function extractDroppedFiles(transfer: DataTransfer): DroppedFile[] {
  const result: DroppedFile[] = []
  const seenPaths = new Set<string>()
  const seenFiles = new Set<File>()
  const getPath = window.hermesDesktop?.getPathForFile

  // In-app drags first — they carry richer metadata (isDirectory) than the
  // File-based fallback can provide, and produce no overlapping native files.
  try {
    const internalRaw = transfer.getData(HERMES_PATHS_MIME)

    if (internalRaw) {
      const parsed = JSON.parse(internalRaw) as {
        path?: unknown
        isDirectory?: unknown
        line?: unknown
        lineEnd?: unknown
      }[]

      const positiveInt = (value: unknown) => (typeof value === 'number' && value > 0 ? Math.floor(value) : undefined)

      for (const entry of parsed) {
        if (!entry || typeof entry.path !== 'string' || !entry.path) {
          continue
        }

        const line = positiveInt(entry.line)
        const rawEnd = positiveInt(entry.lineEnd)
        const lineEnd = line && rawEnd && rawEnd > line ? rawEnd : undefined
        const dedupKey = line ? `${entry.path}:${line}-${lineEnd ?? line}` : entry.path

        if (seenPaths.has(dedupKey)) {
          continue
        }

        seenPaths.add(dedupKey)
        result.push({ isDirectory: entry.isDirectory === true, line, lineEnd, path: entry.path })
      }
    }
  } catch {
    // Malformed payload — fall through to native files.
  }

  const fileList = transfer.files

  if (fileList) {
    for (let i = 0; i < fileList.length; i += 1) {
      const file = fileList.item(i)

      if (!file || seenFiles.has(file)) {
        continue
      }

      seenFiles.add(file)
      let path = ''

      if (getPath) {
        try {
          path = getPath(file) || ''
        } catch {
          path = ''
        }
      }

      if (path && seenPaths.has(path)) {
        continue
      }

      if (path) {
        seenPaths.add(path)
      }

      result.push({ file, path })
    }
  }

  const items = transfer.items

  if (items) {
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i]

      if (!item || item.kind !== 'file') {
        continue
      }

      const file = item.getAsFile()

      if (!file || seenFiles.has(file)) {
        continue
      }

      seenFiles.add(file)
      let path = ''

      if (getPath) {
        try {
          path = getPath(file) || ''
        } catch {
          path = ''
        }
      }

      if (path && seenPaths.has(path)) {
        continue
      }

      if (path) {
        seenPaths.add(path)
      }

      result.push({ file, path })
    }
  }

  return result
}

/**
 * Split dropped entries by origin. OS/Finder drops carry a native `File`
 * handle; in-app drags (project tree, gutter line refs) are path-only.
 *
 * The distinction is load-bearing: an in-app path is workspace-relative and
 * resolves on the gateway as-is, so it stays an inline `@file:`/`@line:` ref.
 * An OS drop is an absolute path on *this* machine — the gateway can't read it
 * in remote mode, and an image needs its bytes uploaded to get vision either
 * way. So OS drops must go through the attachment/upload pipeline rather than
 * leaking a local path into the prompt text.
 */
export function partitionDroppedFiles(candidates: DroppedFile[]): {
  osDrops: DroppedFile[]
  inAppRefs: DroppedFile[]
} {
  const osDrops: DroppedFile[] = []
  const inAppRefs: DroppedFile[] = []

  for (const candidate of candidates) {
    if (candidate.file) {
      osDrops.push(candidate)
    } else {
      inAppRefs.push(candidate)
    }
  }

  return { osDrops, inAppRefs }
}

interface ComposerActionsOptions {
  activeSessionId: string | null
  currentCwd: string
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

/** Add to the main composer and focus it. All sidebar/picker/drop attach paths funnel through here. */
const attachToMain = (attachment: ComposerAttachment) => {
  addComposerAttachment(attachment)
  requestComposerFocus('main')
}

export function useComposerActions({ activeSessionId, currentCwd, requestGateway }: ComposerActionsOptions) {
  const { t } = useI18n()
  const copy = t.desktop
  const addTextToDraft = useCallback((text: string) => {
    requestComposerInsert(text, { mode: 'block' })
  }, [])

  const addTerminalSelectionAttachment = useCallback((text: string, label = 'selection') => {
    const trimmed = text.trim()
    const normalizedLabel = label.trim() || 'selection'
    const refText = `@terminal:${formatRefValue(normalizedLabel)}`

    if (!trimmed) {
      return
    }

    setComposerTerminalSelection(normalizedLabel, trimmed)
    requestComposerInsert(refText, { mode: 'inline' })
  }, [])

  const addContextRefAttachment = useCallback((refText: string, label?: string, detail?: string) => {
    const kind: ComposerAttachment['kind'] = refText.startsWith('@folder:')
      ? 'folder'
      : refText.startsWith('@url:')
        ? 'url'
        : 'file'

    attachToMain({
      id: attachmentId(kind, refText),
      kind,
      label: label || refText.replace(/^@(file|folder|url):/, ''),
      detail,
      refText
    })
  }, [])

  const attachBrowserFile = useCallback(
    async (file: File, knownPath = '') => {
      if (!file || file.size === 0) {
        return false
      }

      const isImage = file.type.startsWith('image/') || isImagePath(file.name) || (knownPath && isImagePath(knownPath))
      const kind: ComposerAttachment['kind'] = isImage ? 'image' : 'file'
      const label = pathLabel(knownPath || file.name || (isImage ? 'image' : 'file'))
      const baseAttachment: ComposerAttachment = {
        byteSize: file.size,
        detail: fileDetail(file, knownPath),
        file,
        id: browserFileId(kind, file, knownPath),
        kind,
        label,
        mimeType: file.type || undefined,
        path: knownPath || undefined
      }

      attachToMain(baseAttachment)

      if (!isImage) {
        return true
      }

      try {
        const dataUrl = await blobToDataUrl(file)
        updateComposerAttachment({
          ...baseAttachment,
          dataUrl,
          previewUrl: dataUrl
        })
      } catch (err) {
        notifyError(err, copy.imagePreviewFailed)
      }

      return true
    },
    [copy.imagePreviewFailed]
  )

  const pickContextPaths = useCallback(
    async (kind: 'file' | 'folder') => {
      if (!window.hermesDesktop?.selectPaths) {
        const files = await pickBrowserFiles({ directory: kind === 'folder' })

        for (const file of files) {
          await attachBrowserFile(file)
        }

        return
      }

      const paths = await window.hermesDesktop?.selectPaths({
        title: kind === 'file' ? 'Add files as context' : 'Add folders as context',
        defaultPath: currentCwd || undefined,
        directories: kind === 'folder'
      })

      if (!paths?.length) {
        return
      }

      for (const path of paths) {
        const rel = contextPath(path, currentCwd)

        attachToMain({
          id: attachmentId(kind, rel),
          kind,
          label: pathLabel(path),
          detail: rel,
          refText: `@${kind}:${formatRefValue(rel)}`,
          path
        })
      }
    },
    [attachBrowserFile, currentCwd]
  )

  const insertContextPathInlineRef = useCallback(
    (path: string, isDirectory = false) => {
      if (!path) {
        return false
      }

      const ref = droppedFileInlineRef({ isDirectory, path }, currentCwd)

      if (!ref) {
        return false
      }

      requestComposerInsertRefs([ref])
      requestComposerFocus('main')

      return true
    },
    [currentCwd]
  )

  const attachContextFilePath = useCallback(
    (filePath: string) => {
      if (!filePath) {
        return false
      }

      const rel = contextPath(filePath, currentCwd)

      attachToMain({
        id: attachmentId('file', rel),
        kind: 'file',
        label: pathLabel(filePath),
        detail: rel,
        refText: `@file:${formatRefValue(rel)}`,
        path: filePath
      })

      return true
    },
    [currentCwd]
  )

  const attachImagePath = useCallback(async (filePath: string) => {
    if (!filePath) {
      return false
    }

    const baseAttachment: ComposerAttachment = {
      id: attachmentId('image', filePath),
      kind: 'image',
      label: pathLabel(filePath),
      detail: filePath,
      path: filePath
    }

    attachToMain(baseAttachment)

    try {
      const previewUrl = await window.hermesDesktop?.readFileDataUrl(filePath)

      if (previewUrl) {
        addComposerAttachment({ ...baseAttachment, previewUrl })
      }

      return true
    } catch (err) {
      notifyError(err, copy.imagePreviewFailed)

      return true
    }
  }, [copy.imagePreviewFailed])

  const attachImageBlob = useCallback(
    async (blob: Blob) => {
      if (blob.size === 0) {
        return false
      }

      if (blob.type && !blob.type.startsWith('image/')) {
        return false
      }

      if (!window.hermesDesktop?.saveImageBuffer) {
        const ext = blobExtension(blob)
        const file = new File([blob], `pasted-image${ext}`, {
          lastModified: Date.now(),
          type: blob.type || `image/${ext.slice(1)}`
        })

        return attachBrowserFile(file)
      }

      try {
        const buffer = await blob.arrayBuffer()
        const data = new Uint8Array(buffer)
        const savedPath = await window.hermesDesktop?.saveImageBuffer(data, blobExtension(blob))

        if (!savedPath) {
          notify({ kind: 'error', title: copy.imageAttach, message: copy.imageWriteFailed })

          return false
        }

        return attachImagePath(savedPath)
      } catch (err) {
        notifyError(err, copy.imageAttachFailed)

        return false
      }
    },
    [attachBrowserFile, attachImagePath, copy.imageAttach, copy.imageAttachFailed, copy.imageWriteFailed]
  )

  const pickImages = useCallback(async () => {
    if (!window.hermesDesktop?.selectPaths) {
      const files = await pickBrowserFiles({ accept: 'image/*' })

      for (const file of files) {
        await attachBrowserFile(file)
      }

      return
    }

    const paths = await window.hermesDesktop?.selectPaths({
      title: copy.attachImages,
      defaultPath: currentCwd || undefined,
      filters: [
        {
          name: t.composer.images,
          extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff']
        }
      ]
    })

    if (!paths?.length) {
      return
    }

    for (const path of paths) {
      await attachImagePath(path)
    }
  }, [attachBrowserFile, attachImagePath, copy.attachImages, currentCwd, t.composer.images])

  const pasteClipboardImage = useCallback(async () => {
    try {
      const path = await window.hermesDesktop?.saveClipboardImage()

      if (!path) {
        notify({
          kind: 'warning',
          title: copy.clipboard,
          message: copy.noClipboardImage
        })

        return
      }

      await attachImagePath(path)
    } catch (err) {
      notifyError(err, copy.clipboardPasteFailed)
    }
  }, [attachImagePath, copy.clipboard, copy.clipboardPasteFailed, copy.noClipboardImage])

  const attachContextFolderPath = useCallback(
    (folderPath: string) => {
      if (!folderPath) {
        return false
      }

      const rel = contextPath(folderPath, currentCwd)

      attachToMain({
        id: attachmentId('folder', rel),
        kind: 'folder',
        label: pathLabel(folderPath),
        detail: rel,
        refText: `@folder:${formatRefValue(rel)}`,
        path: folderPath
      })

      return true
    },
    [currentCwd]
  )

  const attachDroppedItems = useCallback(
    async (candidates: DroppedFile[]) => {
      if (candidates.length === 0) {
        return false
      }

      let attached = false
      let lastFailure: string | null = null

      for (const candidate of candidates) {
        const { file, isDirectory, path: knownPath } = candidate

        // Path-only entry (in-app drag from the file browser tree, etc.).
        if (!file) {
          if (isDirectory) {
            if (knownPath && attachContextFolderPath(knownPath)) {
              attached = true

              continue
            }

            lastFailure = `Could not attach folder ${knownPath || ''}`

            continue
          }

          if (knownPath && isImagePath(knownPath)) {
            if (await attachImagePath(knownPath)) {
              attached = true

              continue
            }

            lastFailure = `Could not attach ${knownPath}`

            continue
          }

          if (knownPath && attachContextFilePath(knownPath)) {
            attached = true

            continue
          }

          lastFailure = `Could not attach ${knownPath || 'file'}`

          continue
        }

        const fallbackPath =
          !knownPath && window.hermesDesktop?.getPathForFile ? window.hermesDesktop.getPathForFile(file) : ''

        const filePath = knownPath || fallbackPath || ''
        const isImage = file.type.startsWith('image/') || isImagePath(file.name) || (filePath && isImagePath(filePath))

        if (!filePath && (await attachBrowserFile(file))) {
          attached = true

          continue
        }

        if (isImage) {
          if ((filePath && (await attachImagePath(filePath))) || (await attachImageBlob(file))) {
            attached = true

            continue
          }

          lastFailure = `Could not attach ${file.name || 'image'}`

          continue
        }

        if (filePath && attachContextFilePath(filePath)) {
          attached = true

          continue
        }

        if (await attachBrowserFile(file, filePath)) {
          attached = true

          continue
        }

        lastFailure = `Could not attach ${file.name || 'file'}`
      }

      if (!attached && lastFailure) {
        notify({ kind: 'warning', title: copy.dropFiles, message: lastFailure })
      }

      return attached
    },
    [attachBrowserFile, attachContextFilePath, attachContextFolderPath, attachImageBlob, attachImagePath, copy.dropFiles]
  )

  const removeAttachment = useCallback(
    async (id: string) => {
      const removed = removeComposerAttachment(id)

      if (
        removed?.kind === 'image' &&
        removed.path &&
        activeSessionId &&
        removed.attachedSessionId &&
        removed.attachedSessionId === activeSessionId
      ) {
        await requestGateway<ImageDetachResponse>('image.detach', {
          session_id: activeSessionId,
          path: removed.path
        }).catch(() => undefined)
      }
    },
    [activeSessionId, requestGateway]
  )

  return {
    addContextRefAttachment,
    addTerminalSelectionAttachment,
    addTextToDraft,
    attachContextFilePath,
    attachContextFolderPath,
    attachDroppedItems,
    attachBrowserFile,
    attachImageBlob,
    attachImagePath,
    insertContextPathInlineRef,
    pasteClipboardImage,
    pickContextPaths,
    pickImages,
    removeAttachment
  }
}
