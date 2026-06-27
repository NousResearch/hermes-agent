import { useCallback } from 'react'

import { requestComposerFocus, requestComposerInsert, requestComposerInsertRefs } from '@/app/chat/composer/focus'
import { droppedFileInlineRef } from '@/app/chat/composer/inline-refs'
import { formatRefValue } from '@/components/assistant-ui/directive-text'
import type { HermesWindowInfo } from '@/global'
import { useI18n } from '@/i18n'
import { attachmentId, contextPath, pathLabel } from '@/lib/chat-runtime'
import { readDesktopFileDataUrl, selectDesktopPaths } from '@/lib/desktop-fs'
import { normalize } from '@/lib/text'
import {
  addComposerAttachment,
  type ComposerAttachment,
  removeComposerAttachment,
  setComposerTerminalSelection
} from '@/store/composer'
import { $dockedWindow, dockWindow, undockWindow } from '@/store/dock'
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

function blobExtension(blob: Blob): string {
  const mime = normalize(blob.type.split(';')[0])

  return BLOB_MIME_EXTENSION[mime] || '.png'
}

export function isImagePath(filePath: string): boolean {
  return IMAGE_EXTENSION_PATTERN.test(filePath)
}

/**
 * Read an attachment's thumbnail preview, local disk first. Paperclip picks,
 * clipboard saves, and OS drops always hand us paths on THIS machine — the
 * remote-routed fs facade would 404 them against the gateway and toast a bogus
 * "preview failed" even though the attach itself works (upload reads local
 * bytes too). In-app drags from the remote project tree are the opposite case:
 * the local read fails there, so fall back to the facade (remote fs bridge).
 * In local mode the facade IS the local bridge, so this stays a single read.
 */
export async function attachmentPreviewDataUrl(filePath: string): Promise<string> {
  try {
    const local = await window.hermesDesktop?.readFileDataUrl?.(filePath)

    if (local) {
      return local
    }
  } catch {
    // Not on this machine (or unreadable locally) — try the gateway.
  }

  return readDesktopFileDataUrl(filePath)
}

export interface DroppedFile {
  /** Browser-native File handle. Absent for in-app drags (e.g. project tree). */
  file?: File
  /** Absolute filesystem path. Empty when an OS drop didn't carry one. */
  path: string
  /** True if the entry is a directory. Set by in-app drags, and by OS drops via
   * DataTransferItem.webkitGetAsEntry(). */
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

  // Add a native OS-drop entry. A dropped directory has no byte content to
  // upload, so it's emitted as a path-only entry with `isDirectory: true` —
  // that routes it to a `@folder:` ref / folder attachment (like the folder
  // picker) instead of the file-upload pipeline, which can't stage a directory
  // (the gateway can't read its bytes and there's no data_url to send).
  const pushNativeEntry = (file: File, isDirectory: boolean) => {
    if (seenFiles.has(file)) {
      return
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
      return
    }

    if (path) {
      seenPaths.add(path)
    }

    if (isDirectory) {
      if (path) {
        result.push({ isDirectory: true, path })
      }

      return
    }

    result.push({ file, path })
  }

  // Process items first: DataTransferItem.webkitGetAsEntry() is the only
  // synchronous way to tell a dropped folder from a file, and it lives only on
  // items (not transfer.files). Must be read here, inside the drop handler,
  // before the DataTransfer detaches.
  const items = transfer.items

  if (items) {
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i]

      if (!item || item.kind !== 'file') {
        continue
      }

      let isDirectory = false

      try {
        const entry = typeof item.webkitGetAsEntry === 'function' ? item.webkitGetAsEntry() : null
        isDirectory = entry?.isDirectory === true
      } catch {
        isDirectory = false
      }

      const file = item.getAsFile()

      if (!file) {
        continue
      }

      pushNativeEntry(file, isDirectory)
    }
  }

  // Fallback for environments that populate transfer.files but not items.
  // webkitGetAsEntry isn't available on this path, so directory detection
  // relies on the items pass above; anything reaching here is treated as a file.
  const fileList = transfer.files

  if (fileList) {
    for (let i = 0; i < fileList.length; i += 1) {
      const file = fileList.item(i)

      if (!file) {
        continue
      }

      pushNativeEntry(file, false)
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

  const pickContextPaths = useCallback(
    async (kind: 'file' | 'folder') => {
      const paths = await selectDesktopPaths({
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
    [currentCwd]
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

  const attachImagePath = useCallback(
    async (filePath: string) => {
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
        const previewUrl = await attachmentPreviewDataUrl(filePath)

        if (previewUrl) {
          addComposerAttachment({ ...baseAttachment, previewUrl })
        }

        return true
      } catch (err) {
        notifyError(err, copy.imagePreviewFailed)

        return true
      }
    },
    [copy.imagePreviewFailed]
  )

  const attachImageBlob = useCallback(
    async (blob: Blob) => {
      if (blob.size === 0) {
        return false
      }

      if (blob.type && !blob.type.startsWith('image/')) {
        return false
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
    [attachImagePath, copy.imageAttach, copy.imageAttachFailed, copy.imageWriteFailed]
  )

  const pickImages = useCallback(async () => {
    const paths = await selectDesktopPaths({
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
  }, [attachImagePath, copy.attachImages, currentCwd, t.composer.images])

  const pasteClipboardImage = useCallback(
    async ({ silent = false }: { silent?: boolean } = {}) => {
      try {
        const path = await window.hermesDesktop?.saveClipboardImage()

        if (!path) {
          if (!silent) {
            notify({
              kind: 'warning',
              title: copy.clipboard,
              message: copy.noClipboardImage
            })
          }

          return false
        }

        await attachImagePath(path)

        return true
      } catch (err) {
        if (!silent) {
          notifyError(err, copy.clipboardPasteFailed)
        }

        return false
      }
    },
    [attachImagePath, copy.clipboard, copy.clipboardPasteFailed, copy.noClipboardImage]
  )

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

        lastFailure = `Could not attach ${file.name || 'file'}`
      }

      if (!attached && lastFailure) {
        notify({ kind: 'warning', title: copy.dropFiles, message: lastFailure })
      }

      return attached
    },
    [attachContextFilePath, attachContextFolderPath, attachImageBlob, attachImagePath, copy.dropFiles]
  )

  // Attach an open desktop window as a live target. Unlike file/image
  // attachments (static bytes/paths), this is a handle the agent re-resolves
  // and drives via the hermes-eats-world sidecar. The @window: ref carries the
  // title into the prompt; pid disambiguates same-titled windows. Selecting a
  // window also docks Hermes beside it (app tiled left, Hermes snapped right).
  const attachWindow = useCallback((win: HermesWindowInfo) => {
    const title = (win.name || '').trim() || `PID ${win.pid}`
    const refText = `@window:${formatRefValue(title)}`

    attachToMain({
      id: attachmentId('window', `${refText}#${win.hwnd ?? win.pid}`),
      kind: 'window',
      label: title,
      detail: win.class_name ? `${win.class_name} · PID ${win.pid}` : `PID ${win.pid}`,
      refText,
      pid: win.pid,
      hwnd: win.hwnd ?? undefined
    })

    void dockWindow(win.hwnd, title)
  }, [])

  const removeAttachment = useCallback(
    async (id: string) => {
      const removed = removeComposerAttachment(id)

      // Removing the docked window's chip is the natural "stop controlling it"
      // gesture — undock so the user isn't stranded in the narrow panel with a
      // stale banner.
      if (removed?.kind === 'window' && $dockedWindow.get() === removed.label) {
        void undockWindow()
      }

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
    attachImageBlob,
    attachImagePath,
    attachWindow,
    insertContextPathInlineRef,
    pasteClipboardImage,
    pickContextPaths,
    pickImages,
    removeAttachment
  }
}
