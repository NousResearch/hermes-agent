// IPC surface for remote attachment uploads: the stat-only size preflight and
// the dedicated base64 reader. Extracted from main.ts; the MIME lookup stays
// injected. Both go through the same path bridging and hardening as previews.
import { ipcMain } from 'electron'

import {
  ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
  isMissingFileError,
  missingFileResult,
  readFileDataUrlForIpc,
  resolveRequestedPathForIpc,
  statFilesForAttachIpc
} from './hardening'
import { resolveIpcFileReadPath } from './local-read-path'

export interface AttachIpcDeps {
  mimeTypeForPath: (filePath: string) => string
}

export function registerAttachIpc({ mimeTypeForPath }: AttachIpcDeps) {
  // Stat-only preflight for a send's byte uploads: the renderer sizes the whole
  // batch here and refuses an oversized file or total BEFORE the first
  // readFileDataUrlForAttach call base64-loads anything into memory.
  ipcMain.handle('hermes:statFilesForAttach', async (_event, filePaths) =>
    statFilesForAttachIpc(filePaths, { resolvePath: resolveIpcFileReadPath })
  )

  // Remote attachment transfer is independent of the preview / Settings path.
  // The cap (enforced from stat, before the read) keeps the renderer's base64
  // copy bounded; see ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES.
  ipcMain.handle('hermes:readFileDataUrlForAttach', async (_event, filePath) => {
    const bridgedPath = resolveIpcFileReadPath(filePath)

    try {
      return await readFileDataUrlForIpc(bridgedPath, {
        maxBytes: ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
        mimeType: mimeTypeForPath(resolveRequestedPathForIpc(bridgedPath, { purpose: 'Attachment upload' })),
        purpose: 'Attachment upload'
      })
    } catch (error) {
      if (isMissingFileError(error)) {
        return missingFileResult(filePath, error)
      }

      throw error
    }
  })
}
