import type { IpcMain } from 'electron'

import {
  boundedDataUrlReadMaxBytes,
  isMissingFileError,
  missingFileResult,
  readFileDataUrlForIpc,
  resolveRequestedPathForIpc
} from './hardening'

interface FileDataUrlReadOptions {
  maxBytes: () => number
  mimeTypeForPath: (path: string) => string
  resolveReadPath: (path: unknown) => string
}

/** Keep native path translation and missing-file responses at the IPC boundary. */
export function registerFileDataUrlRead(ipc: Pick<IpcMain, 'handle'>, options: FileDataUrlReadOptions) {
  ipc.handle('hermes:readFileDataUrl', async (_event, filePath, relativeToFile, requestedMaxBytes) => {
    const bridgedPath = options.resolveReadPath(filePath)
    const sourcePath = relativeToFile ? options.resolveReadPath(relativeToFile) : undefined

    try {
      return await readFileDataUrlForIpc(bridgedPath, {
        maxBytes: boundedDataUrlReadMaxBytes(requestedMaxBytes, options.maxBytes()),
        mimeType: options.mimeTypeForPath(resolveRequestedPathForIpc(bridgedPath, { purpose: 'File preview' })),
        purpose: 'File preview',
        relativeToFile: sourcePath
      })
    } catch (error) {
      if (isMissingFileError(error)) {
        return missingFileResult(filePath, error)
      }

      throw error
    }
  })
}
