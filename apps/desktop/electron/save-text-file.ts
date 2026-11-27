// T11 (#134889): file-writing slash commands (/save) write on the Electron side. The gateway may be
// remote, so it only returns the rendered content; this module asks the user where the file goes
// (native save dialog) and writes it from the main process. Never a gateway write to the client's disk.

import fs from 'node:fs'
import path from 'node:path'

export interface SaveTextFileRequest {
  content?: unknown
  defaultPath?: unknown
  filters?: unknown
  title?: unknown
}

export interface SaveTextFileResult {
  canceled?: boolean
  path?: string
  saved: boolean
}

export interface SaveTextFileDeps {
  defaultDir: () => string
  showSaveDialog: (options: {
    defaultPath?: string
    filters?: Array<{ extensions: string[]; name: string }>
    properties?: Array<'createDirectory' | 'showOverwriteConfirmation'>
    title: string
  }) => Promise<{ canceled: boolean; filePath?: string }>
  writeFile?: (filePath: string, content: string) => Promise<void>
}

interface IpcMainLike {
  handle: (channel: string, listener: (event: unknown, ...args: any[]) => unknown) => void
}

/** Rendered transcripts can be large; cap well above any realistic /save output. */
export const SAVE_TEXT_FILE_MAX_BYTES = 64 * 1024 * 1024

function safeFilters(value: unknown): Array<{ extensions: string[]; name: string }> | undefined {
  if (!Array.isArray(value)) {
    return undefined
  }

  const filters = value.flatMap(entry => {
    if (!entry || typeof entry !== 'object') {
      return []
    }

    const { extensions, name } = entry as { extensions?: unknown; name?: unknown }

    if (typeof name !== 'string' || !Array.isArray(extensions)) {
      return []
    }

    return [{ name, extensions: extensions.filter((ext): ext is string => typeof ext === 'string') }]
  })

  return filters.length ? filters : undefined
}

/** Only a bare file name is honoured from the renderer: the directory is always the user's pick. */
function suggestedName(value: unknown): string | undefined {
  if (typeof value !== 'string' || !value.trim()) {
    return undefined
  }

  const base = path.basename(value.trim().replaceAll('\\', '/'))

  return base && base !== '.' && base !== '..' ? base : undefined
}

export async function saveTextFileAs(
  request: SaveTextFileRequest,
  deps: SaveTextFileDeps
): Promise<SaveTextFileResult> {
  if (typeof request?.content !== 'string') {
    throw new Error('Invalid content')
  }

  const content = request.content

  if (Buffer.byteLength(content, 'utf8') > SAVE_TEXT_FILE_MAX_BYTES) {
    throw new Error('Content too large')
  }

  const name = suggestedName(request.defaultPath)

  const result = await deps.showSaveDialog({
    title: typeof request.title === 'string' && request.title ? request.title : 'Save',
    defaultPath: name ? path.join(deps.defaultDir(), name) : undefined,
    filters: safeFilters(request.filters),
    // Linux dialogs replace an existing file silently unless asked to confirm.
    properties: ['createDirectory', 'showOverwriteConfirmation']
  })

  if (result.canceled || !result.filePath) {
    return { saved: false, canceled: true }
  }

  const write = deps.writeFile ?? ((filePath: string, text: string) => fs.promises.writeFile(filePath, text, 'utf8'))
  await write(result.filePath, content)

  return { saved: true, path: result.filePath }
}

/** Native save-location picker (profile export etc.): only picks the path; the write happens elsewhere. */
export async function selectSavePath(options: any, deps: SaveTextFileDeps): Promise<null | string> {
  const result = await deps.showSaveDialog({
    title: options?.title || 'Save',
    defaultPath: options?.defaultPath ? String(options.defaultPath) : undefined,
    filters: Array.isArray(options?.filters) ? options.filters : undefined
  })

  return result.canceled || !result.filePath ? null : result.filePath
}

/** `hermes:selectSavePath` (path only) and `hermes:saveTextFileAs` (dialog + main-process write). */
export function registerSaveDialogIpc(ipcMain: IpcMainLike, deps: SaveTextFileDeps): void {
  ipcMain.handle('hermes:selectSavePath', (_event, options = {}) => selectSavePath(options, deps))
  ipcMain.handle('hermes:saveTextFileAs', (_event, request) => saveTextFileAs(request || {}, deps))
}
