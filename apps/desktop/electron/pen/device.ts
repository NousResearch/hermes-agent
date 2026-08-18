// ResourceDevice — the host contract the editor + device manager program
// against (save, dirty, file watching, previews, host IPC doors).

import { randomUUID } from 'node:crypto'
import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

import {
  deletePenFromLibrary,
  findPenInstallation,
  listPenLibrary,
  PEN_SOCKET_APP_NAME,
  type PenInstallation,
  penLibraryPathFor,
  type PenLibraryEntry,
  penLibraryRoot,
  penLoggedIn,
  penSessionFilePath,
  penTemporaryDocumentsRoot,
  renamePenInLibrary,
  requirePenModule
} from '../pen-host'

import { penHostBackground } from './chrome'
import { closeDocument, schedulePenAutosave } from './documents'
import { documents, events, log, type PenRuntime, runtime } from './state'

// ---------------------------------------------------------------------------
// ResourceDevice — the host contract the editor + device manager program
// against. Semantics ported from Pen.app's DesktopResourceDevice, minus the
// BrowserWindow (our canvas lives in a hermes webview) and minus the embedded
// chat agent auth (hermes IS the designer here).
// ---------------------------------------------------------------------------

export function createResourceDevice(rt: PenRuntime, fileURI: string, fileContent: string, hostWindow: () => any) {
  let isDirty = false
  let latestContent = fileContent
  const watched = new Map<string, { refCount: number; close: () => void }>()
  const emitter = new EventEmitter()

  const isTemporary = () => fileURI.startsWith(pathToFileURL(penTemporaryDocumentsRoot()).href)

  const resourceFolder = () => path.dirname(fileURLToPath(fileURI))

  const readSession = () => {
    try {
      return JSON.parse(fs.readFileSync(penSessionFilePath(), 'utf8'))
    } catch {
      return undefined
    }
  }

  let onSave: (uri: string, options?: Record<string, unknown>) => Promise<string> = async () => latestContent

  const device: any = {
    // -- wiring used by the manager in this module (not part of the pen contract)
    __emitter: emitter,
    __setOnSave: (fn: typeof onSave) => {
      onSave = fn
    },
    on: emitter.on.bind(emitter),
    off: emitter.off.bind(emitter),
    emit: emitter.emit.bind(emitter),

    // -- identity
    getResourceURI: () => fileURI,
    getResourceContents: () => fileContent,
    getIsDirty: () => isDirty,
    getDeviceId: () => {
      const machineId = os.hostname() + os.platform() + os.arch()

       
      return require('node:crypto').createHash('md5').update(machineId).digest('hex')
    },
    getHostVersion: () => rt.install.version || '1.0.0',
    isTemporary,
    getResourceFolderPath: async () => resourceFolder(),

    // -- pen.dev login (shared file with Pen.app — one login covers both)
    getSession: () => {
      const session = readSession()

      return session?.email && session?.token ? { email: session.email, token: session.token } : undefined
    },
    setSession: (email: string, token: string) => {
      const existing = readSession() ?? {}

      fs.mkdirSync(path.dirname(penSessionFilePath()), { recursive: true })
      fs.writeFileSync(penSessionFilePath(), JSON.stringify({ ...existing, email, token }, null, 2))
    },
    signOut: () => {
      try {
        fs.unlinkSync(penSessionFilePath())
      } catch {
        // already signed out
      }
    },
    getLastOnlineAt: () => readSession()?.lastOnlineAt,
    setLastOnlineAt: (timestamp: number) => {
      const existing = readSession()

      if (existing) {
        fs.writeFileSync(penSessionFilePath(), JSON.stringify({ ...existing, lastOnlineAt: timestamp }, null, 2))
      }
    },
    getCurrentWorkspace: () => readSession()?.currentWorkspace,
    setCurrentWorkspace: (selection: unknown) => {
      const existing = readSession()

      if (existing) {
        fs.writeFileSync(
          penSessionFilePath(),
          JSON.stringify({ ...existing, currentWorkspace: selection }, null, 2)
        )
      }
    },

    // -- file I/O relative to the document
    readFile: async (filePath: string) => {
      const resolved = path.isAbsolute(filePath) ? filePath : path.join(resourceFolder(), filePath)

      return new Uint8Array(await fs.promises.readFile(resolved))
    },
    statFile: async (filePath: string) => {
      const resolved = path.isAbsolute(filePath) ? filePath : path.join(resourceFolder(), filePath)

      try {
        const stats = await fs.promises.stat(resolved)

        return { exists: true, isFile: stats.isFile(), mtimeMs: stats.mtimeMs }
      } catch (error: any) {
        if (error?.code === 'ENOENT') {
          return { exists: false, isFile: false }
        }

        throw error
      }
    },
    ensureDir: async (dirPath: string) => {
      fs.mkdirSync(dirPath, { recursive: true })
    },
    writeFile: async (filePath: string, contents: Uint8Array) => {
      fs.writeFileSync(filePath, contents)
    },
    watchFile: (uri: string) => {
      const existing = watched.get(uri)

      if (existing) {
        existing.refCount++

        return
      }

      if (!uri.startsWith('file:')) {
        return
      }

      try {
        const watcher = fs.watch(fileURLToPath(uri), { persistent: false }, () => {
          emitter.emit('file-changed', uri)
        })

        watched.set(uri, { refCount: 1, close: () => watcher.close() })
      } catch (error) {
        log.warn(`watchFile failed for ${uri}`, error)
      }
    },
    unwatchFile: (uri: string) => {
      const entry = watched.get(uri)

      if (!entry) {
        return
      }

      entry.refCount--

      if (entry.refCount <= 0) {
        entry.close()
        watched.delete(uri)
      }
    },

    // -- save / dirty lifecycle
    fileChanged: () => {
      if (!isDirty) {
        isDirty = true
        emitter.emit('dirty-changed', true)
      }
    },
    saveResource: async (params: { userAction: boolean; saveAs?: boolean; destinationPath?: string }) => {
      let destination = fileURI

      if (params.saveAs || isTemporary()) {
         
        const { dialog } = require('electron')

        const response = await dialog.showSaveDialog(hostWindow() ?? undefined, {
          title: isTemporary() ? 'Save new .pen file' : 'Save .pen file as…',
          defaultPath: isTemporary() ? 'untitled.pen' : fileURLToPath(fileURI),
          filters: [
            { name: 'Pen Design Files', extensions: ['pen'] },
            { name: 'All Files', extensions: ['*'] }
          ]
        })

        if (response.canceled || !response.filePath) {
          return true // cancelled
        }

        destination = pathToFileURL(response.filePath).href
      }

      try {
        fileContent = await onSave(destination, { assignNewFileToken: params.saveAs })
        latestContent = fileContent
        fs.writeFileSync(fileURLToPath(destination), fileContent, 'utf8')
      } catch (error) {
        log.error('save failed', error)

        return false
      }

      if (destination !== fileURI) {
        // Saved-as: hand the new location back through the load-file path so
        // the tab re-homes onto the real file.
        emitter.emit('load-file', { fileURI: destination, zoomToFit: false, closeCurrent: true })
      }

      if (isDirty) {
        isDirty = false
        emitter.emit('dirty-changed', false)
      }

      return false
    },
    loadFile: (uri: string) => {
      emitter.emit('load-file', { fileURI: uri, zoomToFit: true })
    },

    // -- imports (drag/drop + paste land through these)
    importFiles: async (files: { fileName: string; fileContents: ArrayBufferLike }[]) => {
      const baseDirectory = resourceFolder()
      const imagesDirectory = isTemporary() ? path.join(baseDirectory, 'images') : baseDirectory

      await fs.promises.mkdir(imagesDirectory, { recursive: true })

      const result: { filePath: string }[] = []

      for (const { fileName, fileContents } of files) {
        const ext = path.extname(fileName)
        const base = path.basename(fileName, ext)
        const buffer = Buffer.from(fileContents)
        let candidate = path.join(imagesDirectory, `${base}${ext}`)
        let counter = 0

        for (;;) {
          try {
            await fs.promises.writeFile(candidate, buffer, { flag: 'wx' })
            result.push({ filePath: path.relative(baseDirectory, candidate) })

            break
          } catch (error: any) {
            if (error?.code !== 'EEXIST') {
              throw error
            }
          }

          const existing = await fs.promises.readFile(candidate).catch(() => null)

          if (existing?.equals(buffer)) {
            result.push({ filePath: path.relative(baseDirectory, candidate) })

            break
          }

          counter++
          candidate = path.join(imagesDirectory, `${base}-${counter}${ext}`)
        }
      }

      return result
    },
    importFileByName: async (fileName: string, fileContents: ArrayBufferLike) => {
      const imported = await device.importFiles([{ fileName, fileContents }])

      if (!imported[0]) {
        throw new Error('Failed to import file')
      }

      return imported[0]
    },
    importFileByUri: async (fileUriString: string) => {
      const sourceFile = fileURLToPath(fileUriString)
      const fileContents = fs.readFileSync(sourceFile)
      const imported = await device.importFileByName(path.basename(sourceFile), fileContents.buffer)

      return { filePath: imported.filePath, fileContents: fileContents.buffer }
    },

    // -- appearance / window
    getActiveThemeKind: () => {
      // Follow HERMES's theme, not the OS's. The canvas blends with the app it
      // sits beside; nativeTheme diverges the moment the user themes hermes
      // differently from macOS (dark canvas + black chrome inside a light
      // app). Luminance of the host window background is the ground truth the
      // app itself paints with.
      const hex = penHostBackground().replace('#', '')

      if (hex.length >= 6) {
        const r = parseInt(hex.slice(0, 2), 16)
        const g = parseInt(hex.slice(2, 4), 16)
        const b = parseInt(hex.slice(4, 6), 16)

        return 0.2126 * r + 0.7152 * g + 0.0722 * b < 128 ? 'dark' : 'light'
      }

      const { nativeTheme } = require('electron')

      return nativeTheme.shouldUseDarkColors ? 'dark' : 'light'
    },
    toggleDesignMode: () => {},
    setLeftSidebarVisible: () => {},
    openExternalUrl: (url: string, options?: { showInFolder?: boolean }) => {
       
      const { shell } = require('electron')

      let scheme = ''

      try {
        scheme = new URL(url).protocol.toLowerCase()
      } catch {
        return
      }

      if (options?.showInFolder && scheme === 'file:') {
        shell.showItemInFolder(fileURLToPath(url))

        return
      }

      if (scheme === 'http:' || scheme === 'https:' || scheme === 'file:') {
        shell.openExternal(url)
      }
    },

    // -- the editor's own chat panel (Claude/Codex in-canvas). Hermes is the
    // designer here; the panel degrades to not-connected without these.
    submitPrompt: async (prompt: string) => {
      // "Add to chat" from the canvas — forward into the hermes composer.
      events.emit('add-to-chat', { fileURI, prompt })
    },
    getAgentPackagePath: () => undefined,
    getAgentLoginType: () => undefined,
    getAgentApiKey: () => undefined,
    getAgentEnv: () => undefined,
    agentIncludePartialMessages: () => true,

    // -- temp files (clipboard paste path)
    saveTempFile: async (base64Data: string, ext: string, name?: string) => {
      const tmpDir = path.join(os.tmpdir(), 'pencil-clipboard')

      await fs.promises.mkdir(tmpDir, { recursive: true })

      const filePath = path.join(tmpDir, name || `clipboard-${Date.now()}.${ext}`)

      await fs.promises.writeFile(filePath, Buffer.from(base64Data, 'base64'))

      return filePath
    },
    cleanupTempFiles: async (paths: string[]) => {
      const tmpDir = path.resolve(os.tmpdir(), 'pencil-clipboard')

      for (const p of paths) {
        const resolved = path.resolve(p)

        if (path.dirname(resolved) === tmpDir) {
          await fs.promises.unlink(resolved).catch(() => {})
        }
      }
    },

    // -- workspace folder (the agent cwd for canvas work)
    getWorkspaceFolderPath: async () => resourceFolder(),
    setWorkspaceFolderPath: async () => {},

    // -- libraries
    findLibraries: async () => {
      if (!fileURI.startsWith('file:')) {
        return []
      }

      const libraries: string[] = []
      const ignored = new Set(['node_modules', '.git'])
      const visited = new Set<string>()

      const collect = async (target: string) => {
        let entries: string[]

        try {
          let stats = await fs.promises.stat(target)

          if (stats.isSymbolicLink()) {
            target = await fs.promises.realpath(target)
            stats = await fs.promises.stat(target)
          }

          if (visited.has(target)) {
            return
          }

          visited.add(target)

          if (stats.isDirectory()) {
            entries = await fs.promises.readdir(target)
          } else {
            if (stats.isFile() && target.toLowerCase().endsWith('.lib.pen')) {
              libraries.push(target)
            }

            return
          }
        } catch {
          return
        }

        for (const entry of entries) {
          if (!ignored.has(entry)) {
            await collect(path.join(target, entry))
          }
        }
      }

      await collect(resourceFolder())

      return libraries
    },
    turnIntoLibrary: async () => {
      throw new Error('turn-into-library is not supported in the hermes canvas yet')
    },
    browseLibraries: async (multiple: boolean) => {
       
      const { dialog } = require('electron')

      const result = await dialog.showOpenDialog(hostWindow() ?? undefined, {
        filters: [{ name: 'Pen Libraries', extensions: ['lib.pen'] }],
        properties: multiple ? ['multiSelections'] : undefined
      })

      return result.canceled ? undefined : result.filePaths
    },

    // -- share snapshot import
    pickSnapshotExtractDirectory: async () => {
       
      const { dialog } = require('electron')

      const result = await dialog.showOpenDialog(hostWindow() ?? undefined, {
        properties: ['openDirectory', 'createDirectory'],
        title: 'Choose folder to extract snapshot'
      })

      if (result.canceled || result.filePaths.length === 0) {
        return { cancelled: true }
      }

      return { cancelled: false, directoryPath: result.filePaths[0] }
    },
    writeSnapshotImport: async (
      destinationDir: string,
      items: { relativePath: string; data: Uint8Array | ArrayBufferLike }[],
      rootPenRelativePath: string
    ) => {
      for (const item of items) {
        const outputPath = path.join(destinationDir, item.relativePath)

        await fs.promises.mkdir(path.dirname(outputPath), { recursive: true })
        await fs.promises.writeFile(
          outputPath,
          item.data instanceof Uint8Array ? item.data : new Uint8Array(item.data)
        )
      }

      return { rootFilePath: path.join(destinationDir, rootPenRelativePath) }
    },

    dispose: async () => {
      for (const entry of watched.values()) {
        entry.close()
      }

      watched.clear()
      emitter.removeAllListeners()
    }
  }

  return device
}
