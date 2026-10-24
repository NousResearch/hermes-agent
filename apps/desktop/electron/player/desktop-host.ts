import path from 'node:path'

import { app, type BrowserWindow, dialog, ipcMain, shell } from 'electron'

import { registerPlayerIpc } from '../player-ipc'
import { PlayerMediaJobs } from '../player-media-jobs'

import { PlayerManager } from './manager'

/** Composes native player ownership without changing Hermes gateway startup. */
export function createDesktopPlayerHost() {
  const mediaJobs = new PlayerMediaJobs({
    stateDirectory: path.join(app.getPath('userData'), 'hermestv-media'),
    workerPath: path.join(
      app.isPackaged ? path.join(process.resourcesPath, 'app.asar.unpacked') : app.getAppPath(),
      'dist',
      'hermestv-worker.py'
    ),
    nodeExecutable: process.execPath,
    async chooseDestination(owner, fileName) {
      const extension = fileName.endsWith('.m4a') ? 'm4a' : 'mp4'

      const result = await dialog.showSaveDialog(owner, {
        defaultPath: path.join(app.getPath('downloads'), fileName),
        filters: [{ name: extension.toUpperCase(), extensions: [extension] }]
      })

      return result.canceled ? null : (result.filePath ?? null)
    },
    async choosePythonExecutable(owner) {
      const result = await dialog.showOpenDialog(owner, {
        properties: ['openFile'],
        filters: process.platform === 'win32' ? [{ name: 'Python 3.10–3.14', extensions: ['exe'] }] : []
      })

      return result.canceled ? null : (result.filePaths[0] ?? null)
    },
    revealFile: file => shell.showItemInFolder(file)
  })

  const manager = new PlayerManager({ mediaJobs })
  const ipc = registerPlayerIpc(ipcMain, manager)

  return {
    registerOwner(window: BrowserWindow, page: string) {
      ipc.registerOwner(window, page)
    },
    dispose() {
      ipc.dispose()
      manager.dispose()
      mediaJobs.dispose()
    }
  }
}
