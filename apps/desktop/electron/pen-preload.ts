// Pen canvas webview preload — the bridge the pen.dev editor bundle expects
// from an Electron host. Ported from Pen.app's own out/preload.js (the editor
// keys its host detection off `window.electronAPI`, then falls back to
// `window.PENCIL_INIT_PARAMS` for boot params, which our hermes-pen:// protocol
// handler injects into the served index.html per document).
import { contextBridge, ipcRenderer, webUtils } from 'electron'

function getInitParams() {
  const arg = process.argv.find(a => a.startsWith('--init-params='))

  if (!arg) {
    return null
  }

  try {
    return JSON.parse(arg.slice('--init-params='.length))
  } catch {
    return null
  }
}

contextBridge.exposeInMainWorld('PENCIL_APP_NAME', 'Electron')
contextBridge.exposeInMainWorld('PENCIL_ARCH', process.arch)
contextBridge.exposeInMainWorld('IS_DEV', false)
contextBridge.exposeInMainWorld('electronAPI', {
  sendMessage: (message: unknown) => {
    ipcRenderer.send('ipc-message', message)
  },
  onMessageReceived: (callback: (message: unknown) => void) => {
    ipcRenderer.on('ipc-message', (_event, message) => {
      callback(message)
    })
  },
  resolveFilePath: (file: File) => {
    return webUtils.getPathForFile(file)
  },
  initParams: getInitParams(),
  newFilePickerData: null
})
