import { contextBridge, ipcRenderer } from 'electron'

// Deliberate fixture IPC only; the renderer still uses production gateway/WS code.
contextBridge.exposeInMainWorld('hermesDesktop', {
  getConnection: (profile: string) => ipcRenderer.invoke('retirement-fixture:descriptor', profile),
  getConnectionFor: ({ profile }: { profile: string }) => ipcRenderer.invoke('retirement-fixture:descriptor', profile)
})
contextBridge.exposeInMainWorld('retirementFixture', {
  onRetiring: (handler: (key: string) => void) => {
    ipcRenderer.on('retirement-fixture:retiring', (_event, key: string) => handler(key))
  }
})
