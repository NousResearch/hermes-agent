export interface DesktopConnectionNotificationsDeps {
  BrowserWindow: any
  getMainWindow: () => any
}

export function createDesktopConnectionNotifications(deps: DesktopConnectionNotificationsDeps) {
  const { BrowserWindow, getMainWindow } = deps

  function sendConnectionApplied() {
    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    const { webContents } = mainWindow

    if (!webContents || webContents.isDestroyed()) {
      return
    }

    webContents.send('hermes:connection:applied')
  }

  // Registry lifecycle push: a connection was removed or materially edited, so
  // every window must tear down (and, for edits, re-dial) its secondary sockets
  // scoped to that connection. Without this, a removed remote/cloud source keeps
  // its renderer WebSocket open and streaming as a ghost, and an edited one
  // keeps talking to the OLD endpoint until idle-reap.
  function broadcastConnectionsChanged(payload: { connectionId: string; reason: 'removed' | 'saved' | 'updated' }) {
    for (const win of BrowserWindow.getAllWindows()) {
      const { webContents } = win

      if (webContents && !webContents.isDestroyed()) {
        webContents.send('hermes:connections:changed', payload)
      }
    }
  }

  return { sendConnectionApplied, broadcastConnectionsChanged }
}
