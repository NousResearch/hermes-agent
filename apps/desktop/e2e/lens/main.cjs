const { app, BrowserWindow } = require('electron')
app.whenReady().then(() => {
  const window = new BrowserWindow({
    width: 1280,
    height: 900,
    webPreferences: {
      webviewTag: true,
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true
    }
  })
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  window.loadURL('http://127.0.0.1:5176/e2e/lens/host.html')
})
app.on('window-all-closed', () => app.quit())
