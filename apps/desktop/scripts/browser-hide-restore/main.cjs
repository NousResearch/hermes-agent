const { app, BrowserWindow } = require('electron')
app.setPath('userData', process.env.HIDE_RESTORE_USER_DATA)
app.whenReady().then(() => {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
    webPreferences: { webviewTag: true, contextIsolation: true, nodeIntegration: false, sandbox: true }
  })
  win.loadURL(process.env.HIDE_RESTORE_FIXTURE_URL)
})
app.on('window-all-closed', () => app.quit())
