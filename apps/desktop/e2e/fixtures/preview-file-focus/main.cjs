const { app, BrowserWindow, Menu } = require('electron')

let window

app.whenReady().then(() => {
  // Match Desktop's native Edit roles: these exercise actual OS shortcuts.
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([{ label: 'Edit', submenu: [{ role: 'copy' }, { role: 'selectAll' }] }])
  )
  window = new BrowserWindow({ width: 1100, height: 700 })
  window.loadURL(process.argv.find(arg => arg.startsWith('http://127.0.0.1:')))
})
app.on('window-all-closed', () => app.quit())
