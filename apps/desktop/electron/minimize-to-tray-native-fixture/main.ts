// Targeted native lifecycle probe: compile this fixture with esbuild (external:
// electron, dbus-native), then run Electron <fixture.cjs> <fresh userData> <icon>.
// No backend, renderer dev server, CDP listener, or real user state is touched.
import assert from 'node:assert/strict'
import { once } from 'node:events'
import path from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'

import { app, BrowserWindow } from 'electron'

import { createMinimizeToTray } from '../minimize-to-tray'

app.setPath('userData', process.argv[2])
app.on('window-all-closed', () => {})

const watchdog = setTimeout(() => {
  console.error('Native tray probe timed out')
  app.exit(1)
}, 20_000)

async function run() {
  await app.whenReady()
  const primary = new BrowserWindow({ width: 440, height: 320, show: false })
  const peer = new BrowserWindow({ width: 440, height: 320, show: false })

  const controller = createMinimizeToTray({
    preferencesPath: path.join(app.getPath('userData'), 'minimize-to-tray.json'),
    getIconPath: () => process.argv[3],
    restoreMainWindow: () => {
      primary.show()
      primary.focus()
    },
    isQuittingForHandoff: () => false,
    log: console.log
  })

  controller.registerWindow(primary)
  controller.registerWindow(peer)

  const html =
    'data:text/html,<title>Hermes tray lifecycle probe</title><p>Isolated tray lifecycle probe</p><script>window.draft="preserved"</script>'

  await Promise.all([primary.loadURL(html), peer.loadURL(html)])
  primary.show()
  peer.show()
  assert.deepEqual(await controller.start(), { enabled: false, available: false })
  assert.deepEqual(await controller.setEnabled(true), { enabled: true, available: true })
  const primaryMinimized = once(primary, 'minimize')
  primary.minimize()
  await primaryMinimized
  await delay(250)
  assert.equal(primary.isDestroyed(), false)
  assert.equal(primary.isVisible(), false)
  assert.equal(peer.isVisible(), true)

  if (process.platform === 'darwin') {
    assert.equal(app.dock!.isVisible(), true)
  }

  const minimized = once(peer, 'minimize')
  peer.minimize()
  await minimized
  await delay(250)
  assert.equal(peer.isVisible(), false)

  if (process.platform === 'darwin') {
    assert.equal(app.dock!.isVisible(), false)
  }

  assert.equal(await primary.webContents.executeJavaScript('window.draft'), 'preserved')
  controller.restore()
  await delay(250)
  assert.equal(primary.isVisible(), true)
  assert.equal(peer.isVisible(), true)
  assert.equal(peer.isMinimized(), false)

  if (process.platform === 'darwin') {
    assert.equal(app.dock!.isVisible(), true)
  }

  primary.minimize()
  await delay(250)
  peer.close()
  assert.equal(peer.isDestroyed(), true)
  assert.equal(primary.isDestroyed(), false)
  await controller.setEnabled(false)
  assert.equal(primary.isVisible(), true)
  assert.equal(controller.status().available, false)
  await controller.setEnabled(true)
  primary.minimize()
  await delay(250)
  app.on('before-quit', () => controller.beginQuit())
  app.on('will-quit', () => {
    assert.equal(primary.isDestroyed(), true)
    clearTimeout(watchdog)
    console.log(
      JSON.stringify({
        ok: true,
        platform: process.platform,
        checks: [
          'default off',
          'native tray creation',
          'primary minimize preserves renderer',
          'visible peer retains Dock',
          'native minimize hides peer',
          'last hidden window removes Dock',
          'renderer draft survives',
          'restore reveals all tray-hidden windows and Dock',
          'peer close remains close',
          'disable restores before destroying tray',
          'explicit quit exits'
        ]
      })
    )
  })
  app.quit()
}

void run().catch(error => {
  clearTimeout(watchdog)
  console.error(error)
  app.exit(1)
})
