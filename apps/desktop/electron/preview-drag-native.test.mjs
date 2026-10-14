import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { app, BrowserWindow } from 'electron'
import { build } from 'esbuild'

// A dedicated hidden test application, never the user's Desktop/profile.
const runtime = mkdtempSync(join(tmpdir(), 'hermes-preview-drag-'))
app.setPath('userData', join(runtime, 'user-data'))
app.setPath('sessionData', join(runtime, 'session-data'))
const desktop = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const timeout = setTimeout(() => { console.error('Native drag fixture timed out'); app.exit(1) }, 90_000)

async function run() {
  const compiled = await build({
    entryPoints: [join(desktop, 'src/app/chat/right-rail/preview-drag-native-fixture.ts')],
    absWorkingDir: desktop,
    bundle: true,
    write: false,
    platform: 'browser',
    format: 'iife',
    define: { 'import.meta.env': '{"DEV":false}', 'import.meta.hot': 'undefined' },
    alias: { '@': join(desktop, 'src') }
  })
  const window = new BrowserWindow({
    show: false,
    skipTaskbar: true,
    focusable: false,
    width: 1240,
    height: 700,
    webPreferences: { webviewTag: true, offscreen: true, backgroundThrottling: false, nodeIntegration: false, contextIsolation: true }
  })
  const host = join(runtime, 'host.html')
  writeFileSync(host, '<!doctype html><html><body style="margin:0"></body></html>')
  try {
    console.log('native fixture: compiled; loading hidden host')
    window.webContents.on('console-message', event => console.log('fixture renderer:', event.message))
    await window.loadFile(host)
    console.log('native fixture: injecting production action bundle')
    await window.webContents.executeJavaScript(compiled.outputFiles[0].text)
    console.log('native fixture: exercising guests')
    const result = await window.webContents.executeJavaScript('window.runNativeDragFixture()')
    assert.equal(result.success, true)
    assert.equal(window.isVisible(), false, 'fixture must never show a native window')
    const image = await window.webContents.capturePage()
    assert.ok(image.getSize().width >= 1200, 'both synthetic guests must fit in the captured viewport')
    const screenshot = join(runtime, 'native-drag.png')
    writeFileSync(screenshot, image.toPNG())
    console.log(JSON.stringify({ ...result, screenshot }, null, 2))
  } finally {
    window.destroy()
  }
}

app.whenReady().then(run).then(() => { clearTimeout(timeout); app.exit(0) }).catch(error => {
  console.error(error)
  clearTimeout(timeout)
  app.exit(1)
})
