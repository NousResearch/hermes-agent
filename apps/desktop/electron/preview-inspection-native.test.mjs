import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { app, BrowserWindow } from 'electron'
import { build } from 'esbuild'

// A standalone hidden fixture: no Desktop backend, persistent browser partition,
// credentials, CDP, or OS input. Capture includes the entire fixture viewport.
const runtimeDir = mkdtempSync(join(tmpdir(), 'hermes-preview-inspection-'))
app.setPath('userData', runtimeDir)
app.setPath('sessionData', runtimeDir)
const here = dirname(fileURLToPath(import.meta.url))

async function run() {
  const bundle = await build({
    entryPoints: [join(here, '../src/lib/preview-act/inspect-in-page.ts')],
    bundle: true,
    write: false,
    format: 'iife',
    globalName: 'inspectionFixture',
    target: 'es2022'
  })
  const window = new BrowserWindow({
    show: false,
    focusable: false,
    skipTaskbar: true,
    width: 900,
    height: 700,
    webPreferences: {
      offscreen: true,
      backgroundThrottling: false,
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  try {
    const fixture = join(runtimeDir, 'fixture.html')
    writeFileSync(
      fixture,
      `<!doctype html><meta charset="utf-8"><title>Preview inspection fixture</title>
      <style>
        body { margin: 0; font: 16px sans-serif; background: #eee; }
        h1, p { margin-left: 24px; }
        #box { position: absolute; left: 60px; top: 140px; width: 200px; height: 80px; background: #acf; }
        #handle { position: absolute; right: -10px; bottom: -10px; width: 20px; height: 20px; }
        #overlay { position: absolute; left: 245px; top: 205px; width: 180px; height: 36px; z-index: 2; background: #fa8; }
        svg { position: absolute; left: 60px; top: 300px; width: 300px; height: 100px; }
        #secret { position: absolute; left: 60px; top: 430px; }
      </style>
      <h1>Read-only geometry and hit testing</h1><p>HTML handle, SVG handles, and a covering toolbar</p>
      <div id="box"><button id="handle" aria-label="Resize">↘</button></div>
      <div id="overlay" hidden>Covering toolbar</div>
      <svg><circle id="first" cx="50" cy="50" r="12" fill="#07b"/><circle id="second" cx="100" cy="50" r="12" fill="#b70"/></svg>
      <input id="secret" value="SYNTHETIC_PRIVATE_VALUE" title="SYNTHETIC_PRIVATE_TITLE">
      <script>
        window.inputLedger = [];
        for (const kind of ['pointerdown','pointermove','pointerup','keydown','input']) {
          document.addEventListener(kind, event => inputLedger.push({kind, trusted: event.isTrusted, buttons: event.buttons}));
        }
      </script>`
    )
    await window.loadFile(fixture)
    await window.webContents.executeJavaScript(bundle.outputFiles[0].text)
    // Serialize the compiled function, just like the production dispatcher.
    const source = await window.webContents.executeJavaScript('inspectionFixture.inspectTargetInPage.toString()')
    const inspect = async action =>
      JSON.parse(
        await window.webContents.executeJavaScript(
          `JSON.stringify((${source})(document, window.__hermesActHolder, ${JSON.stringify(action)}))`
        )
      )
    const readState = () =>
      window.webContents.executeJavaScript(`JSON.stringify({
      html: document.body.innerHTML, focus: document.activeElement.id,
      x: scrollX, y: scrollY, holder: Object.hasOwn(window, '__hermesActHolder'), events: inputLedger
    })`)
    const observations = []

    for (const zoom of [1, 1.25]) {
      window.webContents.setZoomFactor(zoom)
      await window.webContents.executeJavaScript('new Promise(requestAnimationFrame)')
      await window.webContents.executeJavaScript('document.getElementById("secret").focus()')
      const before = await readState()
      const html = await inspect({ selector: '#handle' })
      assert.equal(html.success, true)
      assert.equal(html.inspection.candidates[0].hit.relationship, 'self')
      assert.deepEqual(html.inspection.candidates[0].point, { x: 260, y: 220 })
      const svg = await inspect({ selector: 'svg circle:first-child' })
      assert.equal(svg.inspection.candidates[0].node.tag, 'circle')
      assert.equal(svg.inspection.candidates[0].hit.relationship, 'self')
      assert.equal(await readState(), before, 'inspection must not alter DOM, focus, scroll, holder or input')
      assert.ok(!JSON.stringify([html, svg]).includes('SYNTHETIC_PRIVATE'))

      await window.webContents.executeJavaScript('document.getElementById("overlay").hidden = false')
      const covered = await inspect({ selector: '#handle' })
      assert.equal(covered.inspection.candidates[0].hit.node.id, 'overlay')
      assert.equal(covered.inspection.candidates[0].hit.relationship, 'unrelated')
      await window.webContents.executeJavaScript(
        'document.getElementById("overlay").hidden = true; document.getElementById("second").setAttribute("cx", "54")'
      )
      const overlap = await inspect({ selector: '#first' })
      assert.equal(overlap.inspection.candidates[0].hit.node.id, 'second')
      assert.equal(overlap.inspection.candidates[0].hit.relationship, 'unrelated')
      await window.webContents.executeJavaScript('document.getElementById("second").setAttribute("cx", "100")')
      observations.push({ zoom, center: html.inspection.candidates[0].point, covered: 'overlay', overlap: 'second' })
    }

    assert.deepEqual((await inspect({ selector: '.absent' })).inspection.candidates, [])
    assert.equal((await inspect({ selector: '[' })).success, false)
    assert.equal((await inspect({ ref: 'unknown', selector: '#handle' })).success, false)
    assert.equal(await window.webContents.executeJavaScript('inputLedger.length'), 0)
    await window.webContents.executeJavaScript('document.getElementById("overlay").hidden = false')
    const screenshot = await window.webContents.capturePage()
    assert.ok(screenshot.getSize().width >= 800, 'capture must not clip the fixture width')
    writeFileSync(join(runtimeDir, 'inspection.png'), screenshot.toPNG())
    console.log(
      JSON.stringify({
        passed: true,
        platform: process.platform,
        electron: process.versions.electron,
        observations,
        inputEvents: 0,
        screenshot: join(runtimeDir, 'inspection.png')
      })
    )
  } finally {
    window.destroy()
  }
}

app
  .whenReady()
  .then(run)
  .then(() => app.exit(0))
  .catch(error => {
    console.error(error)
    app.exit(1)
  })
