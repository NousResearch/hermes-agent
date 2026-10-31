import assert from 'node:assert/strict'
import { once } from 'node:events'
import { mkdtempSync } from 'node:fs'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { app, BrowserWindow, session } from 'electron'

// Real Electron, not a mocked frame or a debugger with focus emulation.
const runtimeDir = mkdtempSync(join(tmpdir(), 'hermes-preview-activity-'))
app.setPath('userData', runtimeDir)
app.setPath('sessionData', runtimeDir)
const watchdog = setTimeout(() => {
  console.error('Native preview activity fixture exceeded 30 seconds')
  app.exit(1)
}, 30_000)

async function bounded(promise) {
  let timeout

  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new Error('Hidden preview did not finish its operation')), 5_000)
      })
    ])
  } finally {
    clearTimeout(timeout)
  }
}

async function run() {
  const { beginPreviewGuestActivity, endPreviewGuestActivity, executePreviewGuestScript } =
    await import('./preview-guest-throttle.ts')
  const { createStreamThrottle } = await import('./stream-throttle.ts')
  let response
  let requested = Promise.withResolvers()
  const html = '<!doctype html><title>Preview activity</title><input id="name" value="retained">'
  const server = createServer((request, reply) => {
    const send = () => {
      reply.writeHead(200, { 'content-type': 'text/html', 'cache-control': 'no-store' })
      reply.end(html)
    }

    if (request.url.startsWith('/slow')) {
      response = send
      requested.resolve()
    } else {
      send()
    }
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const base = `http://127.0.0.1:${server.address().port}`
  const window = new BrowserWindow({
    show: true,
    width: 640,
    height: 480,
    webPreferences: { webviewTag: true, backgroundThrottling: true, contextIsolation: true, nodeIntegration: false }
  })
  const throttle = createStreamThrottle()
  throttle.register(window)

  try {
    const attached = once(window.webContents, 'did-attach-webview')
    await window.loadURL(
      'data:text/html,' +
        encodeURIComponent(
          `<webview partition="persist:hermes-preview" src="${base}/old" style="width:600px;height:400px"></webview>`
        )
    )
    const [, guest] = await attached

    if (!guest.getURL() || guest.isLoadingMainFrame()) {
      await once(guest, 'did-stop-loading')
    }

    const begin = () =>
      beginPreviewGuestActivity(window.webContents, guest, session.fromPartition('persist:hermes-preview'), () =>
        throttle.holdWindow(window)
      )
    const execute = code => executePreviewGuestScript(window.webContents, begin(), code)
    const assertIdle = () => {
      assert.equal(window.isVisible(), false)
      assert.equal(window.webContents.getBackgroundThrottling(), true)
      assert.equal(guest.getBackgroundThrottling(), true)
    }
    const frame = 'new Promise(resolve => requestAnimationFrame(() => resolve(location.pathname)))'
    window.hide()

    // A queued script belongs to the incoming document, not the document whose
    // navigation commits before Electron would normally dispatch the script.
    const navigation = guest.loadURL(base + '/slow-first')
    await requested.promise
    const queued = execute(frame)
    response()
    await navigation
    assert.equal(await bounded(queued), '/slow-first')
    assertIdle()

    // A genuinely abandoned document must release its own lease, without
    // cancelling or sleeping another operation waiting for the next document.
    const old = execute('window.started = true; new Promise(() => {})')
    const abandoned = assert.rejects(old, Error)
    await bounded(
      (async () => {
        while (!(await guest.executeJavaScript('Boolean(window.started)'))) {
          await new Promise(resolve => setTimeout(resolve, 10))
        }
      })()
    )
    requested = Promise.withResolvers()
    const replacement = guest.loadURL(base + '/slow-second')
    await requested.promise
    const next = execute(frame)
    response()
    await replacement
    await bounded(abandoned)
    assert.equal(await bounded(next), '/slow-second')
    assertIdle()

    // The plugin API's completion values must keep structured-clone semantics,
    // not the JSON-like conversion performed by WebFrameMain.executeJavaScript.
    const values = await bounded(
      execute(`(() => {
      const cycle = {}; cycle.self = cycle;
      return { missing: undefined, nan: NaN, date: new Date(0), bytes: new Uint8Array([1, 2, 3]),
        map: new Map([['x', 1]]), bigint: 123n, cycle };
    })()`)
    )
    assert.equal(values.missing, undefined)
    assert(Number.isNaN(values.nan))
    assert.deepEqual(values.date, new Date(0))
    assert.deepEqual(values.bytes, new Uint8Array([1, 2, 3]))
    assert.deepEqual(values.map, new Map([['x', 1]]))
    assert.equal(values.bigint, 123n)
    assert.equal(values.cycle.self, values.cycle)
    await assert.rejects(execute('throw new Error("fixture error")'), Error)
    assertIdle()

    // Cancelling while loading prevents dispatch even after that load finishes.
    requested = Promise.withResolvers()
    const cancelledNavigation = guest.loadURL(base + '/slow-cancelled')
    await requested.promise
    const token = begin()
    const cancelled = assert.rejects(
      executePreviewGuestScript(window.webContents, token, 'window.mustNotRun = true'),
      Error
    )
    endPreviewGuestActivity(window.webContents, token)
    await bounded(cancelled)
    response()
    await cancelledNavigation
    assert.equal(await guest.executeJavaScript('Boolean(window.mustNotRun)'), false)
    assertIdle()
    console.log('Native preview navigation, overlap, cancellation and completion-value contracts passed')
  } finally {
    window.destroy()
    server.closeAllConnections()
    server.close()
  }
}

app
  .whenReady()
  .then(run)
  .then(() => {
    clearTimeout(watchdog)
    app.exit(0)
  })
  .catch(error => {
    console.error(error)
    app.exit(1)
  })
