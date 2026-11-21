const assert = require('node:assert/strict')
const { writeFileSync } = require('node:fs')
const { join } = require('node:path')

const { app, BrowserWindow } = require('electron')

const runtime = process.argv[2]
const pointer = require(join(runtime, 'native-pointer.node'))
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
app.setName('Hermes titlebar regression fixture')
app.setPath('userData', join(runtime, 'userData'))
app.setPath('sessionData', join(runtime, 'sessionData'))
app.setPath('crashDumps', join(runtime, 'crashes'))
app.setActivationPolicy('accessory')
const watchdog = setTimeout(() => app.exit(1), 30_000)

async function run() {
  const start = JSON.parse(pointer.desktopState())
  const window = new BrowserWindow({
    title: 'Hermes isolated titlebar regression',
    x: 80,
    y: 80,
    width: 1000,
    height: 600,
    show: false,
    acceptFirstMouse: true,
    titleBarStyle: 'hidden',
    trafficLightPosition: { x: 12, y: 10 },
    backgroundColor: '#202020',
    webPreferences: { backgroundThrottling: false, nodeIntegration: false, contextIsolation: true }
  })
  const handle = window.getNativeWindowHandle()
  const evaluate = code => window.webContents.executeJavaScript(code)
  const snapshot = () => evaluate('window.titlebarFixture.snapshot()')
  const clicks = []
  const states = []

  async function click(id, mode = 'cocoa', delivered = true) {
    const before = await snapshot()
    const button = before.buttons.find(button => button.id === id)
    assert.equal(button.domTarget, id, `DOM hit target for ${id}`)

    if (mode === 'cocoa') {
      pointer.post(handle, button.x, button.y)
    } else if (mode === 'electron-input') {
      const point = { x: Math.round(button.x), y: Math.round(button.y), button: 'left', clickCount: 1 }
      window.webContents.sendInputEvent({ ...point, type: 'mouseDown' })
      window.webContents.sendInputEvent({ ...point, type: 'mouseUp' })
    } else {
      await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`)
    }

    await sleep(160)
    const after = await snapshot()
    const delta = (after.counts[id] ?? 0) - (before.counts[id] ?? 0)
    clicks.push({ id, mode, delta, expected: Number(delivered) })
    assert.equal(
      delta,
      Number(delivered),
      `${mode} ${id} click was ${delivered ? 'lost' : 'not swallowed by drag region'}`
    )
  }

  function assertSafe(state) {
    for (const button of state.buttons.filter(button => button.id !== 'body')) {
      assert.equal(button.domTarget, button.id)
      assert.equal(
        state.drags.some(
          drag =>
            button.x >= drag.x &&
            button.x < drag.x + drag.width &&
            button.y >= drag.y &&
            button.y < drag.y + drag.height
        ),
        false,
        `native drag overlaps ${button.id}`
      )
    }

    assert.ok(
      state.panels.every(panel => panel.retained),
      'panels must move without remounting'
    )
    assert.deepEqual(Object.fromEntries(state.panels.map(panel => [panel.id, panel.width])), {
      sidebar: 250,
      main: 750
    })
  }

  try {
    window.setMenu(null)
    await window.loadFile(join(runtime, 'fixture.html'))
    window.showInactive()
    await sleep(700)
    const initialBounds = window.getBounds()
    const initial = await snapshot()
    assertSafe(initial)
    states.push(initial)

    for (const id of ['settings', 'layoutTool', 'hud', 'rhide', 'hide', 'body']) {
      await click(id)
    }

    // Qualification: the input MUST be swallowed by native drag geometry even
    // though elementFromPoint(), DOM click and sendInputEvent all bypass it.
    await evaluate('window.titlebarFixture.setNegativeControl(true)')
    await sleep(240)
    await click('hide', 'cocoa', false)
    await click('body')
    await click('hide', 'dom')
    await click('hide', 'electron-input')
    await evaluate('window.titlebarFixture.setNegativeControl(false)')
    await sleep(240)
    await click('hide')

    for (let swap = 0; swap < 10; swap++) {
      // This native action changes the same atom rendered by the REAL root.
      // There is no fixture chrome-change event, fix flag or resize workaround.
      await click('swap')
      const state = await snapshot()
      assert.deepEqual(
        state.panels.map(panel => panel.id),
        swap % 2 === 0 ? ['main', 'sidebar'] : ['sidebar', 'main']
      )
      states.push(state)

      for (const id of ['settings', 'layoutTool', 'hud', 'rhide', 'hide', 'body']) {
        await click(id)
      }

      assertSafe(state)
      assert.deepEqual(window.getBounds(), initialBounds, 'no native resize may heal the fixture')
      assert.equal(window.isMaximized(), false)
      assert.equal(window.isFullScreen(), false)
    }

    const finish = JSON.parse(pointer.desktopState())
    assert.deepEqual(finish, start, 'fixture must preserve the frontmost app and cursor')
    const result = {
      versions: process.versions,
      swaps: states.length - 1,
      successfulNativeClicks: clicks.filter(click => click.mode === 'cocoa' && click.delta === 1).length,
      negativeControl: clicks.filter(click => click.expected === 0 || click.mode !== 'cocoa'),
      focusAndCursorPreserved: true,
      start,
      finish,
      clicks,
      states
    }
    writeFileSync(join(runtime, 'results.json'), JSON.stringify(result, null, 2))
  } catch (error) {
    writeFileSync(
      join(runtime, 'failure.json'),
      JSON.stringify({ error: error.message, clicks, states, last: await snapshot() }, null, 2)
    )
    throw error
  } finally {
    window.destroy()
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
