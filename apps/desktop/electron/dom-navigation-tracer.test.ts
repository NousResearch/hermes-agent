import assert from 'node:assert/strict'

import { test } from 'vitest'

import { formatDomNavigationLine, traceWindowDomNavigation } from './dom-navigation-tracer'

// Fake Electron surface — real listener wiring, no Electron import. Mirrors
// how the rest of electron/*.test.ts exercises Electron-free modules.
function makeFakeWindow(overrides: { destroyed?: boolean } = {}) {
  const listeners = new Map<string, ((...args: any[]) => void)[]>()
  let destroyed = overrides.destroyed ?? false

  const win = {
    isDestroyed: () => destroyed,
    setDestroyed: (value: boolean) => {
      destroyed = value
    },
    webContents: {
      on: (event: string, listener: (...args: any[]) => void) => {
        const list = listeners.get(event) ?? []

        list.push(listener)
        listeners.set(event, list)
      },
      removeListener: (event: string, listener: (...args: any[]) => void) => {
        listeners.set(
          event,
          (listeners.get(event) ?? []).filter(candidate => candidate !== listener)
        )
      },
      emit: (event: string, ...args: unknown[]) => {
        for (const listener of listeners.get(event) ?? []) {
          listener(...args)
        }
      },
      listenerCount: (event: string) => (listeners.get(event) ?? []).length
    }
  }

  return win
}

function makeTracer(win: ReturnType<typeof makeFakeWindow>, label = 'main') {
  const logs: string[] = []

  const dispose = traceWindowDomNavigation(win as never, label, {
    log: (line: string) => {
      logs.push(line)
    }
  })

  return { logs, dispose }
}

test('logs a did-start-navigation line with url, main-frame and in-place fields', () => {
  const win = makeFakeWindow()
  const { logs } = makeTracer(win)

  win.webContents.emit('did-start-navigation', {}, 'file:///app/index.html', false, true)

  assert.deepEqual(logs, ['[domnav:main] did-start-navigation url=file:///app/index.html main=true inPlace=false'])
})

test('logs did-navigate, in-page navigation and did-finish-load lines', () => {
  const win = makeFakeWindow()
  const { logs } = makeTracer(win, 'session-window')

  win.webContents.emit('did-navigate', {}, 'file:///app/index.html')
  win.webContents.emit('did-navigate-in-page', {}, 'file:///app/index.html#/chat', true)
  win.webContents.emit('did-finish-load', {})

  assert.deepEqual(logs, [
    '[domnav:session-window] did-navigate url=file:///app/index.html',
    '[domnav:session-window] did-navigate-in-page url=file:///app/index.html#/chat main=true',
    '[domnav:session-window] did-finish-load'
  ])
})

test('formats did-fail-load with code, description, url and main-frame', () => {
  const win = makeFakeWindow()
  const { logs } = makeTracer(win)

  win.webContents.emit('did-fail-load', {}, -3, 'ERR_ABORTED', 'file:///app/index.html', true)

  assert.deepEqual(logs, ['[domnav:main] did-fail-load code=-3 desc=ERR_ABORTED url=file:///app/index.html main=true'])
})

test('renders missing did-fail-load fields as ? instead of undefined', () => {
  assert.equal(
    formatDomNavigationLine('hud', 'did-fail-load', ['code=?', 'desc=?', 'url=?', 'main=?']),
    '[domnav:hud] did-fail-load code=? desc=? url=? main=?'
  )
})

test('logs render-process-gone with reason and exit code', () => {
  const win = makeFakeWindow()
  const { logs } = makeTracer(win)

  win.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 1 })

  assert.deepEqual(logs, ['[domnav:main] render-process-gone reason=crashed exitCode=1'])
})

test('emits nothing while the window is destroyed (teardown noise stays out of the log)', () => {
  const win = makeFakeWindow({ destroyed: true })
  const { logs } = makeTracer(win)

  win.webContents.emit('did-start-navigation', {}, 'file:///app/index.html', false, true)
  win.webContents.emit('did-finish-load', {})
  win.webContents.emit('render-process-gone', {}, { reason: 'killed', exitCode: 0 })

  assert.deepEqual(logs, [])
})

test('dispose removes every listener so window recreation cannot stack handlers', () => {
  const win = makeFakeWindow()
  const { dispose } = makeTracer(win)

  const events = [
    'did-start-navigation',
    'did-navigate',
    'did-navigate-in-page',
    'did-finish-load',
    'did-fail-load',
    'render-process-gone'
  ]

  for (const event of events) {
    assert.equal(win.webContents.listenerCount(event), 1)
  }

  dispose()

  for (const event of events) {
    assert.equal(win.webContents.listenerCount(event), 0)
  }
})
