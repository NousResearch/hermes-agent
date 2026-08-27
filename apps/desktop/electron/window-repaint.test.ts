import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createWindowRepaintController, type RepaintWindowBounds, type RepaintWindowLike } from './window-repaint'

function makeTimers() {
  const pending = new Map<number, () => void>()
  let nextId = 1

  return {
    clearTimeout: (handle: unknown) => {
      pending.delete(handle as number)
    },
    fire() {
      const jobs = [...pending.values()]
      pending.clear()

      for (const job of jobs) {
        job()
      }
    },
    get pendingCount() {
      return pending.size
    },
    setTimeout: (fn: () => void, _ms: number) => {
      const id = nextId++
      pending.set(id, fn)

      return id
    }
  }
}

interface WindowOverrides {
  destroyed?: boolean
  visible?: boolean
  maximized?: boolean
  fullscreen?: boolean
}

function makeWindow(overrides: WindowOverrides = {}) {
  const listeners = new Map<string, () => void>()
  const setBoundsLog: RepaintWindowBounds[] = []
  const getBoundsLog: number[] = []

  let bounds: RepaintWindowBounds = { x: 10, y: 20, width: 800, height: 600 }
  let destroyed = overrides.destroyed ?? false
  let visible = overrides.visible ?? true
  let maximized = overrides.maximized ?? false
  let fullscreen = overrides.fullscreen ?? false

  const win = {
    setBoundsLog,
    getBoundsLog,
    bounds() {
      return { ...bounds }
    },
    close() {
      destroyed = true
      listeners.get('closed')?.()
    },
    emit(event: string) {
      listeners.get(event)?.()
    },
    getBounds() {
      getBoundsLog.push(1)

      return { ...bounds }
    },
    isDestroyed: () => destroyed,
    isFullScreen: () => fullscreen,
    isMaximized: () => maximized,
    isVisible: () => visible,
    on(event: string, fn: () => void) {
      listeners.set(event, fn)
    },
    setBounds(next: RepaintWindowBounds) {
      setBoundsLog.push({ ...next })
      bounds = { ...next }
    }
  }

  return win as unknown as RepaintWindowLike & {
    bounds: () => RepaintWindowBounds
    close: () => void
    emit: (event: string) => void
    getBoundsLog: number[]
    setBoundsLog: RepaintWindowBounds[]
  }
}

test('show on a registered window nudges +2 DIPs then restores', () => {
  const timers = makeTimers()
  const win = makeWindow()
  const controller = createWindowRepaintController(timers as never)

  controller.register(win)

  win.emit('show')

  assert.equal(win.setBoundsLog.length, 1)
  assert.deepEqual(win.setBoundsLog[0], { x: 10, y: 20, width: 802, height: 600 })

  timers.fire()

  assert.equal(win.setBoundsLog.length, 2)
  assert.deepEqual(win.setBoundsLog[1], { x: 10, y: 20, width: 800, height: 600 })
})

test('restore arriving mid-nudge does not double-nudge', () => {
  const timers = makeTimers()
  const win = makeWindow()
  const controller = createWindowRepaintController(timers as never)

  controller.register(win)

  win.emit('show')
  win.emit('restore')

  // Only ONE nudge: the second reveal finds a restore already pending and
  // skips, so the restore target can never be the nudged size.
  assert.equal(win.setBoundsLog.length, 1)

  timers.fire()

  assert.equal(win.setBoundsLog.length, 2)
  assert.deepEqual(win.setBoundsLog[1], { x: 10, y: 20, width: 800, height: 600 })
})

test('restore alone nudges like show', () => {
  const timers = makeTimers()
  const win = makeWindow()
  const controller = createWindowRepaintController(timers as never)

  controller.register(win)

  win.emit('restore')
  assert.equal(win.setBoundsLog.length, 1)

  timers.fire()
  assert.equal(win.setBoundsLog.length, 2)
})

test('hidden windows are not nudged', () => {
  const timers = makeTimers()
  const win = makeWindow({ visible: false })
  const controller = createWindowRepaintController(timers as never)

  controller.register(win)
  win.emit('show')

  assert.equal(win.setBoundsLog.length, 0)
  assert.equal(timers.pendingCount, 0)
})

test('maximized and fullscreen windows are not nudged', () => {
  for (const overrides of [{ maximized: true }, { fullscreen: true }]) {
    const timers = makeTimers()
    const win = makeWindow(overrides)
    const controller = createWindowRepaintController(timers as never)

    controller.register(win)
    win.emit('show')

    assert.equal(win.setBoundsLog.length, 0, JSON.stringify(overrides))
    assert.equal(timers.pendingCount, 0)
  }
})

test('destroyed windows are not nudged and close unregisters', () => {
  const timers = makeTimers()
  const win = makeWindow()
  const controller = createWindowRepaintController(timers as never)

  controller.register(win)
  win.close()

  win.emit('show')
  controller.kickAll()

  assert.equal(win.setBoundsLog.length, 0)
  assert.equal(timers.pendingCount, 0)
})

test('a window destroyed mid-nudge skips the restore', () => {
  const timers = makeTimers()
  const win = makeWindow()
  const controller = createWindowRepaintController(timers as never)

  controller.register(win)
  win.emit('show')
  assert.equal(win.setBoundsLog.length, 1)

  win.close()
  timers.fire()

  // Only the nudge happened; the restore was skipped for the destroyed window.
  assert.equal(win.setBoundsLog.length, 1)
})

test('kick nudges a tracked window immediately', () => {
  const timers = makeTimers()
  const win = makeWindow()
  const controller = createWindowRepaintController(timers as never)

  controller.register(win)
  controller.kick(win)

  assert.equal(win.setBoundsLog.length, 1)

  timers.fire()
  assert.equal(win.setBoundsLog.length, 2)
})

test('kickAll nudges every tracked window', () => {
  const timers = makeTimers()
  const a = makeWindow()
  const b = makeWindow()
  const controller = createWindowRepaintController(timers as never)

  controller.register(a)
  controller.register(b)
  controller.kickAll()

  assert.equal(a.setBoundsLog.length, 1)
  assert.equal(b.setBoundsLog.length, 1)

  timers.fire()

  assert.equal(a.setBoundsLog.length, 2)
  assert.equal(b.setBoundsLog.length, 2)
})

test('a getBounds throw is swallowed and schedules nothing', () => {
  const timers = makeTimers()
  const win = makeWindow()
  const controller = createWindowRepaintController(timers as never)

  win.getBounds = () => {
    throw new Error('boom')
  }

  controller.register(win)
  win.emit('show')

  assert.equal(win.setBoundsLog.length, 0)
  assert.equal(timers.pendingCount, 0)
})

test('a setBounds throw during nudge cancels the pending restore', () => {
  const timers = makeTimers()
  const win = makeWindow()
  const controller = createWindowRepaintController(timers as never)

  win.setBounds = () => {
    throw new Error('boom')
  }

  controller.register(win)
  win.emit('show')

  assert.equal(win.setBoundsLog.length, 0)
  assert.equal(timers.pendingCount, 0)
})
