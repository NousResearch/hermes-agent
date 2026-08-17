import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// Extract the clock functions only — the plugin is an ES module, so the whole
// file can't be evaluated in a bare vm context. The slice is anchored on the
// function declarations themselves.
const start = pluginSource.indexOf('function startFaceClock()')
const end = pluginSource.indexOf('/**\n * Live math face.')
assert.ok(start > 0 && end > start, 'face clock functions present in plugin.js')
const clockSource = pluginSource.slice(start, end)

function load({ faces = [], intersectionObserver = true } = {}) {
  const rafQueue = new Map()
  let rafSeq = 0
  let observerInstance = null

  class FakeIntersectionObserver {
    constructor(callback) {
      this.callback = callback
      this.observed = new Set()
      this.disconnected = false
      observerInstance = this
    }

    observe(el) {
      this.observed.add(el)
    }

    unobserve(el) {
      this.observed.delete(el)
    }

    disconnect() {
      this.disconnected = true
      this.observed.clear()
    }

    emit(entries) {
      this.callback(entries)
    }
  }

  const windowObj = {
    requestAnimationFrame: cb => {
      rafSeq += 1
      rafQueue.set(rafSeq, cb)
      return rafSeq
    },
    cancelAnimationFrame: id => {
      rafQueue.delete(id)
    }
  }

  const context = {
    window: windowObj,
    document: { hidden: false },
    performance: { now: () => 0 },
    IntersectionObserver: intersectionObserver ? FakeIntersectionObserver : undefined,
    walkMathFaces: () => [...faces],
    paintMathFace: () => undefined
  }
  vm.createContext(context)
  vm.runInContext(`${clockSource}; __start = startFaceClock; __stop = stopFaceClock`, context)

  let now = 0
  const pump = (advance = 100) => {
    now += advance
    const pending = [...rafQueue.entries()]
    rafQueue.clear()
    for (const [, cb] of pending) {
      cb(now)
    }
  }

  return {
    start: () => context.__start(),
    stop: () => context.__stop(),
    pump,
    pending: () => rafQueue.size,
    observer: () => observerInstance,
    clock: () => context.window.__hbFaceClock,
    setFaces: next => {
      faces.length = 0
      faces.push(...next)
    }
  }
}

test('unit: the clock parks itself when no faces are mounted', () => {
  const h = load({ faces: [] })
  h.start()
  assert.equal(h.pending(), 1, 'clock schedules its first frame')
  h.pump()
  assert.equal(h.pending(), 0, 'no faces -> dormant, no further frames scheduled')
})

test('unit: a mounting face (startFaceClock re-entry) wakes a parked clock', () => {
  const h = load({ faces: [] })
  h.start()
  h.pump()
  assert.equal(h.pending(), 0, 'parked')

  const face = { isConnected: true }
  h.setFaces([face])
  h.start() // BotFace render path
  assert.equal(h.pending(), 1, 'wake scheduled a frame')
})

test('unit: the clock parks when all faces scroll out and the observer wakes it back', () => {
  const face = { isConnected: true }
  const h = load({ faces: [face] })
  h.start()
  h.pump()
  // Face observed but not yet visible -> visibleFaces empty -> parked.
  assert.equal(h.pending(), 0, 'no visible faces -> dormant')

  h.observer().emit([{ target: face, isIntersecting: true }])
  assert.equal(h.pending(), 1, 'visibility wakes the clock')
  h.pump()
  assert.equal(h.pending(), 1, 'visible face keeps the clock running')

  h.observer().emit([{ target: face, isIntersecting: false }])
  h.pump()
  h.pump()
  assert.equal(h.pending(), 0, 'all faces hidden -> parked again')
})

test('unit: stopFaceClock cancels the frame, disconnects the observer, and clears the handle', () => {
  const face = { isConnected: true }
  const h = load({ faces: [face] })
  h.start()
  h.observer().emit([{ target: face, isIntersecting: true }])
  h.pump()
  assert.equal(h.pending(), 1, 'running')

  h.stop()
  assert.equal(h.pending(), 0, 'pending frame cancelled')
  assert.equal(h.observer().disconnected, true, 'observer disconnected')
  assert.equal(h.clock(), undefined, 'window handle removed')

  // A fresh start after teardown re-initializes rather than hitting a dead handle.
  h.start()
  assert.equal(h.pending(), 1, 'clock restarts cleanly after stop')
})

test('source: plugin registers face-clock teardown via ctx.onDispose', () => {
  assert.match(pluginSource, /ctx\.onDispose\(stopFaceClock\)/)
})
