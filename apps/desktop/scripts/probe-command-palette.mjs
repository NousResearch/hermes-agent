// ⌘K open latency, measured in-page (no CDP round-trip in the number).
//
//   node scripts/probe-command-palette.mjs [--port 9222] [--rounds 8]
//
// Reports, per round, the time from the keydown the app actually receives to:
//   frame_ms  — the dialog frame + input in the DOM and painted (what "instant"
//               means: the overlay owes you a frame immediately)
//   rows_ms   — the row list painted (may lag frame_ms; rows are deferred)
// plus any long tasks in the window, so a slow open is attributable.
import { CDP, sleep } from './perf/lib/cdp.mjs'

const args = process.argv.slice(2)
const flag = name => {
  const i = args.indexOf(`--${name}`)

  return i >= 0 ? args[i + 1] : undefined
}

const port = Number(flag('port') ?? 9222)
const rounds = Number(flag('rounds') ?? 8)

const cdp = await CDP.connect({ port })

await cdp.send('Runtime.enable')

const INSTALL = `
  (() => {
    if (window.__CMDK__) window.__CMDK__.stop()

    const state = { t0: null, frame: null, rows: 0, rowsAt: null, tasks: [], armed: false }

    // Time from the keydown the APP receives — excludes CDP transport, so the
    // number is what a user's finger actually experiences.
    const onKey = e => {
      if (state.armed && (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        state.t0 = performance.now()
        state.armed = false
      }
    }

    window.addEventListener('keydown', onKey, true)

    const obs = new MutationObserver(() => {
      if (state.t0 === null) return
      if (state.frame === null && document.querySelector('[cmdk-input]')) {
        state.frame = performance.now() - state.t0
      }
      const n = document.querySelectorAll('[cmdk-item]').length
      if (n > state.rows) { state.rows = n; state.rowsAt = performance.now() - state.t0 }
    })

    obs.observe(document.body, { childList: true, subtree: true })

    const po = new PerformanceObserver(list => {
      for (const e of list.getEntries()) state.tasks.push({ start: e.startTime, dur: Math.round(e.duration) })
    })

    try { po.observe({ entryTypes: ['longtask'] }) } catch {}

    window.__CMDK__ = {
      arm: () => { state.t0 = null; state.frame = null; state.rows = 0; state.rowsAt = null; state.tasks = []; state.armed = true },
      read: () => ({
        frame_ms: state.frame === null ? -1 : Math.round(state.frame),
        rows_ms: state.rowsAt === null ? -1 : Math.round(state.rowsAt),
        rows: state.rows,
        longtask_ms: state.t0 === null ? 0 : state.tasks.filter(t => t.start >= state.t0).reduce((s, t) => s + t.dur, 0)
      }),
      stop: () => { window.removeEventListener('keydown', onKey, true); obs.disconnect(); po.disconnect() }
    }

    return true
  })()
`

// Settle: frame painted AND rows stopped growing for two frames.
const WAIT = `
  new Promise(resolve => {
    let stable = 0
    let last = -1
    const started = performance.now()
    const tick = () => {
      const r = window.__CMDK__.read()
      if (r.frame_ms >= 0 && r.rows === last && r.rows > 0) {
        if (++stable >= 2) { resolve(r); return }
      } else { stable = 0 }
      last = r.rows
      if (performance.now() - started > 8000) { resolve(window.__CMDK__.read()); return }
      requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  })
`

const key = async type =>
  cdp.send('Input.dispatchKeyEvent', {
    type,
    key: 'k',
    code: 'KeyK',
    windowsVirtualKeyCode: 75,
    nativeVirtualKeyCode: 75,
    modifiers: 4
  })

const esc = async () => {
  for (const type of ['keyDown', 'keyUp']) {
    await cdp.send('Input.dispatchKeyEvent', { type, key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 })
  }

  await sleep(400)
}

await cdp.eval(INSTALL)
await esc()

const samples = []

for (let i = 0; i < rounds; i++) {
  await sleep(250)
  await cdp.eval('window.__CMDK__.arm()')
  await key('rawKeyDown')
  await key('keyUp')
  const r = await cdp.eval(WAIT)
  samples.push(r)
  console.log(`round ${i}:`, r)
  await esc()
}

await cdp.eval('window.__CMDK__.stop()')

const stat = k => {
  const v = samples.map(s => s[k]).filter(n => n >= 0).sort((a, b) => a - b)

  if (!v.length) return null

  return {
    min: v[0],
    median: v[Math.floor(v.length / 2)],
    max: v[v.length - 1]
  }
}

console.log('\nkeydown → dialog frame painted (ms):', stat('frame_ms'))
console.log('keydown → rows painted        (ms):', stat('rows_ms'))
console.log('long-task time in window      (ms):', stat('longtask_ms'))

cdp.close()
