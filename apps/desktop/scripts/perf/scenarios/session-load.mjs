// Visual stability of a session LOAD. Sibling to `submit` (which measures the
// jump on Enter); this one measures the jump on opening a session — the
// prepend/settle path, not the append path.
//
// Clicks sidebar rows and tracks the bottom-most turn's on-screen top every
// frame. A clean load never moves it after first paint; a janky one strands it
// thousands of px away while the render-budget backfill and stick-to-bottom
// argue. Backend tier: needs real stored sessions in the sidebar.
//
//   node scripts/perf/run.mjs session-load --rows 2,5,8 --rounds 2

import { SELECTORS, sleep } from '../lib/cdp.mjs'
import { summarize } from '../lib/stats.mjs'

// Below this a shift is sub-perceptual (sub-pixel rounding, a settling caret).
const SHIFT_PX = 4

const ARM = `
  (() => {
    const samples = []
    const t0 = performance.now()
    let running = true

    const tick = () => {
      if (!running) return
      const v = document.querySelector(${JSON.stringify(SELECTORS.threadViewport)})

      if (v) {
        const turns = v.querySelectorAll(
          ${JSON.stringify(SELECTORS.turnPair)} + ',' + ${JSON.stringify(SELECTORS.assistantMessage)}
        )
        const last = turns[turns.length - 1]
        const rect = last && last.getBoundingClientRect()
        samples.push({
          st: Math.round(v.scrollTop),
          sh: v.scrollHeight,
          ch: v.clientHeight,
          bottomTop: rect ? Math.round(rect.top - v.getBoundingClientRect().top) : null,
          turns: turns.length,
          t: Math.round(performance.now() - t0)
        })
      }

      requestAnimationFrame(tick)
    }

    requestAnimationFrame(tick)
    window.__SL = { samples, stop() { running = false } }
  })()
`

const CLICK = index => `
  (() => {
    const rows = [...document.querySelectorAll(${JSON.stringify(SELECTORS.rowButton)})].filter(el => el.offsetParent)
    const row = rows[${index}]
    if (!row) return null
    row.click()
    return (row.textContent ?? '').slice(0, 34)
  })()
`

/** Total/max on-screen movement of the bottom turn after it first paints. */
function measureLoad(samples) {
  const painted = samples.findIndex(s => s.turns > 0)
  const after = painted === -1 ? [] : samples.slice(painted)
  const load = { maxShiftPx: 0, offBottomFrames: 0, settledMs: 0, shiftedPx: 0, shifts: 0 }
  let previous = null

  for (const sample of after) {
    if (sample.sh - (sample.st + sample.ch) > 2) {
      load.offBottomFrames += 1
    }

    if (previous?.bottomTop != null && sample.bottomTop != null) {
      const delta = Math.abs(sample.bottomTop - previous.bottomTop)

      if (delta > SHIFT_PX) {
        load.maxShiftPx = Math.max(load.maxShiftPx, delta)
        load.settledMs = sample.t
        load.shiftedPx += delta
        load.shifts += 1
      }
    }

    previous = sample
  }

  return load
}

export default {
  name: 'session-load',
  tier: 'backend',
  description: 'Visual stability of opening a session: how far the transcript moves after first paint.',
  async run(cdp, opts = {}) {
    const rows = String(opts.rows ?? '2,5,8').split(',').map(Number)
    const rounds = Number(opts.rounds ?? 2)
    const watchMs = Number(opts.watchMs ?? 4500)

    await cdp.send('Runtime.enable')

    const loads = []

    for (let round = 0; round < rounds; round++) {
      for (const index of rows) {
        await cdp.eval(ARM)

        if (!(await cdp.eval(CLICK(index)))) {
          continue
        }

        await sleep(watchMs)
        const { samples } = await cdp.eval('(() => { window.__SL.stop(); return window.__SL })()')
        loads.push(measureLoad(samples))
      }

      await sleep(500)
    }

    if (!loads.length) {
      throw new Error('session-load found no sidebar rows to click')
    }

    const total = key => loads.reduce((sum, load) => sum + load[key], 0)

    return {
      metrics: {
        load_max_shift_px: Math.max(...loads.map(load => load.maxShiftPx)),
        load_off_bottom_frames: Math.round(total('offBottomFrames') / loads.length),
        load_settled_p95_ms: summarize(loads.map(load => load.settledMs)).p95,
        load_shifted_px: Math.round(total('shiftedPx') / loads.length)
      },
      detail: { loads: loads.length, shiftsPerLoad: total('shifts') / loads.length }
    }
  }
}
