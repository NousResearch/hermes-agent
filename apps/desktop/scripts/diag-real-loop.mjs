// The real-app perf loop: drive HER hgui instance (real profile, real
// sessions) through the three interactions that matter — session switch,
// sidebar drag, composer typing — and report honest single-clock numbers.
//
//   node scripts/diag-real-loop.mjs [--port 9222] [--switches 6]
//
// Unlike the synthetic scenarios this clicks REAL sidebar rows, so session
// switching is measured as the user feels it: click -> transcript painted.

import { attach } from './perf/lib/launch.mjs'
import { sleep } from './perf/lib/cdp.mjs'

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`)

  return i === -1 ? fallback : process.argv[i + 1]
}

const port = Number(arg('port', 9222))
const SWITCHES = Number(arg('switches', 6))

const { cdp, teardown } = await attach({ port })

// ---------------------------------------------------------------------------
// Session switch: click a sidebar session row, await the transcript settling.
// Measures click -> first paint of the new transcript AND click -> settled
// (two rAFs with no further DOM mutation in the thread viewport).
// ---------------------------------------------------------------------------
const SWITCH = swaps => `
  (async () => {
    const rows = [...document.querySelectorAll('[data-slot="row-button"]')]
      .filter(el => el.offsetParent && (el.textContent ?? '').trim())
    if (rows.length < 2) return JSON.stringify({ error: 'need 2+ visible session rows, found ' + rows.length })

    const results = []
    for (let i = 0; i < ${swaps}; i++) {
      const row = rows[i % Math.min(rows.length, 4)]
      const viewport = () => document.querySelector('[data-slot="aui_thread-viewport"]')
      const t0 = performance.now()
      row.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true, pointerId: 1, isPrimary: true, button: 0, buttons: 1 }))
      row.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, cancelable: true, pointerId: 1, isPrimary: true, button: 0 }))
      row.click()

      // First paint: next two rAFs after the click.
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))
      const firstPaint = performance.now() - t0

      // Settled: no mutations in the viewport for 2 consecutive frames, capped 3s.
      let lastMutation = performance.now()
      const target = viewport() ?? document.body
      const mo = new MutationObserver(() => { lastMutation = performance.now() })
      mo.observe(target, { childList: true, subtree: true, characterData: true })
      const deadline = performance.now() + 3000
      while (performance.now() < deadline) {
        await new Promise(r => requestAnimationFrame(r))
        if (performance.now() - lastMutation > 120) break
      }
      mo.disconnect()
      results.push({ firstPaint: Math.round(firstPaint), settled: Math.round(performance.now() - t0 - 120) })
      await new Promise(r => setTimeout(r, 250))
    }
    return JSON.stringify(results)
  })()
`

// ---------------------------------------------------------------------------
// Drag the first visible sash, single-clock frames.
// ---------------------------------------------------------------------------
const DRAG = `
  (async () => {
    const handle = [...document.querySelectorAll('[role="separator"]')].find(el => el.offsetParent || el.getBoundingClientRect().width > 0)
    if (!handle) return JSON.stringify({ error: 'no sash' })
    const box = handle.getBoundingClientRect()
    const y = box.top + box.height / 2
    const x0 = box.left + box.width / 2
    let x = x0
    const o = { bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse', isPrimary: true, button: 0, buttons: 1 }
    const frames = []
    let last = performance.now()
    handle.dispatchEvent(new PointerEvent('pointerdown', { ...o, clientX: x, clientY: y }))
    for (let i = 0; i < 60; i++) {
      x += (i < 30 ? 2 : -2)
      window.dispatchEvent(new PointerEvent('pointermove', { ...o, clientX: x, clientY: y }))
      await new Promise(r => requestAnimationFrame(r))
      const now = performance.now(); frames.push(now - last); last = now
    }
    window.dispatchEvent(new PointerEvent('pointerup', { ...o, buttons: 0, clientX: x, clientY: y }))
    const total = frames.reduce((a, b) => a + b, 0)
    const sorted = [...frames].sort((a, b) => a - b)
    return JSON.stringify({
      fps: Math.round((frames.length / total) * 1000 * 10) / 10,
      p95: Math.round(sorted[Math.floor(sorted.length * 0.95)] * 10) / 10,
      worst: Math.round(sorted[sorted.length - 1] * 10) / 10,
      slow33: frames.filter(f => f > 33).length
    })
  })()
`

// ---------------------------------------------------------------------------
// Type into the composer, single-clock frames (one mark per keystroke frame).
// ---------------------------------------------------------------------------
const TYPE = `
  (async () => {
    const el = [...document.querySelectorAll('[contenteditable="true"]')].find(e => e.offsetParent)
    if (!el) return JSON.stringify({ error: 'no composer' })
    el.focus()
    const frames = []
    let last = performance.now()
    for (let i = 0; i < 40; i++) {
      const ch = 'the quick brown fox '[i % 20]
      el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: ch }))
      el.textContent += ch
      el.dispatchEvent(new InputEvent('input', { bubbles: true, data: ch, inputType: 'insertText' }))
      el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: ch }))
      await new Promise(r => requestAnimationFrame(r))
      const now = performance.now(); frames.push(now - last); last = now
      await new Promise(r => setTimeout(r, 20))
    }
    // Clear what we typed.
    el.textContent = ''
    el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' }))
    const moving = frames
    const total = moving.reduce((a, b) => a + b, 0)
    const sorted = [...moving].sort((a, b) => a - b)
    return JSON.stringify({
      fps: Math.round((moving.length / total) * 1000 * 10) / 10,
      p95: Math.round(sorted[Math.floor(sorted.length * 0.95)] * 10) / 10,
      worst: Math.round(sorted[sorted.length - 1] * 10) / 10,
      slow33: moving.filter(f => f > 33).length
    })
  })()
`

try {
  await cdp.send('Runtime.enable')

  console.log('== SESSION SWITCH (click -> paint / settled ms) ==')
  console.log(await cdp.eval(SWITCH(SWITCHES)))

  await sleep(500)
  console.log('\n== SIDEBAR DRAG ==')
  console.log(await cdp.eval(DRAG))

  await sleep(500)
  console.log('\n== COMPOSER TYPING ==')
  console.log(await cdp.eval(TYPE))
} finally {
  teardown?.()
}
