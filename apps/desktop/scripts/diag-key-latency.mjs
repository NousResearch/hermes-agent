// Typing latency, isolated: keystroke -> next paint, with and without an
// active stream. Distinguishes "input is slow" from "the frame budget is
// consumed by streaming flushes" — the fix differs completely.
import { attach } from './perf/lib/launch.mjs'

const { cdp, teardown } = await attach({ port: 9222 })

const TYPE = `
  (async () => {
    const el = [...document.querySelectorAll('[contenteditable="true"]')].find(e => e.offsetParent)
    if (!el) return JSON.stringify({ error: 'no composer' })
    el.focus()
    const perKey = []
    for (let i = 0; i < 30; i++) {
      const ch = 'abcdefghij'[i % 10]
      const t0 = performance.now()
      el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: ch }))
      el.textContent += ch
      el.dispatchEvent(new InputEvent('input', { bubbles: true, data: ch, inputType: 'insertText' }))
      el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: ch }))
      await new Promise(r => requestAnimationFrame(r))
      perKey.push(performance.now() - t0)
      // Human-ish 80ms cadence so streaming flushes interleave realistically.
      await new Promise(r => setTimeout(r, 80))
    }
    el.textContent = ''
    el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' }))
    const sorted = [...perKey].sort((a, b) => a - b)
    const pct = p => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))]
    const busy = (() => { try { return document.querySelectorAll('[data-status="running"]').length } catch { return -1 } })()
    return JSON.stringify({
      keyToPaint_p50: Math.round(pct(0.5) * 10) / 10,
      keyToPaint_p95: Math.round(pct(0.95) * 10) / 10,
      worst: Math.round(sorted[sorted.length - 1] * 10) / 10,
      over16: perKey.filter(f => f > 16.7).length,
      over33: perKey.filter(f => f > 33).length,
      streamingParts: busy
    })
  })()
`

try {
  await cdp.send('Runtime.enable')
  console.log(await cdp.eval(TYPE))
} finally {
  teardown?.()
}
