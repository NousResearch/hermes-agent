// Composer input latency — keystroke → next paint. Subsumes measure-latency,
// profile-typing, and leak-typing. This is the most-felt latency in a chat app
// (users type constantly) and nothing measured it against a baseline before.
//
// Each synthetic char records the time from dispatch to the first rAF after the
// composer mutates (a paint proxy). Metrics are p50/p95/p99 and the count of
// keystrokes that missed a 16ms frame.

import { SELECTORS, sleep } from '../lib/cdp.mjs'
import { percentile } from '../lib/stats.mjs'

const INSTALL = `
  (() => {
    const el = document.querySelector(${JSON.stringify(SELECTORS.composer)})
    if (!el) return false
    el.focus()
    const range = document.createRange()
    range.selectNodeContents(el)
    range.collapse(false)
    const sel = window.getSelection()
    sel.removeAllRanges()
    sel.addRange(range)
    window.__KEY__ = { samples: [], pending: null }
    const obs = new MutationObserver(() => {
      const start = window.__KEY__.pending
      if (start === null) return
      window.__KEY__.pending = null
      requestAnimationFrame(() => window.__KEY__.samples.push(performance.now() - start))
    })
    obs.observe(el, { childList: true, subtree: true, characterData: true })
    window.__KEY__.obs = obs
    return true
  })()
`

const CLEAR = `
  (() => {
    const el = document.querySelector(${JSON.stringify(SELECTORS.composer)})
    if (el) { el.innerHTML = ''; el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' })) }
    window.__KEY__ && window.__KEY__.obs && window.__KEY__.obs.disconnect()
  })()
`

const SENTENCE =
  'the quick brown fox jumps over the lazy dog while typing into this composer, which should feel instant. '

export default {
  name: 'keystroke',
  tier: 'ci',
  description: 'Composer keystroke → paint latency while idle.',
  async run(cdp, opts = {}) {
    const chars = Number(opts.chars ?? 120)
    const cps = Number(opts.cps ?? 15)

    await cdp.send('Runtime.enable')

    const installed = await cdp.eval(INSTALL)

    if (!installed) {
      throw new Error(`composer not found (${SELECTORS.composer}); is a chat view open?`)
    }

    let text = ''

    while (text.length < chars) {
      text += SENTENCE
    }

    text = text.slice(0, chars)
    const intervalMs = Math.max(1, Math.round(1000 / cps))
    const start = Date.now()

    for (let i = 0; i < text.length; i++) {
      await cdp.eval('window.__KEY__.pending = performance.now()')
      await cdp.send('Input.dispatchKeyEvent', { type: 'char', text: text[i], unmodifiedText: text[i] })
      const wait = start + (i + 1) * intervalMs - Date.now()

      if (wait > 0) {
        await sleep(wait)
      }
    }

    await sleep(300)
    const samples = await cdp.eval('window.__KEY__.samples')
    await cdp.eval(CLEAR)

    const round = n => Math.round(n * 10) / 10

    return {
      metrics: {
        keystroke_p50_ms: round(percentile(samples, 0.5)),
        keystroke_p95_ms: round(percentile(samples, 0.95)),
        keystroke_p99_ms: round(percentile(samples, 0.99)),
        keystroke_slow_16: samples.filter(s => s > 16).length
      },
      detail: { n: samples.length, typed: text.length }
    }
  }
}
