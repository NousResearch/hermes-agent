// Measures the real composer input path after a long transcript has fully settled.
// Keeping focus settlement outside the counters isolates typing from focus work.

import { SELECTORS, sleep } from '../lib/cdp.mjs'
import { percentile } from '../lib/stats.mjs'

const visibleComposer = `
  ([...document.querySelectorAll(${JSON.stringify(SELECTORS.composer)})].find(el => {
    const box = el.getBoundingClientRect()
    const style = getComputedStyle(el)
    return box.width > 0 && box.height > 0 && style.display !== 'none' && style.visibility !== 'hidden'
  }) ?? null)
`

const quiesce = (quietMs, timeoutMs) => `
  (async () => {
    const rc = window.__RENDER_COUNTS__
    rc.start()
    const deadline = Date.now() + ${timeoutMs}
    let last = -1
    let stableSince = Date.now()
    while (Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 100))
      const commits = rc.commits()
      if (commits !== last) {
        last = commits
        stableSince = Date.now()
      } else if (Date.now() - stableSince >= ${quietMs}) {
        rc.stop()
        rc.clear()
        return 'quiet:' + commits
      }
    }
    const report = rc.report(20)
    rc.stop()
    rc.clear()
    return 'timeout:' + JSON.stringify({ commits: last, report })
  })()
`

const focusComposer = `
  (() => {
    const el = ${visibleComposer}
    if (!el) return false
    el.focus()
    const range = document.createRange()
    range.selectNodeContents(el)
    range.collapse(false)
    const selection = window.getSelection()
    selection.removeAllRanges()
    selection.addRange(range)
    return true
  })()
`

const install = `
  (() => {
    const el = ${visibleComposer}
    if (!el || !window.__RENDER_COUNTS__ || !window.__ATOM_CHURN__) return false

    window.__LOADED_KEY__ = { samples: [], pending: null }
    const observer = new MutationObserver(() => {
      const start = window.__LOADED_KEY__.pending
      if (start === null) return
      window.__LOADED_KEY__.pending = null
      requestAnimationFrame(() => window.__LOADED_KEY__.samples.push(performance.now() - start))
    })
    observer.observe(el, { childList: true, subtree: true, characterData: true })
    window.__LOADED_KEY__.observer = observer
    window.__RENDER_COUNTS__.start()
    window.__ATOM_CHURN__.start()
    return true
  })()
`

const collect = `
  (() => {
    window.__RENDER_COUNTS__.stop()
    window.__ATOM_CHURN__.stop()
    window.__LOADED_KEY__.observer.disconnect()
    const renderTotals = [...window.__RENDER_COUNTS__.counts.values()].reduce(
      (totals, row) => ({
        renders: totals.renders + row.renders,
        wasted: totals.wasted + row.wasted
      }),
      { renders: 0, wasted: 0 }
    )
    return JSON.stringify({
      samples: window.__LOADED_KEY__.samples,
      commits: window.__RENDER_COUNTS__.commits(),
      renderOwners: window.__RENDER_COUNTS__.counts.size,
      renderTotals,
      renders: window.__RENDER_COUNTS__.report(200),
      atoms: window.__ATOM_CHURN__.report(200)
    })
  })()
`

const cleanup = `
  (() => {
    try { window.__LOADED_KEY__?.observer?.disconnect() } catch {}
    window.__RENDER_COUNTS__?.stop()
    window.__ATOM_CHURN__?.stop()
    window.__RENDER_COUNTS__?.clear()
    window.__ATOM_CHURN__?.clear()
    const el = ${visibleComposer}
    if (el) {
      el.innerHTML = ''
      el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' }))
    }
    window.__PERF_DRIVE__?.reset()
    window.__LOADED_KEY__ = null
    return 'cleaned'
  })()
`

const sentence = 'the quick brown fox jumps over the lazy dog while Hermes stays smooth under load. '

export default {
  name: 'loaded-keystroke',
  tier: 'report',
  description: 'Composer latency and render/store churn after a settled long transcript.',
  async run(cdp, opts = {}) {
    const turns = Number(opts.turns ?? 180)
    const chars = Number(opts.chars ?? 120)
    const cps = Number(opts.cps ?? 15)
    const historySettleMs = Number(opts.historySettleMs ?? (turns > 0 ? 12000 : 1000))

    await cdp.send('Runtime.enable')

    if (turns > 0) {
      await cdp.eval(`window.__PERF_DRIVE__.loadTranscript(${turns})`)
    }

    await sleep(historySettleMs)
    const focused = await cdp.eval(focusComposer)
    if (!focused) throw new Error('loaded-keystroke could not focus the visible composer')
    await sleep(1000)
    const settle = await cdp.eval(quiesce(1000, 20000))
    const installed = await cdp.eval(install)
    if (!installed) throw new Error('loaded-keystroke could not find the visible composer or perf counters')

    let text = ''
    while (text.length < chars) text += sentence
    text = text.slice(0, chars)
    const intervalMs = Math.max(1, Math.round(1000 / cps))
    const startedAt = Date.now()

    try {
      for (let index = 0; index < text.length; index += 1) {
        await cdp.eval('window.__LOADED_KEY__.pending = performance.now()')
        await cdp.send('Input.dispatchKeyEvent', {
          type: 'char',
          text: text[index],
          unmodifiedText: text[index]
        })
        const wait = startedAt + (index + 1) * intervalMs - Date.now()
        if (wait > 0) await sleep(wait)
      }

      await sleep(500)
      const data = JSON.parse(await cdp.eval(collect))
      const notifications = data.atoms.reduce((sum, row) => sum + row.notifications, 0)
      const listenerCalls = data.atoms.reduce((sum, row) => sum + row.listenerCalls, 0)
      const shikiStateUpdates = data.renders.find(row => row.name === 'CachedShikiBlock')?.stateChanged ?? 0
      const round = value => Math.round(value * 10) / 10

      return {
        metrics: {
          keystroke_p50_ms: round(percentile(data.samples, 0.5)),
          keystroke_p95_ms: round(percentile(data.samples, 0.95)),
          keystroke_p99_ms: round(percentile(data.samples, 0.99)),
          keystroke_slow_16: data.samples.filter(sample => sample > 16).length,
          commits: data.commits,
          total_renders: data.renderTotals.renders,
          wasted_renders: data.renderTotals.wasted,
          shiki_state_updates: shikiStateUpdates,
          atom_notifications: notifications,
          atom_listener_calls: listenerCalls
        },
        detail: {
          turns,
          chars,
          cps,
          n: data.samples.length,
          renderOwners: data.renderOwners,
          settle,
          topOwners: data.renders.filter(row => row.stateChanged > 0 && row.propsChanged === 0).slice(0, 20),
          topRenders: data.renders.slice(0, 30),
          topAtoms: data.atoms.slice(0, 30)
        }
      }
    } finally {
      await cdp.eval(cleanup)
    }
  }
}
