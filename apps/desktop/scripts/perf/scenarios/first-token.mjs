// Time-to-first-token — Enter → first assistant token painted. The latency an
// agent app is uniquely judged on, spanning the desktop submit path AND the
// backend/agent-loop first-token time. Backend tier: fires a REAL prompt, needs
// a live backend (and credits). Report-only.
//
//   node scripts/perf/run.mjs first-token --spawn --prompt "hi"

import { SELECTORS, sleep, typeIntoComposer } from '../lib/cdp.mjs'
import { summarize } from '../lib/stats.mjs'

export default {
  name: 'first-token',
  tier: 'backend',
  description: 'Enter → first assistant token painted (real backend).',
  async run(cdp, opts = {}) {
    const rounds = Number(opts.rounds ?? 3)
    const prompt = opts.prompt ?? 'reply with a single short sentence'
    const timeoutMs = Number(opts.timeoutMs ?? 60000)

    await cdp.send('Runtime.enable')

    const firstTokens = []

    for (let i = 0; i < rounds; i++) {
      const baseText = await cdp.eval(`(() => {
        const a = document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)})
        return a.length ? a[a.length - 1].textContent.length : 0
      })()`)
      const baseCount = await cdp.eval(`document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)}).length`)

      await typeIntoComposer(cdp, `${prompt} (${i})`, { cps: 60 })
      const submitAt = Date.now()
      await cdp.eval(`(() => {
        const el = document.querySelector(${JSON.stringify(SELECTORS.composer)})
        el && el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true, cancelable: true }))
      })()`)

      const deadline = Date.now() + timeoutMs
      let firstTokenMs = null

      while (Date.now() < deadline) {
        await sleep(25)
        const grown = await cdp.eval(`(() => {
          const a = document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)})
          if (a.length > ${baseCount}) return true
          return a.length ? a[a.length - 1].textContent.length > ${baseText} : false
        })()`)

        if (grown) {
          firstTokenMs = Date.now() - submitAt
          break
        }
      }

      if (firstTokenMs !== null) {
        firstTokens.push(firstTokenMs)
      }

      // Let the turn finish before the next round.
      const turnDeadline = Date.now() + timeoutMs
      while (Date.now() < turnDeadline) {
        await sleep(250)
        const busy = await cdp.eval(`!!document.querySelector('[data-status="running"], [data-busy="true"]')`)

        if (!busy) {
          break
        }
      }

      await sleep(500)
    }

    if (!firstTokens.length) {
      throw new Error('no first token observed — is a backend with credits connected?')
    }

    const s = summarize(firstTokens)

    return {
      metrics: { first_token_p50_ms: s.p50, first_token_p95_ms: s.p95 },
      detail: { rounds, samples: firstTokens, summary: s }
    }
  }
}
