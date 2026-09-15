// Profile-switch latency. Subsumes measure-profile-switch. Backend tier: needs
// a configured profile in the rail and a live backend. Report-only.
//
//   node scripts/perf/run.mjs profile-switch --profile <name>

import { SELECTORS, sleep } from '../lib/cdp.mjs'

export default {
  name: 'profile-switch',
  tier: 'backend',
  description: 'Click a profile in the rail and wait for its sidebar to settle.',
  requiredOpts: ['profile'],
  async run(cdp, opts = {}) {
    const profile = opts.profile
    const settleTimeoutMs = Number(opts.settleTimeoutMs ?? 60000)

    if (!profile) {
      throw new Error('profile-switch needs --profile <name>')
    }

    await cdp.send('Runtime.enable')

    const t0 = await cdp.eval(`(() => {
      const rail = document.querySelector(${JSON.stringify(SELECTORS.profileRail)})
      if (!rail) return null
      const target = [...rail.querySelectorAll('button, [role="tab"]')].find(b =>
        ((b.getAttribute('aria-label') || '') + ' ' + (b.title || '') + ' ' + (b.textContent || ''))
          .toLowerCase().includes(${JSON.stringify(String(profile).toLowerCase())}))
      if (!target) return null
      target.click()
      return performance.now()
    })()`)

    if (t0 === null) {
      throw new Error(`profile "${profile}" not found in the rail`)
    }

    const deadline = Date.now() + settleTimeoutMs
    let settledMs = null

    while (Date.now() < deadline) {
      await sleep(100)
      const s = await cdp.eval(`(() => {
        const label = [...document.querySelectorAll('div[aria-hidden]')].find(el => /waking up/i.test(el.textContent || ''))
        const overlayVisible = label ? Number(getComputedStyle(label).opacity) > 0.05 : false
        return { t: performance.now(), overlayVisible, rows: document.querySelectorAll(${JSON.stringify(SELECTORS.rowButton)}).length }
      })()`)

      if (!s.overlayVisible && s.rows > 0) {
        settledMs = s.t - t0
        break
      }
    }

    return {
      metrics: { profile_switch_settled_ms: settledMs === null ? -1 : Math.round(settledMs) },
      detail: { profile, timedOut: settledMs === null }
    }
  }
}
