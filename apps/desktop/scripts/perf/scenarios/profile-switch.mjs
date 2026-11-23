// Profile-switch latency. Subsumes measure-profile-switch. Backend tier: needs
// a configured profile in the rail and a live backend. Report-only.
//
//   node scripts/perf/run.mjs profile-switch --profile <name>

import { SELECTORS, sleep } from '../lib/cdp.mjs'

export default {
  name: 'profile-switch',
  tier: 'backend',
  description: 'Click a profile in the rail and wait for the switch indicator and rail square to settle.',
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
      // Held for the poll below: the clicked square's aria-busy is half of
      // the readiness signal.
      window.__perfProfileSwitchTarget = target
      target.click()
      return performance.now()
    })()`)

    if (t0 === null) {
      throw new Error(`profile "${profile}" not found in the rail`)
    }

    const deadline = Date.now() + settleTimeoutMs
    let settledMs = null
    let failed = false

    while (Date.now() < deadline) {
      await sleep(100)
      // Deterministic readiness, not overlay text: the chat's switch
      // indicator publishes data-switch-state (pending/slow/stalled while a
      // switch is in flight, failed on a rejected one) and the clicked rail
      // square carries aria-busy until its activation settles.
      const s = await cdp.eval(`(() => {
        const indicator = document.querySelector('[data-slot="chat-switch-indicator"]')
        const target = window.__perfProfileSwitchTarget
        return {
          t: performance.now(),
          state: indicator ? indicator.getAttribute('data-switch-state') : null,
          busy: Boolean(target && target.getAttribute('aria-busy') === 'true'),
          rows: document.querySelectorAll(${JSON.stringify(SELECTORS.rowButton)}).length
        }
      })()`)

      if (s.state === 'failed') {
        failed = true
        break
      }

      const switching = s.busy || s.state === 'pending' || s.state === 'slow' || s.state === 'stalled'

      if (s.state !== null && !switching && s.rows > 0) {
        settledMs = s.t - t0
        break
      }
    }

    return {
      metrics: { profile_switch_settled_ms: settledMs === null ? -1 : Math.round(settledMs) },
      detail: { failed, profile, timedOut: settledMs === null && !failed }
    }
  }
}
