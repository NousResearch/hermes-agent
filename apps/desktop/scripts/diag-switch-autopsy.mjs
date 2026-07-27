// Session-switch autopsy: click between the two heaviest rows repeatedly,
// recording per-switch (a) settled ms, (b) React commits, (c) top rendered
// components — so slow switches name themselves.
import { attach } from './perf/lib/launch.mjs'
import { sleep } from './perf/lib/cdp.mjs'

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`)

  return i === -1 ? fallback : process.argv[i + 1]
}

const port = Number(arg('port', 9222))
const ROUNDS = Number(arg('rounds', 8))

const { cdp, teardown } = await attach({ port })

const SWITCH_ONE = index => `
  (async () => {
    const rows = [...document.querySelectorAll('[data-slot="row-button"]')].filter(el => el.offsetParent)
    if (rows.length < 2) return JSON.stringify({ error: 'rows' })
    const row = rows[${index} % 2]
    const rc = window.__RENDER_COUNTS__
    rc.start()
    const t0 = performance.now()
    row.click()
    let lastMutation = performance.now()
    const mo = new MutationObserver(() => { lastMutation = performance.now() })
    mo.observe(document.body, { childList: true, subtree: true, characterData: true })
    const deadline = performance.now() + 4000
    while (performance.now() < deadline) {
      await new Promise(r => requestAnimationFrame(r))
      if (performance.now() - lastMutation > 150) break
    }
    mo.disconnect()
    rc.stop()
    const settled = Math.round(performance.now() - t0 - 150)
    const report = rc.report(6).map(r => r.name + ':' + r.renders + '(' + Math.round(r.totalMs) + 'ms)')
    return JSON.stringify({ label: (row.textContent ?? '').slice(0, 24), settled, commits: rc.commits(), top: report })
  })()
`

try {
  await cdp.send('Runtime.enable')

  for (let i = 0; i < ROUNDS; i++) {
    console.log(await cdp.eval(SWITCH_ONE(i)))
    await sleep(400)
  }
} finally {
  teardown?.()
}
