// What happens during a SLOW session switch? Click a heavy row with tracing
// on, dump the style/layout/script split plus top callsites.
import { attach } from './perf/lib/launch.mjs'
import { sleep } from './perf/lib/cdp.mjs'

const { cdp, teardown } = await attach({ port: 9222 })

const CLICK_HEAVIEST = `
  (() => {
    const rows = [...document.querySelectorAll('[data-slot="row-button"]')].filter(el => el.offsetParent)
    if (rows.length < 2) return 'need rows'
    // Alternate between the first two rows so every run actually switches.
    const current = location.hash
    const target = rows.find(r => !r.getAttribute('data-active')) ?? rows[1]
    target.click()
    return 'clicked: ' + (target.textContent ?? '').slice(0, 40)
  })()
`

const events = []
let complete = false
cdp.on('Tracing.dataCollected', p => events.push(...(p.value ?? [])))
cdp.on('Tracing.tracingComplete', () => {
  complete = true
})

try {
  await cdp.send('Runtime.enable')

  await cdp.send('Tracing.start', {
    transferMode: 'ReportEvents',
    traceConfig: { includedCategories: ['devtools.timeline'] }
  })

  console.log(await cdp.eval(CLICK_HEAVIEST))
  await sleep(2500)
  console.log(await cdp.eval(CLICK_HEAVIEST))
  await sleep(2500)

  await cdp.send('Tracing.end')

  for (let w = 0; !complete && w < 10000; w += 200) {
    await sleep(200)
  }

  const totals = new Map()
  const byFn = new Map()

  for (const e of events) {
    if (e.ph !== 'X' || typeof e.dur !== 'number') {
      continue
    }

    totals.set(e.name, (totals.get(e.name) ?? 0) + e.dur / 1000)

    if (e.name === 'FunctionCall') {
      const d = e.args?.data ?? {}
      const key = `${d.functionName || '(anon)'} @ ${(d.url || '?').split('/').pop()}:${d.lineNumber ?? '?'}`
      byFn.set(key, (byFn.get(key) ?? 0) + e.dur / 1000)
    }
  }

  const style = totals.get('UpdateLayoutTree') ?? 0
  const layout = totals.get('Layout') ?? 0
  const script = (totals.get('FunctionCall') ?? 0) + (totals.get('EvaluateScript') ?? 0) + (totals.get('TimerFire') ?? 0)
  console.log(`\nVERDICT over 2 switches: style=${style.toFixed(0)}ms layout=${layout.toFixed(0)}ms script=${script.toFixed(0)}ms paint=${(totals.get('Paint') ?? 0).toFixed(0)}ms`)
  console.log('\nTOP CALLSITES:')

  for (const [name, ms] of [...byFn.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12)) {
    console.log(`  ${ms.toFixed(1).padStart(8)}  ${name}`)
  }
} finally {
  teardown?.()
}
