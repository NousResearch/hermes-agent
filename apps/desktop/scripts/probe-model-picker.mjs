// Model picker open latency probe: click the composer model pill, time until
// the dropdown/dialog content paints, repeat. Run with --cpuprofile via the
// harness runner once promoted; standalone for iteration.
//   node scripts/perf/probe-model-picker.mjs [--port 9222] [--rounds 5]
import { CDP } from './perf/lib/cdp.mjs'

const args = process.argv.slice(2)
const flag = name => {
  const i = args.indexOf(`--${name}`)
  return i >= 0 ? args[i + 1] : undefined
}
const port = Number(flag('port') ?? 9222)
const rounds = Number(flag('rounds') ?? 5)
const sleep = ms => new Promise(r => setTimeout(r, ms))

const cdp = await CDP.connect({ port })
await cdp.send('Runtime.enable')

// Find the pill (dropdown path) — the composer model selector button.
const PILL = `(() => {
  const btns = [...document.querySelectorAll('button[aria-label]')]
  const pill = btns.find(b => /model/i.test(b.getAttribute('aria-label') || '') && b.closest('[data-slot]'))
  return pill ? (pill.getAttribute('aria-label') || 'found') : null
})()`

console.log('pill:', await cdp.eval(PILL))

const MEASURE = `
  (async () => {
    const btns = [...document.querySelectorAll('button[aria-label]')]
    const pill = btns.find(b => /model/i.test(b.getAttribute('aria-label') || ''))
    if (!pill) return JSON.stringify({ error: 'no pill' })

    const t0 = performance.now()
    pill.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
    pill.dispatchEvent(new PointerEvent('pointerup', { bubbles: true }))
    pill.click()

    // Wait for menu/dialog content to exist AND paint (double rAF after found).
    const found = await new Promise(resolve => {
      const deadline = performance.now() + 5000
      const check = () => {
        const menu = document.querySelector('[role="menu"], [role="dialog"] [cmdk-list]')
        if (menu && menu.childElementCount > 0) {
          requestAnimationFrame(() => requestAnimationFrame(() => resolve(performance.now())))
          return
        }
        if (performance.now() > deadline) { resolve(null); return }
        requestAnimationFrame(check)
      }
      check()
    })

    const openMs = found ? found - t0 : null
    const rows = document.querySelectorAll('[role="menu"] [role="menuitem"], [cmdk-item]').length

    // Close: Escape.
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
    const menu = document.querySelector('[role="menu"], [role="dialog"]')
    menu?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
    await new Promise(r => setTimeout(r, 300))

    return JSON.stringify({ openMs, rows })
  })()
`

const samples = []

for (let i = 0; i < rounds; i++) {
  const raw = await cdp.eval(MEASURE, { awaitPromise: true })
  const r = JSON.parse(raw)
  console.log(`round ${i}:`, r)
  if (typeof r.openMs === 'number') samples.push(r.openMs)
  await sleep(500)
}

samples.sort((a, b) => a - b)
console.log('\nopen latency ms — min/median/max:',
  Math.round(samples[0]), '/', Math.round(samples[Math.floor(samples.length / 2)]), '/', Math.round(samples.at(-1)))
cdp.close()
