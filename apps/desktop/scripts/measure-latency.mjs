#!/usr/bin/env node
// Measure end-to-end keystroke→paint latency in the Electron renderer.
//
// For each synthetic keystroke we record:
//   t0 = Input.dispatchKeyEvent send time
//   t1 = first observed mutation of [data-slot="composer-rich-input"] childList/character data
//   t2 = first requestAnimationFrame callback after t1 (proxy for next paint)
//
// We use Page.startScreencast briefly to also get frame-presentation timestamps;
// alternatively rely on rAF timing which is close enough for typing UX.
//
// Output: per-char latency histogram (min/p50/p95/p99/max) + samples > 16ms.
//
// Usage:
//   node apps/desktop/scripts/measure-latency.mjs [--chars=100] [--cps=15] [--port=9222]

import { writeFileSync } from 'node:fs'

const args = Object.fromEntries(
  process.argv.slice(2).flatMap(s => {
    const m = s.match(/^--([^=]+)(?:=(.*))?$/)
    return m ? [[m[1], m[2] ?? true]] : []
  })
)
const PORT = Number(args.port ?? 9222)
const CHARS = Number(args.chars ?? 100)
const CPS = Number(args.cps ?? 15)

const log = (...m) => console.log('[latency]', ...m)

async function pickRenderer() {
  const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
  return list.find(t => t.type === 'page' && t.url.startsWith('http'))
}

function connect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url)
    let id = 0
    const pending = new Map()
    const events = new Map()
    ws.addEventListener('open', () =>
      resolve({
        send(method, params = {}) {
          const myId = ++id
          ws.send(JSON.stringify({ id: myId, method, params }))
          return new Promise((res, rej) => pending.set(myId, { res, rej }))
        },
        on(method, h) {
          if (!events.has(method)) events.set(method, [])
          events.get(method).push(h)
        },
        close: () => ws.close()
      })
    )
    ws.addEventListener('error', reject)
    ws.addEventListener('message', ev => {
      const m = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString('utf8'))
      if (m.id != null) {
        const p = pending.get(m.id)
        if (!p) return
        pending.delete(m.id)
        m.error ? p.rej(new Error(m.error.message)) : p.res(m.result)
      } else if (m.method) {
        ;(events.get(m.method) ?? []).forEach(h => h(m.params))
      }
    })
  })
}

async function evalInPage(cdp, expr) {
  const r = await cdp.send('Runtime.evaluate', { expression: expr, returnByValue: true })
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text)
  return r.result.value
}

async function main() {
  const tgt = await pickRenderer()
  log(`target ${tgt.url}`)
  const cdp = await connect(tgt.webSocketDebuggerUrl)
  await cdp.send('Runtime.enable')

  await evalInPage(
    cdp,
    `(() => {
      const el = document.querySelector('[data-slot="composer-rich-input"]')
      if (!el) return false
      el.focus()
      const range = document.createRange()
      range.selectNodeContents(el)
      range.collapse(false)
      const sel = window.getSelection()
      sel.removeAllRanges()
      sel.addRange(range)
      window.__keypressTimings = []
      window.__pendingKey = null
      // Observe the composer for content/text changes; record the time relative
      // to the most recent simulated keypress timestamp set on window.__pendingKey.
      const obs = new MutationObserver(() => {
        const start = window.__pendingKey
        if (start === null) return
        const mutationT = performance.now()
        window.__pendingKey = null
        requestAnimationFrame(() => {
          const paintT = performance.now()
          window.__keypressTimings.push({
            start, mutationT, paintT,
            mutationLatency: mutationT - start,
            paintLatency: paintT - start
          })
        })
      })
      obs.observe(el, { childList: true, subtree: true, characterData: true })
      window.__keystrokeObserver = obs
      return true
    })()`
  )

  const lorem =
    'the quick brown fox jumps over the lazy dog while typing into this composer feels like wading through molasses on a hot afternoon. '
  let text = ''
  while (text.length < CHARS) text += lorem
  text = text.slice(0, CHARS)

  const intervalMs = Math.max(1, Math.round(1000 / CPS))
  const start = Date.now()
  for (let i = 0; i < text.length; i++) {
    // Mark the keypress time inside the page so it's measured from the same clock.
    await evalInPage(cdp, `window.__pendingKey = performance.now()`)
    await cdp.send('Input.dispatchKeyEvent', { type: 'char', text: text[i], unmodifiedText: text[i] })
    const expected = start + (i + 1) * intervalMs
    const wait = expected - Date.now()
    if (wait > 0) await new Promise(r => setTimeout(r, wait))
  }

  await new Promise(r => setTimeout(r, 500))
  const samples = await evalInPage(cdp, `window.__keypressTimings`)
  log(`${samples.length} keystroke samples measured out of ${text.length} typed`)

  // Clear composer for next run
  await evalInPage(cdp, `
    (() => {
      const el = document.querySelector('[data-slot="composer-rich-input"]')
      if (el) { el.innerHTML = ''; el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' })) }
      window.__keystrokeObserver?.disconnect()
    })()
  `)

  const mutLat = samples.map(s => s.mutationLatency).sort((a, b) => a - b)
  const paintLat = samples.map(s => s.paintLatency).sort((a, b) => a - b)
  const stat = arr => ({
    n: arr.length,
    min: arr[0]?.toFixed(2),
    p50: arr[Math.floor(arr.length * 0.5)]?.toFixed(2),
    p90: arr[Math.floor(arr.length * 0.9)]?.toFixed(2),
    p95: arr[Math.floor(arr.length * 0.95)]?.toFixed(2),
    p99: arr[Math.floor(arr.length * 0.99)]?.toFixed(2),
    max: arr[arr.length - 1]?.toFixed(2),
    mean: arr.length ? (arr.reduce((s, x) => s + x, 0) / arr.length).toFixed(2) : 0
  })

  console.log('\n=== keypress → mutation latency (ms) ===')
  console.log(' ', stat(mutLat))
  console.log('\n=== keypress → next rAF (≈paint) latency (ms) ===')
  console.log(' ', stat(paintLat))

  const slow = samples.filter(s => s.paintLatency > 16)
  console.log(`\n=== ${slow.length}/${samples.length} keystrokes >16ms (one frame) ===`)
  if (slow.length) {
    const slowSorted = [...slow].sort((a, b) => b.paintLatency - a.paintLatency).slice(0, 10)
    for (const s of slowSorted) {
      console.log(`  paint=${s.paintLatency.toFixed(1)}ms  mut=${s.mutationLatency.toFixed(1)}ms  at t=${s.start.toFixed(0)}`)
    }
  }

  writeFileSync('/tmp/hermes-latency-samples.json', JSON.stringify(samples, null, 2))

  cdp.close()
}

main().catch(e => {
  console.error('[latency] fatal:', e.stack ?? e.message)
  process.exit(1)
})
