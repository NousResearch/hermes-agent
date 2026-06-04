#!/usr/bin/env node
// Leak-detection harness — measure detached DOM, listener count, and FiberNode
// growth as a function of keystrokes typed.
//
// Workflow:
//   1. Open session, focus composer
//   2. forceGC; capture baseline counts
//   3. Repeat N rounds: type M chars, forceGC, capture counts, clear composer
//   4. Print growth-per-round table
//
// Usage:
//   node apps/desktop/scripts/leak-typing.mjs [--rounds=6] [--chars=200] [--cps=40] [--port=9222]

import { writeFileSync } from 'node:fs'

const args = Object.fromEntries(
  process.argv.slice(2).flatMap(s => {
    const m = s.match(/^--([^=]+)(?:=(.*))?$/)
    return m ? [[m[1], m[2] ?? true]] : []
  })
)
const PORT = Number(args.port ?? 9222)
const ROUNDS = Number(args.rounds ?? 6)
const CHARS = Number(args.chars ?? 200)
const CPS = Number(args.cps ?? 40)

const log = (...m) => console.log('[leak]', ...m)

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

async function forceGCAndSettle(cdp) {
  for (let i = 0; i < 3; i++) {
    await cdp.send('HeapProfiler.collectGarbage')
    await new Promise(r => setTimeout(r, 60))
  }
}

async function focusComposer(cdp) {
  return await evalInPage(
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
      return true
    })()`
  )
}

async function clearComposer(cdp) {
  await evalInPage(
    cdp,
    `(() => {
      const el = document.querySelector('[data-slot="composer-rich-input"]')
      if (!el) return false
      // Clear via the same path as the composer's clear flow:
      // dispatch a single Backspace until empty would be N round-trips; quicker
      // to directly assign empty text and fire input.
      el.innerHTML = ''
      el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' }))
      el.focus()
      return el.innerText.length === 0
    })()`
  )
}

async function snapshotCounts(cdp) {
  // Counts via Runtime.evaluate using internal V8 counters where possible.
  // For DOM stats we directly query the document.
  // Performance metrics include JSHeapUsedSize, Nodes, JSEventListeners, etc.
  const { metrics } = await cdp.send('Performance.getMetrics')
  const byName = Object.fromEntries(metrics.map(m => [m.name, m.value]))
  // Total nodes in document
  const docNodes = await evalInPage(
    cdp,
    `document.getElementsByTagName('*').length + document.querySelectorAll('*').length / 2`
  )
  return {
    heapUsedMB: (byName.JSHeapUsedSize / 1024 / 1024) || 0,
    heapTotalMB: (byName.JSHeapTotalSize / 1024 / 1024) || 0,
    nodes: byName.Nodes || 0,
    jsListeners: byName.JSEventListeners || 0,
    docNodes,
    layoutCount: byName.LayoutCount || 0,
    recalcStyleCount: byName.RecalcStyleCount || 0,
    fps: byName.FramesPerSecond || 0
  }
}

async function typeChars(cdp, text, cps) {
  const intervalMs = Math.max(1, Math.round(1000 / cps))
  const start = Date.now()
  for (let i = 0; i < text.length; i++) {
    await cdp.send('Input.dispatchKeyEvent', { type: 'char', text: text[i], unmodifiedText: text[i] })
    const expected = start + (i + 1) * intervalMs
    const wait = expected - Date.now()
    if (wait > 0) await new Promise(r => setTimeout(r, wait))
  }
}

const lorem =
  'the quick brown fox jumps over the lazy dog while the agent thinks really hard about why typing into this composer feels like wading through molasses on a hot afternoon '
function genText(n) {
  let s = ''
  while (s.length < n) s += lorem
  return s.slice(0, n)
}

async function main() {
  log(`port ${PORT} · ${ROUNDS} rounds × ${CHARS} chars @ ${CPS} cps`)
  const tgt = await pickRenderer()
  log(`target ${tgt.url}`)
  const cdp = await connect(tgt.webSocketDebuggerUrl)
  await cdp.send('Runtime.enable')
  await cdp.send('Performance.enable')
  await cdp.send('DOM.enable')

  const focused = await focusComposer(cdp)
  if (!focused) {
    console.error('composer not focusable')
    process.exit(2)
  }

  await forceGCAndSettle(cdp)
  const baseline = await snapshotCounts(cdp)
  log('baseline:', JSON.stringify(baseline))

  const text = genText(CHARS)
  const history = [{ round: 0, ...baseline, charsTyped: 0 }]

  for (let r = 1; r <= ROUNDS; r++) {
    await typeChars(cdp, text, CPS)
    await new Promise(res => setTimeout(res, 200))
    await clearComposer(cdp)
    await forceGCAndSettle(cdp)
    const snap = await snapshotCounts(cdp)
    snap.charsTyped = r * CHARS
    snap.round = r
    history.push(snap)
    log(
      `round ${r}: heap=${snap.heapUsedMB.toFixed(1)}MB ` +
        `nodes=${snap.nodes} listeners=${snap.jsListeners} ` +
        `domNodes=${Math.round(snap.docNodes)} ` +
        `layoutCount=${snap.layoutCount} ` +
        `Δheap=+${(snap.heapUsedMB - baseline.heapUsedMB).toFixed(2)}MB ` +
        `Δnodes=+${snap.nodes - baseline.nodes} ` +
        `Δlisteners=+${snap.jsListeners - baseline.jsListeners}`
    )
  }

  console.log('\n=== GROWTH PER ROUND (averaged over last 5 rounds) ===')
  const tail = history.slice(-5)
  const first = tail[0]
  const last = tail[tail.length - 1]
  const rounds = last.round - first.round
  const cells = ['heapUsedMB', 'nodes', 'jsListeners', 'docNodes', 'layoutCount']
  for (const c of cells) {
    const delta = last[c] - first[c]
    const per = delta / Math.max(1, rounds)
    const perChar = delta / Math.max(1, rounds * CHARS)
    console.log(`  ${c.padEnd(16)}  Δtotal=${delta.toFixed(2).padStart(10)}  /round=${per.toFixed(2).padStart(8)}  /char=${perChar.toFixed(4).padStart(8)}`)
  }

  writeFileSync('/tmp/hermes-leak-history.json', JSON.stringify(history, null, 2))
  log('wrote /tmp/hermes-leak-history.json')
  cdp.close()
}

main().catch(e => {
  console.error('[leak] fatal:', e.stack ?? e.message)
  process.exit(1)
})
