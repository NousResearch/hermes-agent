#!/usr/bin/env node
// Robust A/B: measure each surface N times, report median wasted renders.
// Reduces timing noise from stream ticks landing on different clicks.
import WebSocket from 'ws'

const RUNS = 3
let msgId = 1
function send(ws, method, params = {}) {
  const id = msgId++
  return new Promise((resolve, reject) => {
    const h = (data) => { const m = JSON.parse(data); if (m.id === id) { ws.off('message', h); m.error ? reject(m.error) : resolve(m.result) } }
    ws.on('message', h); ws.send(JSON.stringify({ id, method, params }))
  })
}
async function ev(ws, e) { return (await send(ws, 'Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true }))?.result?.value }
const sleep = ms => new Promise(r => setTimeout(r, ms))
const median = arr => { const s = [...arr].sort((a,b)=>a-b); return s[Math.floor(s.length/2)] }

async function wsUrl() {
  const d = await (await fetch('http://127.0.0.1:9222/json/list')).json()
  return d.find(t => t.type === 'page' && (t.url||'').includes('5174')).webSocketDebuggerUrl
}

async function measureOnce(ws, setup, holdMs) {
  await setup(ws)
  await sleep(300)
  await ev(ws, `__RENDER_COUNTS__.start(); true`)
  await sleep(holdMs)
  const rep = JSON.parse(await ev(ws, `JSON.stringify(__RENDER_COUNTS__.report())`))
  return rep.reduce((s, c) => s + c.wasted, 0)
}

async function main() {
  const ws = new WebSocket(await wsUrl())
  await new Promise((res, rej) => { ws.on('open', res); ws.on('error', rej) })
  await send(ws, 'Runtime.enable')

  const surfaces = [
    ['Artifacts', ws => ev(ws, `window.location.hash='#/artifacts'; true`), 2000],
    ['Messaging', ws => ev(ws, `window.location.hash='#/messaging'; true`), 2000],
    ['Cron', ws => ev(ws, `window.location.hash='#/cron'; true`), 2000],
    ['Profiles', ws => ev(ws, `window.location.hash='#/profiles'; true`), 2000],
    ['Agents', ws => ev(ws, `window.location.hash='#/agents'; true`), 2000],
    ['Starmap', ws => ev(ws, `window.location.hash='#/starmap'; true`), 2000],
    ['Webhooks', ws => ev(ws, `window.location.hash='#/webhooks'; true`), 2000],
    ['CommandCenter/System', async ws => { await ev(ws, `window.location.hash='#/command-center'; true`); await sleep(400); await ev(ws, `Array.from(document.querySelectorAll('button')).find(b=>b.textContent?.trim()==='System')?.click(); true`) }, 2000],
  ]

  const label = process.argv[2] || 'RUN'
  console.log(`\n=== ${label} (median of ${RUNS}, wasted renders, 2s idle) ===`)
  for (const [name, setup, hold] of surfaces) {
    const runs = []
    for (let i = 0; i < RUNS; i++) {
      runs.push(await measureOnce(ws, setup, hold))
      await ev(ws, `window.location.hash='#/'; true`); await sleep(300)
    }
    console.log(`  ${name.padEnd(24)} median ${String(median(runs)).padStart(6)}  (runs: ${runs.join(', ')})`)
  }
  ws.close()
}
main().catch(e => { console.error(e); process.exit(1) })
