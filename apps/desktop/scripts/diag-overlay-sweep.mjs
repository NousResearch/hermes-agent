#!/usr/bin/env node
// Full surface sweep: every overlay, every settings sub-page, every system overlay.
// Read-only — navigates and clicks, never closes the app.
import WebSocket from 'ws'

const WS_URL = process.env.CDP_WS || 'ws://127.0.0.1:9222/devtools/page/6E095DBE024BD280C674D00023C01201'
let msgId = 1

function send(ws, method, params = {}) {
  const id = msgId++
  return new Promise((resolve, reject) => {
    const handler = (data) => {
      const msg = JSON.parse(data.toString())
      if (msg.id === id) {
        ws.off('message', handler)
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result)
      }
    }
    ws.on('message', handler)
    ws.send(JSON.stringify({ id, method, params }))
  })
}
async function eval_(ws, expr) {
  const r = await send(ws, 'Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })
  return r?.result?.value
}
async function sleep(ms) { return new Promise(r => setTimeout(r, ms)) }

async function measure(ws, label, fn, settleMs = 300) {
  await eval_(ws, `__RENDER_COUNTS__.start(); true`)
  await fn(ws)
  await sleep(settleMs)
  const report = JSON.parse(await eval_(ws, `JSON.stringify(__RENDER_COUNTS__.report())`))
  const totalRenders = report.reduce((s, c) => s + c.renders, 0)
  const totalWasted = report.reduce((s, c) => s + c.wasted, 0)
  const totalMs = report.reduce((s, c) => s + c.totalMs, 0)
  const top = report.filter(c => c.wasted > 0).sort((a,b) => b.wasted - a.wasted).slice(0, 6)
  console.log(`\n${label}`)
  console.log(`  ${totalRenders} renders · ${totalWasted} wasted · ${totalMs.toFixed(1)}ms`)
  for (const c of top) console.log(`    ${c.name}: ${c.renders}r / ${c.wasted}w / ${c.totalMs.toFixed(1)}ms`)
  if (top.length === 0) console.log(`    (clean)`)
  return { label, totalRenders, totalWasted, totalMs }
}

async function nav(ws, hash) {
  await eval_(ws, `window.location.hash = '#${hash}'; true`)
}
async function clickText(ws, text, tag='button') {
  return eval_(ws, `
    const el = Array.from(document.querySelectorAll('${tag}')).find(b => b.textContent?.trim() === ${JSON.stringify(text)})
    if (el) { el.click(); true } else false
  `)
}

async function main() {
  const ws = new WebSocket(WS_URL)
  await new Promise((res, rej) => { ws.on('open', res); ws.on('error', rej) })
  await send(ws, 'Runtime.enable')
  if (await eval_(ws, `typeof __RENDER_COUNTS__`) !== 'object') { console.error('counter not loaded'); process.exit(1) }

  const results = []

  // System overlays — open each, idle 2s (streaming underneath is the churn test)
  for (const [label, route] of [
    ['Artifacts', '/artifacts'],
    ['Messaging', '/messaging'],
    ['Cron', '/cron'],
    ['Profiles', '/profiles'],
    ['Agents', '/agents'],
    ['Starmap', '/starmap'],
    ['Webhooks', '/webhooks'],
  ]) {
    results.push(await measure(ws, `OVERLAY: ${label} (open + 2s idle)`, async (ws) => {
      await nav(ws, route)
      await sleep(2200)
    }, 100))
    await nav(ws, '/')
    await sleep(300)
  }

  // Settings sub-pages — open settings, click each nav item, idle 1.5s on it
  await nav(ws, '/settings')
  await sleep(600)
  for (const label of ['Model','Session','Appearance','Notifications','Providers','Gateway','Keybinds','API Keys','Plugins','Archived Chats','About']) {
    results.push(await measure(ws, `SETTINGS: ${label} (1.5s idle)`, async (ws) => {
      await clickText(ws, label)
      await sleep(1500)
    }, 100))
  }
  await nav(ws, '/')
  await sleep(300)

  // Messaging sub-tabs (platform detail) — click through if present
  await nav(ws, '/messaging')
  await sleep(800)
  results.push(await measure(ws, 'MESSAGING: idle 2s w/ platform list', async (ws) => {
    await sleep(2000)
  }, 100))
  await nav(ws, '/')
  await sleep(300)

  console.log('\n\n=== SUMMARY (wasted renders) ===')
  console.log('Surface'.padEnd(48) + 'Renders'.padStart(9) + 'Wasted'.padStart(9) + 'ms'.padStart(9))
  console.log('-'.repeat(75))
  for (const r of results) {
    console.log(r.label.padEnd(48) + String(r.totalRenders).padStart(9) + String(r.totalWasted).padStart(9) + r.totalMs.toFixed(1).padStart(9))
  }
  ws.close()
}
main().catch(e => { console.error(e); process.exit(1) })
