#!/usr/bin/env node
// Comprehensive overlay render-churn measurement on the live instance.
// Measures: cmdk root + submenus, settings, command center, capabilities, system overlays.
// Read-only — never closes the app, only navigates and clicks.
import WebSocket from 'ws'

const WS_URL = 'ws://127.0.0.1:9222/devtools/page/6E095DBE024BD280C674D00023C01201'
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

async function measure(ws, label, fn, settleMs = 400) {
  await eval_(ws, `__RENDER_COUNTS__.start(); true`)
  await fn(ws)
  await sleep(settleMs)
  const report = await eval_(ws, `JSON.stringify(__RENDER_COUNTS__.report())`)
  const parsed = JSON.parse(report)
  const totalRenders = parsed.reduce((s, c) => s + c.renders, 0)
  const totalWasted = parsed.reduce((s, c) => s + c.wasted, 0)
  const totalMs = parsed.reduce((s, c) => s + c.totalMs, 0)
  const top5 = parsed.filter(c => c.wasted > 0).sort((a,b) => b.wasted - a.wasted).slice(0, 5)
  console.log(`\n${label}`)
  console.log(`  total: ${totalRenders} renders, ${totalWasted} wasted, ${totalMs.toFixed(1)}ms`)
  if (top5.length > 0) {
    for (const c of top5) {
      console.log(`    ${c.name}: ${c.renders} renders, ${c.wasted} wasted, ${c.totalMs.toFixed(1)}ms`)
    }
  } else {
    console.log(`    (no wasted renders)`)
  }
  return { label, totalRenders, totalWasted, totalMs, top: top5 }
}

async function clickByText(ws, text, tag = 'button') {
  return eval_(ws, `
    const el = Array.from(document.querySelectorAll('${tag}')).find(b => b.textContent?.trim() === '${text}')
    if (el) { el.click(); true } else false
  `)
}

async function clickByHref(ws, partial) {
  return eval_(ws, `
    const el = Array.from(document.querySelectorAll('a,button')).find(e => e.getAttribute('href')?.includes('${partial}'))
    if (el) { el.click(); true } else false
  `)
}

async function typeInInput(ws, text) {
  for (const ch of text) {
    await eval_(ws, `
      const el = document.querySelector('[cmdk-input]') || document.querySelector('input[type="text"]') || document.querySelector('input')
      if (el) { el.focus(); el.value = el.value + '${ch}'; el.dispatchEvent(new Event('input', {bubbles:true})) }
      true
    `)
    await sleep(50)
  }
}

async function pressKey(ws, key, mods = {}) {
  await eval_(ws, `
    document.dispatchEvent(new KeyboardEvent('keydown', { key: '${key}', ${Object.entries(mods).map(([k,v]) => `${k}:${v}`).join(', ')}, bubbles: true }))
    true
  `)
}

async function main() {
  const ws = new WebSocket(WS_URL)
  await new Promise((resolve, reject) => {
    ws.on('open', resolve)
    ws.on('error', reject)
  })
  await send(ws, 'Runtime.enable')

  const avail = await eval_(ws, `typeof __RENDER_COUNTS__`)
  console.log(`__RENDER_COUNTS__: ${avail}`)
  if (avail !== 'object') {
    console.error('Render counter not loaded. Reload the page.')
    process.exit(1)
  }

  const results = []

  // === 1. CmdK root: open, type "setting", close ===
  results.push(await measure(ws, '1. CmdK root (type "setting")', async (ws) => {
    await pressKey(ws, 'k', { metaKey: true })
    await sleep(300)
    await typeInInput(ws, 'setting')
  }))
  await pressKey(ws, 'Escape')
  await sleep(300)

  // === 2. CmdK submenu: open, navigate to theme picker, click a theme, back ===
  results.push(await measure(ws, '2. CmdK theme submenu', async (ws) => {
    await pressKey(ws, 'k', { metaKey: true })
    await sleep(300)
    await typeInInput(ws, 'theme')
    await sleep(200)
    // Click first theme item
    await eval_(ws, `
      const item = document.querySelector('[cmdk-item]')
      if (item) item.click()
      true
    `)
    await sleep(300)
  }))
  await pressKey(ws, 'Escape')
  await sleep(300)

  // === 3. CmdK color-mode submenu ===
  results.push(await measure(ws, '3. CmdK color-mode submenu', async (ws) => {
    await pressKey(ws, 'k', { metaKey: true })
    await sleep(300)
    await typeInInput(ws, 'color mode')
    await sleep(200)
    await eval_(ws, `const item = document.querySelector('[cmdk-item]'); if (item) item.click(); true`)
    await sleep(300)
  }))
  await pressKey(ws, 'Escape')
  await sleep(300)

  // === 4. Settings: open, click through nav items ===
  results.push(await measure(ws, '4. Settings (5 nav clicks)', async (ws) => {
    await clickByHref(ws, 'settings')
    await sleep(500)
    for (const label of ['Appearance', 'Gateway', 'Keys', 'Notifications', 'About']) {
      await clickByText(ws, label)
      await sleep(100)
    }
  }))

  // === 5. Command Center: open to System tab, idle 2s ===
  results.push(await measure(ws, '5. Command Center (System tab, 2s idle)', async (ws) => {
    await clickByHref(ws, 'command-center')
    await sleep(500)
    await clickByText(ws, 'System')
    await sleep(2000)
  }, 100))

  // === 6. Command Center: Sessions tab ===
  results.push(await measure(ws, '6. Command Center (Sessions tab, 1s)', async (ws) => {
    await clickByText(ws, 'Sessions')
    await sleep(1000)
  }, 100))

  // === 7. Capabilities/Skills page ===
  results.push(await measure(ws, '7. Capabilities (Skills tab, 1s)', async (ws) => {
    await clickByHref(ws, 'skills')
    await sleep(800)
    // Click Skills tab if not already
    await clickByText(ws, 'Skills')
    await sleep(1000)
  }, 100))

  // === 8. Capabilities Toolsets tab ===
  results.push(await measure(ws, '8. Capabilities (Toolsets tab, 1s)', async (ws) => {
    await clickByText(ws, 'Tools')
    await sleep(1000)
  }, 100))

  // === 9. Capabilities MCP tab ===
  results.push(await measure(ws, '9. Capabilities (MCP tab, 1s)', async (ws) => {
    await clickByText(ws, 'MCP')
    await sleep(1000)
  }, 100))

  // Navigate back to chat
  await eval_(ws, `window.location.hash = '#/'; true`)
  await sleep(300)

  // Summary table
  console.log('\n\n=== SUMMARY ===')
  console.log('Surface'.padEnd(45) + 'Renders'.padStart(10) + 'Wasted'.padStart(10) + 'ms'.padStart(10))
  console.log('-'.repeat(75))
  for (const r of results) {
    console.log(r.label.padEnd(45) + String(r.totalRenders).padStart(10) + String(r.totalWasted).padStart(10) + r.totalMs.toFixed(1).padStart(10))
  }

  ws.close()
}

main().catch(err => { console.error(err); process.exit(1) })
