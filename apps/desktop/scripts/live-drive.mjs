// Live-drive harness for the REAL hgui instance on :9222.
//
//   node scripts/live-drive.mjs status          — targets, session count, perf-live armed?
//   node scripts/live-drive.mjs fps [seconds]   — raw rAF fps over N seconds (default 4)
//   node scripts/live-drive.mjs drag            — drag the sidebar sash, report fps + LoAF
//   node scripts/live-drive.mjs type            — type into composer, report fps + LoAF
//   node scripts/live-drive.mjs switch          — cycle through sidebar sessions, per-switch ms
//   node scripts/live-drive.mjs send "msg"      — submit a prompt in the focused session
//   node scripts/live-drive.mjs eval "expr"     — arbitrary page eval
//
// Attaches to the page target directly (no perf-harness deps) so it works on
// the app Brooklyn actually runs, with her profile, her sessions, her layout.

import { WebSocket } from 'ws'

const PORT = Number(process.env.CDP_PORT ?? 9222)

async function attach() {
  const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
  const page = list.find(t => t.type === 'page' && !/devtools/.test(t.url))

  if (!page) {
    throw new Error('no page target on :' + PORT)
  }

  const ws = new WebSocket(page.webSocketDebuggerUrl, { maxPayload: 256 * 1024 * 1024 })
  await new Promise((resolve, reject) => {
    ws.once('open', resolve)
    ws.once('error', reject)
  })

  let id = 0
  const pending = new Map()
  ws.on('message', raw => {
    const msg = JSON.parse(raw)

    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id)
      pending.delete(msg.id)
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result)
    }
  })

  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const mid = ++id
      pending.set(mid, { resolve, reject })
      ws.send(JSON.stringify({ id: mid, method, params }))
    })

  await send('Runtime.enable')

  const evaluate = async expression => {
    const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })

    if (r.exceptionDetails) {
      throw new Error(r.exceptionDetails.exception?.description ?? 'eval failed')
    }

    return r.result?.value
  }

  return { evaluate, close: () => ws.close(), send }
}

const FPS = seconds => `
  (async () => {
    const frames = []
    let last = performance.now()
    const end = last + ${seconds * 1000}
    while (performance.now() < end) {
      await new Promise(r => requestAnimationFrame(r))
      const now = performance.now()
      frames.push(now - last)
      last = now
    }
    const total = frames.reduce((a, b) => a + b, 0)
    const sorted = [...frames].sort((a, b) => a - b)
    const pct = p => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))]
    return {
      fps: Math.round((frames.length / total) * 1000 * 10) / 10,
      p95: Math.round(pct(0.95) * 10) / 10,
      worst: Math.round(sorted[sorted.length - 1] * 10) / 10,
      slow33: frames.filter(f => f > 33).length,
      n: frames.length
    }
  })()
`

// LoAF recorder for a window of work driven inside `body`.
const WITH_LOAF = body => `
  (async () => {
    const lofs = []
    const po = new PerformanceObserver(list => {
      for (const e of list.getEntries()) {
        lofs.push({
          ms: Math.round(e.duration),
          block: Math.round(e.blockingDuration ?? 0),
          style: e.styleAndLayoutStart ? Math.round(e.startTime + e.duration - e.styleAndLayoutStart) : 0,
          scripts: (e.scripts ?? []).filter(s => s.duration >= 5).map(s =>
            (s.invokerType ?? '') + ':' + (s.invoker ?? s.sourceFunctionName ?? '?') + '@' +
            ((s.sourceURL ?? '').split('/').pop() ?? '') + ' ' + Math.round(s.duration) + 'ms')
        })
      }
    })
    po.observe({ type: 'long-animation-frame', buffered: false })
    const frames = []
    let last = performance.now()
    let stop = false
    const tick = () => {
      if (stop) return
      const now = performance.now()
      frames.push(now - last)
      last = now
      requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
    ${body}
    stop = true
    po.disconnect()
    const total = frames.reduce((a, b) => a + b, 0)
    const sorted = [...frames].sort((a, b) => a - b)
    const pct = p => sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] : 0
    return {
      fps: total ? Math.round((frames.length / total) * 1000 * 10) / 10 : 0,
      p95: Math.round(pct(0.95) * 10) / 10,
      worst: sorted.length ? Math.round(sorted[sorted.length - 1] * 10) / 10 : 0,
      slow33: frames.filter(f => f > 33).length,
      longFrames: lofs.slice(0, 10)
    }
  })()
`

const DRAG_BODY = `
  const handle = document.querySelector('[role="separator"]')
  if (!handle) throw new Error('no sash')
  const box = handle.getBoundingClientRect()
  const y = box.top + box.height / 2
  const x0 = box.left + box.width / 2
  let x = x0
  const opts = { bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse', isPrimary: true, button: 0, buttons: 1 }
  handle.dispatchEvent(new PointerEvent('pointerdown', { ...opts, clientX: x, clientY: y }))
  for (let i = 0; i < 60; i++) {
    x += (i < 30 ? 3 : -3)
    window.dispatchEvent(new PointerEvent('pointermove', { ...opts, clientX: x, clientY: y }))
    await new Promise(r => requestAnimationFrame(r))
  }
  window.dispatchEvent(new PointerEvent('pointerup', { ...opts, buttons: 0, clientX: x, clientY: y }))
`

const TYPE_BODY = `
  const el = [...document.querySelectorAll('[contenteditable="true"]')].find(e => !e.closest('[data-pane-hidden]'))
  if (!el) throw new Error('no composer')
  el.focus()
  for (let i = 0; i < 40; i++) {
    const ch = 'the quick brown fox '[i % 20]
    el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: ch }))
    el.textContent += ch
    el.dispatchEvent(new InputEvent('input', { bubbles: true, data: ch, inputType: 'insertText' }))
    el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: ch }))
    await new Promise(r => requestAnimationFrame(r))
  }
`

// Session switching: click each sidebar session row, time route->settled.
const SWITCH = `
  (async () => {
    const pick = () => [...document.querySelectorAll('[data-slot="row-button"]')]
      .filter(el => el.offsetParent !== null).slice(0, 8)
    if (pick().length < 2) {
      throw new Error('found ' + pick().length + ' session rows')
    }
    const times = []
    const labels = []
    for (let i = 0; i < Math.min(pick().length, 6); i++) {
      // Re-query each iteration: a switch can re-render the sidebar and
      // detach the previously captured nodes.
      const row = pick()[i]
      if (!row) break
      labels.push((row.textContent || '').slice(0, 24))
      const t0 = performance.now()
      row.click()
      let calm = 0
      while (calm < 2 && performance.now() - t0 < 5000) {
        const f0 = performance.now()
        await new Promise(r => requestAnimationFrame(r))
        const dt = performance.now() - f0
        calm = dt < 20 ? calm + 1 : 0
      }
      times.push(Math.round(performance.now() - t0))
      await new Promise(r => setTimeout(r, 400))
    }
    return { switches: times, labels, avg: Math.round(times.reduce((a, b) => a + b, 0) / times.length) }
  })()
`

const cmd = process.argv[2] ?? 'status'
const arg = process.argv[3]
const { evaluate, close } = await attach()

try {
  if (cmd === 'status') {
    const r = await evaluate(`JSON.stringify({
      url: location.hash,
      perfLive: typeof window.__PERF_LIVE__ !== 'undefined',
      renderCounts: typeof window.__RENDER_COUNTS__ !== 'undefined',
      tiles: document.querySelectorAll('[data-tree-group]').length,
      sessions: document.querySelectorAll('[data-slot*="session-row"], [data-session-row]').length,
      composers: [...document.querySelectorAll('[contenteditable="true"]')].length
    })`)
    console.log(r)
  } else if (cmd === 'fps') {
    console.log(JSON.stringify(await evaluate(FPS(Number(arg ?? 4)))))
  } else if (cmd === 'drag') {
    const r = await evaluate(WITH_LOAF(DRAG_BODY))
    console.log('drag', JSON.stringify({ fps: r.fps, p95: r.p95, worst: r.worst, slow33: r.slow33 }))
    for (const lf of r.longFrames) {
      console.log(`  ⏱ ${lf.ms}ms block=${lf.block} style=${lf.style} → ${lf.scripts.join(' | ') || '(no script ≥5ms)'}`)
    }
  } else if (cmd === 'type') {
    const r = await evaluate(WITH_LOAF(TYPE_BODY))
    console.log('type', JSON.stringify({ fps: r.fps, p95: r.p95, worst: r.worst, slow33: r.slow33 }))
    for (const lf of r.longFrames) {
      console.log(`  ⏱ ${lf.ms}ms block=${lf.block} style=${lf.style} → ${lf.scripts.join(' | ') || '(no script ≥5ms)'}`)
    }
  } else if (cmd === 'switch') {
    console.log(JSON.stringify(await evaluate(SWITCH)))
  } else if (cmd === 'eval') {
    console.log(JSON.stringify(await evaluate(arg)))
  } else if (cmd === 'profile') {
    // CPU-profile one session switch via CDP Profiler (Document Policy blocks
    // the in-page Profiler API, CDP is exempt). arg = row label prefix.
    await send('Profiler.enable')
    await send('Profiler.setSamplingInterval', { interval: 200 })
    await send('Profiler.start')
    const r = await evaluate(`
      (async () => {
        const pick = () => [...document.querySelectorAll('[data-slot="row-button"]')].filter(el => el.offsetParent !== null)
        const row = pick().find(el => (el.textContent || '').startsWith(${JSON.stringify(arg ?? 'GUI')}))
        if (!row) return 'row not found'
        const t0 = performance.now()
        row.click()
        let calm = 0
        while (calm < 3 && performance.now() - t0 < 6000) {
          const f0 = performance.now()
          await new Promise(r => requestAnimationFrame(r))
          calm = (performance.now() - f0) < 20 ? calm + 1 : 0
        }
        return Math.round(performance.now() - t0)
      })()
    `)
    const { profile } = await send('Profiler.stop')
    // Self-time per function.
    const hitById = new Map()
    for (let i = 0; i < profile.samples.length; i++) {
      const id = profile.samples[i]
      const dt = profile.timeDeltas[i] ?? 0
      hitById.set(id, (hitById.get(id) ?? 0) + dt)
    }
    const rows = []
    for (const node of profile.nodes) {
      const us = hitById.get(node.id)
      if (!us || us < 5000) continue
      const f = node.callFrame
      rows.push([Math.round(us / 1000), `${f.functionName || '(anon)'} @ ${(f.url || '').split('/').pop()}:${f.lineNumber}`])
    }
    rows.sort((a, b) => b[0] - a[0])
    console.log('switch took', r, 'ms — top self-time:')
    for (const [ms, name] of rows.slice(0, 18)) {
      console.log(`  ${String(ms).padStart(6)}ms  ${name}`)
    }
  } else if (cmd === 'send') {
    const r = await evaluate(`
      (async () => {
        const el = [...document.querySelectorAll('[contenteditable="true"]')].find(e => !e.closest('[data-pane-hidden]'))
        if (!el) return 'no composer'
        el.focus()
        document.execCommand('insertText', false, ${JSON.stringify(arg ?? 'hello')})
        await new Promise(r => setTimeout(r, 120))
        el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Enter' }))
        return 'sent'
      })()
    `)
    console.log(r)
  } else {
    console.log('unknown cmd', cmd)
  }
} finally {
  close()
}
