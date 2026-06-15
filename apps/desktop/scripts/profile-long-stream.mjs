#!/usr/bin/env node
// Long-running stream profile + frame-rate timeline. Submits a prompt that
// asks for ~30 paragraphs of output, then captures both a CPU profile and
// a per-100ms frame counter so we can see if FPS sags as the message grows.

import { writeFileSync } from 'node:fs'

const args = Object.fromEntries(
  process.argv.slice(2).flatMap(s => {
    const m = s.match(/^--([^=]+)(?:=(.*))?$/)
    return m ? [[m[1], m[2] ?? true]] : []
  })
)
const PORT = Number(args.port ?? 9222)
const OUT = String(args.out ?? `/tmp/hermes-long-stream-${Date.now()}`)
const STREAM_SEC = Number(args.seconds ?? 25)

async function pickRenderer() {
  const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
  return list.find(t => t.type === 'page' && t.url.startsWith('http'))
}

function connect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url)
    let id = 0
    const pending = new Map()
    ws.addEventListener('open', () =>
      resolve({
        send(method, params = {}) {
          const myId = ++id
          ws.send(JSON.stringify({ id: myId, method, params }))
          return new Promise((res, rej) => pending.set(myId, { res, rej }))
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
      }
    })
  })
}

async function evalP(cdp, expr) {
  const r = await cdp.send('Runtime.evaluate', { expression: expr, returnByValue: true })
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text)
  return r.result.value
}

async function main() {
  const tgt = await pickRenderer()
  console.log('target', tgt.url)
  const cdp = await connect(tgt.webSocketDebuggerUrl)
  await cdp.send('Runtime.enable')
  await cdp.send('Profiler.enable')
  await cdp.send('Performance.enable')

  // Submit a long-form prompt
  await evalP(
    cdp,
    `(() => {
      const el = document.querySelector('[data-slot="composer-rich-input"]')
      el.focus()
      const r = document.createRange(); r.selectNodeContents(el); r.collapse(false)
      window.getSelection().removeAllRanges(); window.getSelection().addRange(r)
    })()`
  )
  const prompt = 'write 15 paragraphs about gpu memory bandwidth, memory hierarchies, roofline model, and how modern transformer inference benefits from these. include diagrams in ascii where relevant. no code. fully detailed.'
  for (const c of prompt) {
    await cdp.send('Input.dispatchKeyEvent', { type: 'char', text: c, unmodifiedText: c })
    await new Promise(r => setTimeout(r, 5))
  }
  await new Promise(r => setTimeout(r, 200))
  await cdp.send('Input.dispatchKeyEvent', {
    type: 'rawKeyDown', windowsVirtualKeyCode: 13, key: 'Enter', code: 'Enter', text: '\r', unmodifiedText: '\r'
  })
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', windowsVirtualKeyCode: 13, key: 'Enter', code: 'Enter' })

  console.log('waiting for assistant…')
  let streaming = false
  for (let i = 0; i < 100; i++) {
    const c = await evalP(cdp, `document.querySelectorAll('[data-slot="aui_assistant-message-root"]').length`)
    if (c > 0) { streaming = true; break }
    await new Promise(r => setTimeout(r, 100))
  }
  if (!streaming) {
    console.error('no assistant message')
    cdp.close()
    return
  }

  // Install a per-rAF frame counter
  await evalP(
    cdp,
    `(() => {
      window.__fpsSamples = []
      window.__fpsT0 = performance.now()
      window.__fpsLast = performance.now()
      window.__fpsFrameCount = 0
      window.__fpsHistogram = []  // {t, fps, contentLen}
      const tick = () => {
        const now = performance.now()
        const dt = now - window.__fpsLast
        window.__fpsLast = now
        window.__fpsFrameCount++
        window.__fpsSamples.push({ t: now - window.__fpsT0, dt })
        if (performance.now() - window.__fpsT0 < ${STREAM_SEC * 1000}) {
          requestAnimationFrame(tick)
        }
      }
      requestAnimationFrame(tick)
      // Bucket fps every 500ms
      window.__fpsBucket = setInterval(() => {
        const now = performance.now()
        const recentCount = window.__fpsSamples.filter(s => now - window.__fpsT0 - s.t < 500).length
        const root = document.querySelector('[data-slot="aui_thread-content"]')
        const len = root ? root.innerText.length : 0
        const v = document.querySelector('[data-slot="aui_thread-viewport"]')
        window.__fpsHistogram.push({
          t: now - window.__fpsT0,
          frames500ms: recentCount,
          fps: recentCount * 2,
          contentLen: len,
          scrollTop: v?.scrollTop ?? 0,
          scrollHeight: v?.scrollHeight ?? 0
        })
      }, 500)
    })()`
  )

  // Start CPU profile
  await cdp.send('Profiler.setSamplingInterval', { interval: 1000 })
  await cdp.send('Profiler.start')

  await new Promise(r => setTimeout(r, STREAM_SEC * 1000))

  const { profile } = await cdp.send('Profiler.stop')
  await evalP(cdp, `clearInterval(window.__fpsBucket)`)

  writeFileSync(`${OUT}.cpuprofile`, JSON.stringify(profile))
  console.log(`cpu profile → ${OUT}.cpuprofile`)

  // Pull fps histogram
  const hist = JSON.parse(await evalP(cdp, `JSON.stringify(window.__fpsHistogram || [])`))
  writeFileSync(`${OUT}.fps.json`, JSON.stringify(hist, null, 2))

  console.log(`\n=== FPS over time ===`)
  console.log(`  t(s)   fps   contentLen   scrollTop/scrollHeight`)
  for (const h of hist) {
    const bar = '█'.repeat(Math.min(40, Math.max(0, Math.round(h.fps / 2))))
    console.log(`  ${(h.t / 1000).toFixed(1).padStart(5)}  ${String(h.fps).padStart(3)}  ${String(h.contentLen).padStart(10)}   ${h.scrollTop}/${h.scrollHeight}  ${bar}`)
  }

  // Top self frames
  const total = (profile.endTime - profile.startTime) / 1000
  const intMs = total / Math.max(1, profile.samples?.length ?? 1)
  const counts = new Map()
  for (const s of profile.samples ?? []) counts.set(s, (counts.get(s) ?? 0) + 1)
  const rows = profile.nodes
    .map(n => ({ id: n.id, fn: n.callFrame.functionName || '(anon)', url: n.callFrame.url || '', line: n.callFrame.lineNumber, self: counts.get(n.id) ?? 0 }))
    .sort((a, b) => b.self - a.self)
    .slice(0, 25)
  console.log(`\n=== ${total.toFixed(0)}ms wall, ${profile.samples?.length ?? 0} samples (${intMs.toFixed(2)}ms each) ===`)
  for (const r of rows) {
    if (r.self === 0) break
    const url = r.url.replace(/^.*\/src\//, 'src/').replace(/\?.*$/, '').slice(0, 70)
    console.log(`  ${(r.self * intMs).toFixed(1).padStart(7)}ms  (${String(r.self).padStart(4)} samp)  ${r.fn.padEnd(45)} ${url}:${r.line}`)
  }

  await evalP(cdp, `
    (() => {
      for (const b of document.querySelectorAll('button')) {
        if ((b.getAttribute('aria-label') || '').toLowerCase().includes('stop')) { b.click(); return }
      }
    })()
  `)

  cdp.close()
}

main().catch(e => {
  console.error('fatal:', e.stack ?? e.message)
  process.exit(1)
})
