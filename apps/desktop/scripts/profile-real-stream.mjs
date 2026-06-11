// CPU-profile during a real LLM stream — confirms or refutes whether the
// synthetic stream's hotspots (Streamdown markdown re-parse, FadeText)
// match real-world content.
//
// Run *after* model is set to something fast + cheap (gpt-4o-mini etc.).
// Sends a prompt likely to produce markdown + a numbered list.

import { writeFileSync } from 'node:fs'

const CDP_HTTP = 'http://127.0.0.1:9222'
const PROMPT = process.env.PROMPT || 'Give me a numbered list of 8 useful bash one-liners. For each: a brief description, then the command in a code block. No preamble.'
const OUT = process.env.OUT || `/tmp/real-stream-${Date.now()}.cpuprofile`
const START_TIMEOUT = Number(process.env.START_TIMEOUT || 45000)
const STREAM_TIMEOUT = Number(process.env.STREAM_TIMEOUT || 60000)

class CDP {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map() }
  static async open(url) {
    const ws = new WebSocket(url)
    await new Promise((r) => ws.addEventListener('open', r, { once: true }))
    const cdp = new CDP(ws)
    ws.addEventListener('message', (ev) => {
      const m = JSON.parse(ev.data.toString())
      if (m.id != null && cdp.pending.has(m.id)) {
        const { resolve, reject } = cdp.pending.get(m.id)
        cdp.pending.delete(m.id)
        if (m.error) reject(new Error(m.error.message))
        else resolve(m.result)
      }
    })
    return cdp
  }
  send(method, params) {
    const id = ++this.id
    return new Promise((res, rej) => {
      this.pending.set(id, { resolve: res, reject: rej })
      this.ws.send(JSON.stringify({ id, method, params }))
    })
  }
  async eval(expr) {
    const r = await this.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval')
    return r.result.value
  }
  close() { this.ws.close() }
}

async function main() {
  const list = await (await fetch(`${CDP_HTTP}/json`)).json()
  const target = list.find((t) => t.type === 'page' && /5174/.test(t.url))
  const cdp = await CDP.open(target.webSocketDebuggerUrl)

  const baseCount = await cdp.eval('document.querySelectorAll("[data-slot=aui_assistant-message-root]").length')

  // Submit prompt
  await cdp.eval(`(() => {
    const ed = document.querySelector('[contenteditable="true"]')
    ed.focus()
    document.execCommand('insertText', false, ${JSON.stringify(PROMPT)})
    ed.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', which: 13, keyCode: 13, bubbles: true, cancelable: true }))
    return 'submitted'
  })()`)

  // Wait for real stream start (assistant count grows).
  const submitT0 = Date.now()
  let streamT = null
  for (let i = 0; i < START_TIMEOUT / 50; i++) {
    await new Promise((r) => setTimeout(r, 50))
    const n = await cdp.eval('document.querySelectorAll("[data-slot=aui_assistant-message-root]").length')
    if (n > baseCount) { streamT = Date.now(); break }
  }
  if (!streamT) {
    console.error('stream never started within', START_TIMEOUT, 'ms')
    cdp.close()
    process.exit(2)
  }
  console.log('REAL stream started after', streamT - submitT0, 'ms — starting CPU profile NOW')

  // Start CPU profile NOW, only during stream phase.
  await cdp.send('Profiler.enable')
  await cdp.send('Profiler.setSamplingInterval', { interval: 100 })
  await cdp.send('Profiler.start')

  // Wait until busy goes false + grace, or timeout.
  const cutoff = Date.now() + STREAM_TIMEOUT
  while (Date.now() < cutoff) {
    await new Promise((r) => setTimeout(r, 500))
    const busy = await cdp.eval('!!document.querySelector("[data-status=running], [data-busy=true]")')
    if (!busy) {
      await new Promise((r) => setTimeout(r, 500))
      break
    }
  }

  const { profile } = await cdp.send('Profiler.stop')
  writeFileSync(OUT, JSON.stringify(profile))
  console.log('wrote', OUT)

  const samples = profile.samples || []
  const timeDeltas = profile.timeDeltas || []
  const nodes = new Map(profile.nodes.map((n) => [n.id, n]))
  const selfTime = new Map()
  for (let i = 0; i < samples.length; i++) {
    const id = samples[i]
    const dt = timeDeltas[i] ?? 0
    selfTime.set(id, (selfTime.get(id) || 0) + dt)
  }
  const ranked = [...selfTime.entries()]
    .map(([id, us]) => {
      const n = nodes.get(id)
      const cf = n?.callFrame || {}
      return {
        ms: us / 1000,
        name: cf.functionName || '(anonymous)',
        url: (cf.url || '').slice(-60),
        line: cf.lineNumber
      }
    })
    .filter((x) => !/\(root\)|\(idle\)|\(garbage collector\)|\(program\)/.test(x.name))
    .sort((a, b) => b.ms - a.ms)
    .slice(0, 25)

  const finalText = await cdp.eval(`(() => {
    const all = document.querySelectorAll('[data-slot="aui_assistant-message-root"]')
    return all.length ? all[all.length-1].textContent.length : 0
  })()`)
  console.log('\nfinal assistant message length:', finalText, 'chars')

  console.log('\n=== TOP 25 SELF TIME (ms) DURING REAL STREAM ===')
  for (const r of ranked) {
    console.log(`${r.ms.toFixed(1).padStart(7)}  ${r.name.padEnd(40)}  ${r.url}:${r.line}`)
  }

  cdp.close()
}

main().catch((e) => { console.error(e); process.exit(1) })
