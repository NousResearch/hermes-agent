// CPU-profile a synthetic stream — outputs a .cpuprofile and a top-self ranking.
// Open the .cpuprofile in Chrome DevTools Performance panel for a flamegraph.

import { writeFileSync } from 'node:fs'

const CDP_HTTP = 'http://127.0.0.1:9222'
const TOKENS = Number(process.env.TOKENS || 400)
const INTERVAL_MS = Number(process.env.INTERVAL_MS || 8)
const CHUNK = process.env.CHUNK || '**word** in _italic_ with `code` '
const LABEL = process.env.LABEL || 'profile'
const OUT = process.env.OUT || `synth-${LABEL}.cpuprofile`

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

  if (!await cdp.eval('!!window.__PERF_DRIVE__')) {
    console.error('no __PERF_DRIVE__')
    cdp.close()
    process.exit(2)
  }

  await cdp.send('Profiler.enable')
  // High-resolution sampling: 100us
  await cdp.send('Profiler.setSamplingInterval', { interval: 100 })
  await cdp.send('Profiler.start')

  await cdp.eval(`window.__PERF_DRIVE__.stream({ chunk: ${JSON.stringify(CHUNK)}, intervalMs: ${INTERVAL_MS}, totalTokens: ${TOKENS} })`)
  await new Promise((r) => setTimeout(r, TOKENS * INTERVAL_MS + 1500))
  await cdp.eval('window.__PERF_DRIVE__.reset()')

  const { profile } = await cdp.send('Profiler.stop')
  writeFileSync(OUT, JSON.stringify(profile))
  console.log('wrote', OUT)

  // Compute top self time per function.
  const samples = profile.samples || []
  const timeDeltas = profile.timeDeltas || []
  const nodes = new Map(profile.nodes.map((n) => [n.id, n]))
  const selfTime = new Map() // id -> microseconds
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
        us,
        ms: us / 1000,
        name: cf.functionName || '(anonymous)',
        url: (cf.url || '').slice(-60),
        line: cf.lineNumber
      }
    })
    .filter((x) => !/\(root\)|\(idle\)|\(garbage collector\)|\(program\)/.test(x.name))
    .sort((a, b) => b.us - a.us)
    .slice(0, 30)

  console.log('\n=== TOP 30 SELF TIME (ms) ===')
  for (const r of ranked) {
    console.log(`${r.ms.toFixed(1).padStart(7)}  ${r.name.padEnd(40)}  ${r.url}:${r.line}`)
  }

  cdp.close()
}

main().catch((e) => { console.error(e); process.exit(1) })
