// Simple eval helper — runs an expression and returns the result.value.
//
//   node scripts/eval.mjs "document.title"
//   HERMES_DESKTOP_CDP_PORT=9333 node scripts/eval.mjs "document.title"
//
// Needs a renderer with a debugging port: launch `hgui` / `npm run dev` with
// HERMES_DESKTOP_CDP_PORT set (see electron/dev-cdp.ts).
const port = Number(process.env.HERMES_DESKTOP_CDP_PORT || 9222)
let targets

try {
  targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json()
} catch {
  console.error(
    `no renderer debugging port on 127.0.0.1:${port}. ` +
      'Dev-server runs (`npm run dev` / `hgui`) open one automatically — check the app is running, ' +
      'and that HERMES_DESKTOP_CDP_PORT is not set to "off" or another port.'
  )
  process.exit(1)
}

const t = targets.find((t) => t.url.includes('5174')) ?? targets.find((t) => t.type === 'page')

if (!t) {
  console.error(`no page target on 127.0.0.1:${port} (found ${targets.length} target(s))`)
  process.exit(1)
}

const ws = new WebSocket(t.webSocketDebuggerUrl)
let id = 0
const pending = new Map()
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data)
  if (pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id) }
})
await new Promise((r) => ws.addEventListener('open', r))
const send = (method, params) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })) })

const expr = process.argv[2] || '1+1'
const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })
if (r.result.exceptionDetails) {
  console.error('EXCEPTION:', r.result.exceptionDetails.exception?.description)
} else {
  console.log(JSON.stringify(r.result.result.value, null, 2))
}
ws.close()
