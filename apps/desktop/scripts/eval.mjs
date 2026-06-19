// Simple eval helper — runs an expression and returns the result.value.
const targets = await (await fetch('http://127.0.0.1:9222/json')).json()
const t = targets.find((t) => t.url.includes('5174'))
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
