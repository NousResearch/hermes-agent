// Hard reload the Electron renderer over CDP. Vite-no-HMR mode means edits
// don't auto-apply — call this after editing source.
const targets = await (await fetch('http://127.0.0.1:9222/json')).json()
const t = targets.find((t) => t.url.includes('5174'))
if (!t) {
  console.error('renderer not found')
  process.exit(1)
}
const ws = new WebSocket(t.webSocketDebuggerUrl)
let id = 0
const pending = new Map()
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data)
  if (pending.has(m.id)) {
    pending.get(m.id)(m)
    pending.delete(m.id)
  }
})
await new Promise((r) => ws.addEventListener('open', r))
const send = (method, params = {}) =>
  new Promise((res) => {
    const i = ++id
    pending.set(i, res)
    ws.send(JSON.stringify({ id: i, method, params }))
  })

await send('Page.reload', { ignoreCache: true })
console.log('reload sent')
// Wait for new doc.
await new Promise((r) => setTimeout(r, 2500))
const r = await send('Runtime.evaluate', {
  expression: 'JSON.stringify({ hasProbe: !!window.__PERF_PROBE__, composer: !!document.querySelector("[contenteditable=true]"), url: location.hash })',
  returnByValue: true,
})
console.log(r.result.result.value)
ws.close()
