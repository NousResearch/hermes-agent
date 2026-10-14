// quick probe — read state of the renderer
const list = await (await fetch('http://127.0.0.1:9222/json/list')).json()
const tgt = list.find(t => t.type === 'page' && t.url.startsWith('http'))
console.log('target:', tgt?.url)
if (!tgt) process.exit(1)
const ws = new WebSocket(tgt.webSocketDebuggerUrl)
let id = 0
const pending = new Map()
ws.addEventListener('message', ev => {
  const m = JSON.parse(ev.data)
  if (m.id != null && pending.has(m.id)) {
    pending.get(m.id)(m)
    pending.delete(m.id)
  }
})
await new Promise(r => ws.addEventListener('open', r))
const send = (method, params = {}) =>
  new Promise(r => {
    const i = ++id
    pending.set(i, r)
    ws.send(JSON.stringify({ id: i, method, params }))
  })

const r = await send('Runtime.evaluate', {
  expression: `({
    url: location.href,
    title: document.title,
    rootChildren: document.getElementById('root')?.children.length ?? 0,
    rootInner: (document.getElementById('root')?.innerHTML ?? '').slice(0, 300),
    hasComposer: !!document.querySelector('[data-slot="composer-rich-input"]'),
    bootStage: (document.querySelector('[data-slot*="boot"]')?.getAttribute('data-slot')) ?? null,
    bodyText: document.body.innerText.slice(0, 300),
    errorCount: window.__errors?.length ?? 'n/a'
  })`,
  returnByValue: true
})
console.log('raw:', JSON.stringify(r, null, 2))
ws.close()
