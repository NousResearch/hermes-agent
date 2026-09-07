// Probe the cloud shadows thread state — count messages, turn pairs,
// thread height, composer state
const list = await (await fetch('http://127.0.0.1:9222/json/list')).json()
const tgt = list.find(t => t.type === 'page' && t.url.startsWith('http'))
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
const send = (m, p = {}) =>
  new Promise(r => {
    const i = ++id
    pending.set(i, r)
    ws.send(JSON.stringify({ id: i, method: m, params: p }))
  })

const r = await send('Runtime.evaluate', {
  expression: `JSON.stringify({
    url: location.href,
    title: document.title,
    turnPairs: document.querySelectorAll('[data-slot="aui_turn-pair"]').length,
    assistantMsgs: document.querySelectorAll('[data-slot="aui_assistant-message-root"]').length,
    userMsgs: document.querySelectorAll('[data-message-role="user"], [data-slot="aui_user-message-root"]').length,
    totalDomNodes: document.querySelectorAll('*').length,
    threadViewportScrollHeight: document.querySelector('[data-slot="aui_thread-viewport"]')?.scrollHeight ?? null,
    threadViewportClientHeight: document.querySelector('[data-slot="aui_thread-viewport"]')?.clientHeight ?? null,
    threadViewportScrollTop: document.querySelector('[data-slot="aui_thread-viewport"]')?.scrollTop ?? null,
    composer: !!document.querySelector('[data-slot="composer-rich-input"]'),
    busy: !!document.querySelector('[aria-label*="Stop"]')
  })`,
  returnByValue: true
})
console.log(JSON.parse(r.result.result.value))
ws.close()
