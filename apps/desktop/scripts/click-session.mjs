// Click on a session by partial title match.
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
const send = (method, params = {}) =>
  new Promise(r => {
    const i = ++id
    pending.set(i, r)
    ws.send(JSON.stringify({ id: i, method, params }))
  })

const title = process.argv[2] || 'Phaser particle'
const r = await send('Runtime.evaluate', {
  expression: `
    (() => {
      const titleMatch = ${JSON.stringify(title)}
      const all = document.querySelectorAll('button, a, div[role="button"]')
      const found = [...all].find(el => (el.textContent || '').includes(titleMatch))
      if (!found) return JSON.stringify({ found: false, tried: titleMatch })
      found.scrollIntoView()
      found.click()
      return JSON.stringify({ found: true, tag: found.tagName, text: (found.textContent || '').slice(0, 80) })
    })()
  `,
  returnByValue: true
})
console.log('click raw:', JSON.stringify(r, null, 2))
await new Promise(r => setTimeout(r, 3000))

const status = await send('Runtime.evaluate', {
  expression: `JSON.stringify({
    url: location.href,
    hasComposer: !!document.querySelector('[data-slot="composer-rich-input"]'),
    threadMessages: document.querySelectorAll('[data-slot="aui_message"]').length,
    bodyTextSnippet: document.body.innerText.slice(0, 500),
    title: document.title
  })`,
  returnByValue: true
})
console.log('after click:', status.result.value)
ws.close()
