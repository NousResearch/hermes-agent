// Click on a session by partial title match.
//
// Two flake sources fixed (#97982): a bare scrollIntoView() whose smooth
// scroll could be dropped under load, and a fixed 3000ms post-click sleep
// that raced fast loads and wasted slow ones. The click now scrolls instantly
// ({ behavior: 'auto', block: 'center' }) and the post-click wait polls for
// the session's composer to mount — see click-session-helpers.mjs.
import {
  buildFindClickExpression,
  evaluateJsonValue,
  pollUntil,
  POST_CLICK_READY_EXPRESSION
} from './click-session-helpers.mjs'

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
  expression: buildFindClickExpression(title),
  returnByValue: true
})
console.log('click raw:', JSON.stringify(r, null, 2))

// Wait for the clicked session's composer to mount — bounded polling, not a
// fixed sleep. `false` here means the session view never came up within the
// bound; the state read below then shows what the page actually looks like.
const ready = await pollUntil(send, POST_CLICK_READY_EXPRESSION)
console.log('post-click ready:', ready)

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
// Nested CDP envelope: message.result.result.value — evaluateJsonValue unwraps
// and parses it (the old `status.result.value` read logged undefined).
console.log('after click:', evaluateJsonValue(status))
ws.close()
