// Wrap the thread scroller's properties and observe pin/scroll/RO events
// in real time during a submit, then print the timeline.
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
const evalP = async expr => {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true })
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.text)
  return r.result.result.value
}

await evalP(`(() => {
  const v = document.querySelector('[data-slot="aui_thread-viewport"]')
  if (v) v.scrollTop = v.scrollHeight
})()`)
await new Promise(r => setTimeout(r, 300))

await evalP(`(() => {
  const el = document.querySelector('[data-slot="composer-rich-input"]')
  el.focus()
  const r = document.createRange(); r.selectNodeContents(el); r.collapse(false)
  window.getSelection().removeAllRanges(); window.getSelection().addRange(r)
})()`)

const text = 'short follow-up'
for (const c of text) {
  await send('Input.dispatchKeyEvent', { type: 'char', text: c, unmodifiedText: c })
  await new Promise(r => setTimeout(r, 10))
}
await new Promise(r => setTimeout(r, 300))

// Hook into the viewport scrollTop setter + scroll + RO so we see every event
await evalP(`(() => {
  const v = document.querySelector('[data-slot="aui_thread-viewport"]')
  const events = []
  window.__threadEvents = events
  const t0 = performance.now()
  const push = (kind, detail) => events.push({ t: performance.now() - t0, kind, ...detail })

  // intercept scrollTop writes
  const desc = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollTop')
  Object.defineProperty(v, 'scrollTop', {
    get() { return desc.get.call(this) },
    set(val) {
      push('scrollTop=', { val, fromScrollHeight: this.scrollHeight, stackTop: (new Error()).stack.split('\\n').slice(2, 5).map(s => s.trim()).join(' | ') })
      desc.set.call(this, val)
    },
    configurable: true
  })

  // scroll event
  v.addEventListener('scroll', () => {
    push('scroll', { scrollTop: v.scrollTop, scrollHeight: v.scrollHeight })
  }, { passive: true, capture: true })

  // RO on the viewport itself
  const ro = new ResizeObserver((entries) => {
    for (const e of entries) {
      push('RO', { target: e.target.getAttribute('data-slot') || e.target.tagName, h: e.contentRect.height })
    }
  })
  ro.observe(v)
  if (v.firstElementChild) ro.observe(v.firstElementChild)

  // mutationobserver on the viewport
  const mo = new MutationObserver((muts) => {
    push('mut', { count: muts.length, added: muts.reduce((s, m) => s + m.addedNodes.length, 0), removed: muts.reduce((s, m) => s + m.removedNodes.length, 0) })
  })
  mo.observe(v, { childList: true, subtree: true, characterData: true })

  window.__teardown = () => { ro.disconnect(); mo.disconnect() }
  return true
})()`)

// fire Enter
await send('Input.dispatchKeyEvent', {
  type: 'rawKeyDown', windowsVirtualKeyCode: 13, key: 'Enter', code: 'Enter', text: '\r', unmodifiedText: '\r'
})
await send('Input.dispatchKeyEvent', { type: 'keyUp', windowsVirtualKeyCode: 13, key: 'Enter', code: 'Enter' })

await new Promise(r => setTimeout(r, 1200))

const events = JSON.parse(await evalP(`JSON.stringify(window.__threadEvents || [])`))
console.log(`\n${events.length} events:`)
for (const e of events) {
  const t = String(e.t.toFixed(0)).padStart(5)
  const { kind, t: _t, ...rest } = e
  console.log(`  ${t}ms  ${kind.padEnd(12)} ${JSON.stringify(rest)}`)
}

await evalP(`window.__teardown?.()`)
// Cancel running agent
await evalP(`(() => {
  for (const b of document.querySelectorAll('button')) {
    if ((b.getAttribute('aria-label') || '').toLowerCase().includes('stop')) { b.click(); return 'stopped' }
  }
})()`)

ws.close()
