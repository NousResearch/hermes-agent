// Measure scroll position before and after Enter on a long thread.
// The user's complaint: pressing Enter to submit makes the view "jump up".
//
// Steps:
//   1. Scroll to the bottom of the thread
//   2. Type a short message
//   3. Record scroll position
//   4. Hit Enter
//   5. Record scroll position every 10ms for 1.5s after Enter
//   6. Report deltas
//
// Usage:  node apps/desktop/scripts/measure-jump.mjs

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

// Scroll to bottom
await evalP(`(() => {
  const v = document.querySelector('[data-slot="aui_thread-viewport"]')
  if (v) v.scrollTop = v.scrollHeight
})()`)
await new Promise(r => setTimeout(r, 300))

// Focus composer and type
await evalP(`(() => {
  const el = document.querySelector('[data-slot="composer-rich-input"]')
  el.focus()
  const r = document.createRange(); r.selectNodeContents(el); r.collapse(false)
  window.getSelection().removeAllRanges(); window.getSelection().addRange(r)
})()`)

const text = 'short follow-up message'
for (const c of text) {
  await send('Input.dispatchKeyEvent', { type: 'char', text: c, unmodifiedText: c })
  await new Promise(r => setTimeout(r, 10))
}
await new Promise(r => setTimeout(r, 300))

// Set up sampling — sample scroll position every animation frame
await evalP(`(() => {
  const v = document.querySelector('[data-slot="aui_thread-viewport"]')
  window.__jumpSamples = []
  window.__jumpStart = performance.now()
  const tick = () => {
    if (!v) return
    window.__jumpSamples.push({
      t: performance.now() - window.__jumpStart,
      scrollTop: v.scrollTop,
      scrollHeight: v.scrollHeight,
      clientHeight: v.clientHeight,
      distFromBottom: v.scrollHeight - v.scrollTop - v.clientHeight
    })
    if (performance.now() - window.__jumpStart < 2000) {
      requestAnimationFrame(tick)
    }
  }
  requestAnimationFrame(tick)
})()`)

// Fire Enter
await send('Input.dispatchKeyEvent', {
  type: 'rawKeyDown', windowsVirtualKeyCode: 13, key: 'Enter', code: 'Enter', text: '\r', unmodifiedText: '\r'
})
await send('Input.dispatchKeyEvent', { type: 'keyUp', windowsVirtualKeyCode: 13, key: 'Enter', code: 'Enter' })

await new Promise(r => setTimeout(r, 2200))

const samples = JSON.parse(await evalP(`JSON.stringify(window.__jumpSamples || [])`))
console.log(`\n${samples.length} samples over 2s`)
console.log(`\n  t(ms)  scrollTop  scrollHeight  clientHeight  distFromBottom`)
let prev = null
for (const s of samples) {
  const marker = prev && Math.abs(s.scrollTop - prev.scrollTop) > 5 ? '  ← jump' : ''
  console.log(`  ${String(s.t.toFixed(0)).padStart(5)}  ${String(s.scrollTop).padStart(9)}  ${String(s.scrollHeight).padStart(12)}  ${String(s.clientHeight).padStart(12)}  ${String(s.distFromBottom).padStart(14)}${marker}`)
  prev = s
}

// Cancel any running agent
await evalP(`(() => {
  for (const b of document.querySelectorAll('button')) {
    if ((b.getAttribute('aria-label') || '').toLowerCase().includes('stop')) { b.click(); return 'stopped' }
  }
  return 'no-stop'
})()`).then(r => console.log('\ncancel:', r))

ws.close()
