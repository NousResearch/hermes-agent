// Reproduce + diagnose the "scroll wheel resets position while reading" bug.
//
// The complaint (Windows, mouse wheel): scrolling UP through a chat to re-read
// older content randomly yanks the view to a different position, so you have to
// fight the scrollbar. Mac users on trackpads don't see it.
//
// Hypothesis: the thread scroller has the browser default `overflow-anchor:
// auto`, and the thread renders items in natural document flow (padding
// spacers, NOT transforms). When an item above the viewport is measured by
// @tanstack/react-virtual (its real height differs a lot from the 220px
// estimate) — or when Shiki/images/fonts reflow it — TWO mechanisms both
// adjust scrollTop for the same delta: TanStack's measurement compensation AND
// the browser's native scroll anchoring. The double-correction lurches the
// view. A mouse wheel's coarse, discrete notches mount/measure several
// under-estimated turns per tick, so the over-correction is large and visible;
// a trackpad's ~1-3px/frame keeps it sub-perceptual.
//
// This script drives synthetic mouse-wheel-UP scrolling on a long thread and
// measures how much a tracked on-screen turn jumps, first with
// `overflow-anchor: auto` (reproduce) then `overflow-anchor: none` (the fix).
// If the fix run shows dramatically fewer/smaller jumps, the hypothesis holds.
//
// Prereq: a running desktop app with remote debugging on 9222, on a thread
// with enough history to scroll (the longer / more code+tool blocks, the
// better the repro). Then:  node apps/desktop/scripts/diag-scroll-reset.mjs

const NOTCHES = 14 // wheel-up ticks per sweep
const NOTCH_PX = 120 // Windows wheel notch ≈ 120px
const NOTCH_GAP_MS = 130 // let each smooth-scroll animation settle
const REVERSE_JUMP_PX = 6 // tracked turn moving UP while scrolling up = wrong way
const LURCH_PX = 60 // single-frame on-screen jump that reads as a "reset"

const list = await (await fetch('http://127.0.0.1:9222/json/list')).json()
const tgt = list.find(t => t.type === 'page' && t.url.startsWith('http'))
if (!tgt) {
  console.error('No page target on :9222. Is the desktop app running with --remote-debugging-port=9222?')
  process.exit(1)
}
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
const sleep = ms => new Promise(r => setTimeout(r, ms))

// Install per-sweep instrumentation. `mode` is the overflow-anchor value to
// force inline so we A/B the exact same thread regardless of any CSS fix.
// Starts from ~45% down the thread so there's room to scroll up into
// not-yet-measured turns, tags the turn nearest viewport-center as the anchor,
// then records (per rAF) scrollTop + that turn's on-screen top, plus every
// scrollTop *setter* write (TanStack compensation) and ResizeObserver hit.
async function arm(mode) {
  await evalP(`(() => {
    const v = document.querySelector('[data-slot="aui_thread-viewport"]')
    if (!v) throw new Error('thread viewport not found')

    // Force the overflow-anchor behavior under test (inline beats CSS).
    v.style.overflowAnchor = ${JSON.stringify(mode)}

    // Park ~45% down so a wheel-up sweep climbs into estimated-but-unmeasured
    // turns above the fold (where the measurement correction fires).
    v.scrollTop = Math.round(v.scrollHeight * 0.45)

    // Tag the turn closest to viewport center; we track its on-screen top.
    const vr = v.getBoundingClientRect()
    const center = vr.top + v.clientHeight / 2
    let best = null, bestD = Infinity
    for (const el of v.querySelectorAll('[data-index]')) {
      const r = el.getBoundingClientRect()
      const d = Math.abs((r.top + r.height / 2) - center)
      if (d < bestD) { bestD = d; best = el }
    }
    document.querySelectorAll('[data-se-anchor]').forEach(e => e.removeAttribute('data-se-anchor'))
    if (best) best.setAttribute('data-se-anchor', '1')
    const anchorIndex = best ? best.getAttribute('data-index') : null

    const samples = []
    const writes = []
    const ros = []
    const t0 = performance.now()

    // Intercept scrollTop writes → these are JS (TanStack) corrections.
    // Native browser scroll anchoring does NOT go through this setter, so a
    // scrollTop change with no write in the same frame is a native adjust.
    const desc = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollTop')
    Object.defineProperty(v, 'scrollTop', {
      configurable: true,
      get() { return desc.get.call(this) },
      set(val) {
        writes.push({ t: performance.now() - t0, val, sh: this.scrollHeight })
        desc.set.call(this, val)
      }
    })
    window.__restoreScrollTop = () => Object.defineProperty(v, 'scrollTop', desc)

    const ro = new ResizeObserver(entries => {
      for (const e of entries) {
        ros.push({ t: performance.now() - t0, slot: e.target.getAttribute?.('data-slot') || e.target.tagName, h: Math.round(e.contentRect.height) })
      }
    })
    ro.observe(v)
    if (v.firstElementChild) ro.observe(v.firstElementChild)

    let running = true
    const tick = () => {
      if (!running) return
      const a = v.querySelector('[data-se-anchor]')
      const ar = a ? a.getBoundingClientRect() : null
      samples.push({
        t: performance.now() - t0,
        st: Math.round(v.scrollTop * 100) / 100,
        sh: v.scrollHeight,
        ch: v.clientHeight,
        atop: ar ? Math.round(ar.top * 100) / 100 : null,
        aconn: !!a
      })
      requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)

    window.__se = { samples, writes, ros, anchorIndex, dpr: window.devicePixelRatio, stop() { running = false; ro.disconnect(); window.__restoreScrollTop?.() } }
    return true
  })()`)
}

async function wheelUpSweep() {
  const { x, y } = await evalP(`(() => {
    const v = document.querySelector('[data-slot="aui_thread-viewport"]')
    const r = v.getBoundingClientRect()
    return { x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2) }
  })()`)

  for (let i = 0; i < NOTCHES; i++) {
    await send('Input.dispatchMouseEvent', { type: 'mouseWheel', x, y, deltaX: 0, deltaY: -NOTCH_PX })
    await sleep(NOTCH_GAP_MS)
  }
  await sleep(400)
}

async function collect() {
  const data = JSON.parse(await evalP(`(() => { window.__se.stop(); return JSON.stringify(window.__se) })()`))
  return data
}

function analyze(label, data) {
  const { samples, writes, ros, anchorIndex, dpr } = data
  let reverseJumps = 0
  let reverseSum = 0
  let lurches = 0
  let maxJump = 0
  let nativeMoves = 0
  let prev = null
  for (const s of samples) {
    if (prev && prev.aconn && s.aconn && prev.atop != null && s.atop != null) {
      const dTop = s.atop - prev.atop // wheel-up should move content DOWN → dTop >= 0
      const dSt = s.st - prev.st
      // Native (browser-anchoring) move: scrollTop changed with no setter write in this frame window.
      const wroteThisFrame = writes.some(w => w.t > prev.t && w.t <= s.t)
      if (Math.abs(dSt) > 0.5 && !wroteThisFrame) nativeMoves++
      if (dTop < -REVERSE_JUMP_PX) {
        reverseJumps++
        reverseSum += -dTop
      }
      if (Math.abs(dTop) > LURCH_PX) lurches++
      if (Math.abs(dTop) > maxJump) maxJump = Math.abs(dTop)
    }
    prev = s
  }
  console.log(`\n── ${label} ──`)
  console.log(`  devicePixelRatio:     ${dpr}${Number.isInteger(dpr) ? '' : '  (fractional — Windows scaling, worsens rounding jitter)'}`)
  console.log(`  tracked turn index:   ${anchorIndex}`)
  console.log(`  rAF frames:           ${samples.length}`)
  console.log(`  scrollTop writes:     ${writes.length}   (TanStack measurement corrections)`)
  console.log(`  ResizeObserver hits:  ${ros.length}`)
  console.log(`  native scroll moves:  ${nativeMoves}   (scrollTop moved with NO JS write = browser anchoring)`)
  console.log(`  reverse jumps:        ${reverseJumps}   (tracked turn yanked UP while scrolling up; total ${reverseSum.toFixed(0)}px)`)
  console.log(`  big lurches (>${LURCH_PX}px):   ${lurches}`)
  console.log(`  max single-frame jump: ${maxJump.toFixed(0)}px`)
  return { reverseJumps, reverseSum, lurches, maxJump, nativeMoves }
}

console.log(`Wheel-up repro: ${NOTCHES} notches × ${NOTCH_PX}px, anchored mid-thread.\n`)

await arm('auto')
await sleep(150)
await wheelUpSweep()
const a = analyze('overflow-anchor: auto  (current / repro)', await collect())

await sleep(300)

await arm('none')
await sleep(150)
await wheelUpSweep()
const b = analyze('overflow-anchor: none  (proposed fix)', await collect())

// Clean up our tag.
await evalP(`document.querySelectorAll('[data-se-anchor]').forEach(e => e.removeAttribute('data-se-anchor'))`)

console.log('\n══ verdict ══')
const drop = (x, y) => (x === 0 ? (y === 0 ? '0' : 'n/a') : `${Math.round((1 - y / x) * 100)}% fewer`)
console.log(`  reverse jumps:  auto=${a.reverseJumps}  none=${b.reverseJumps}  (${drop(a.reverseJumps, b.reverseJumps)})`)
console.log(`  big lurches:    auto=${a.lurches}  none=${b.lurches}  (${drop(a.lurches, b.lurches)})`)
console.log(`  max jump:       auto=${a.maxJump.toFixed(0)}px  none=${b.maxJump.toFixed(0)}px`)
console.log(`  native moves:   auto=${a.nativeMoves}  none=${b.nativeMoves}  (browser anchoring should ~vanish at none)`)
if (a.reverseJumps + a.lurches > 0 && b.reverseJumps + b.lurches < a.reverseJumps + a.lurches) {
  console.log('\n  → Jumps drop sharply with overflow-anchor:none → root cause confirmed.')
} else if (a.reverseJumps + a.lurches === 0) {
  console.log('\n  → No jumps captured this run. Use a longer thread (many code/tool blocks),')
  console.log('    raise NOTCHES, and ensure you start scrolled up from the bottom.')
}

ws.close()
