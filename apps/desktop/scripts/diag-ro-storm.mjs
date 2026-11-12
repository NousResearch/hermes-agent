// How many ResizeObserver callbacks does one sash drag actually fire, and for
// how many DISTINCT elements? The trace named use-resize-observer.ts at 977ms
// but not whether that's a few expensive calls or a great many cheap ones —
// and the fix differs completely between those.
//
//   node scripts/diag-ro-storm.mjs [--port 9222] [--tiles 5]

import { attach } from './perf/lib/launch.mjs'
import { sleep } from './perf/lib/cdp.mjs'

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`)

  return i === -1 ? fallback : process.argv[i + 1]
}

const port = Number(arg('port', 9222))
const TILES = Number(arg('tiles', 5))
const TURNS = Number(arg('turns', 20))

const setup = `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    if (!hook) return 'no-hook'
    const turn = (sid, i) => ([
      { id: sid + '-u' + i, role: 'user', timestamp: Date.now(),
        parts: [{ type: 'text', text: 'Question ' + i + ' about the diff and its error path.' }] },
      { id: sid + '-a' + i, role: 'assistant', timestamp: Date.now(), pending: false,
        parts: [{ type: 'text', text: '## Finding ' + i + '\\n\\nProse with **bold** and \`code\`.\\n' }] }
    ])
    window.__R__ = { ids: [] }
    for (let n = 1; n <= ${TILES}; n++) {
      const sid = 'ro-tile-' + n
      const rid = 'ro-rt-' + n
      const messages = []
      for (let i = 0; i < ${TURNS}; i++) messages.push(...turn(sid, i))
      messages.push({ id: sid + '-stream', role: 'assistant', timestamp: Date.now(), pending: true,
        parts: [{ type: 'text', text: 'Working.' }] })
      window.__R__.ids.push({ sid, rid })
      hook.open(sid, 'center')
      hook.patch(sid, { runtimeId: rid })
      hook.publish(rid, {
        storedSessionId: sid, messages, branch: '', cwd: '', model: '', provider: '',
        reasoningEffort: '', serviceTier: '', fast: false, yolo: false, personality: '',
        busy: true, awaitingResponse: false, streamId: sid + '-stream', sawAssistantPayload: true,
        pendingBranchGroup: null, interrupted: false, interimBoundaryPending: false,
        needsInput: false, turnStartedAt: Date.now(), usage: null
      })
    }
    return 'ok'
  })()
`

const reveal = sid => `window.__HERMES_LAYOUT_TREE__.reveal(${JSON.stringify(`session-tile:${sid}`)})`

// Patch ResizeObserver to count callbacks + distinct observed targets, then
// drag and report. Counting happens in the page so nothing crosses CDP per call.
//
// NOTE: the app's shared observer (hooks/use-resize-observer.ts) is created
// lazily on first use, so this patch must be installed BEFORE any surface
// mounts — otherwise the shared instance is a native one this wrapper never
// sees and every counter reads zero. `constructed` is the tell: a run showing
// a handful of constructions and zero callbacks means the patch landed late,
// not that the app stopped observing.
const INSTRUMENT = `
  (() => {
    if (window.__ROSTATS__) return 'already'
    const Native = window.ResizeObserver
    const stats = { constructed: 0, observed: 0, callbacks: 0, entries: 0, targets: new Set(), on: false }
    window.__ROSTATS__ = stats
    window.ResizeObserver = class extends Native {
      constructor(cb) {
        super((entries, obs) => {
          if (stats.on) {
            stats.callbacks += 1
            stats.entries += entries.length
            for (const e of entries) stats.targets.add(e.target)
          }
          return cb(entries, obs)
        })
        stats.constructed += 1
      }
      observe(...args) {
        stats.observed += 1
        return super.observe(...args)
      }
    }
    return 'patched'
  })()
`

const DRAG = `
  (async () => {
    const s = window.__ROSTATS__
    s.callbacks = 0; s.entries = 0; s.targets = new Set(); s.on = true
    const handle = document.querySelector('[role="separator"]')
    if (!handle) { s.on = false; return JSON.stringify({ error: 'no sash' }) }
    const box = handle.getBoundingClientRect()
    const y = box.top + box.height / 2
    const x0 = box.left + box.width / 2
    let x = x0
    const opts = { bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse', isPrimary: true, button: 0, buttons: 1 }
    const t0 = performance.now()
    handle.dispatchEvent(new PointerEvent('pointerdown', { ...opts, clientX: x, clientY: y }))
    for (let i = 0; i < 40; i++) {
      x += (i < 20 ? 3 : -3)
      window.dispatchEvent(new PointerEvent('pointermove', { ...opts, clientX: x, clientY: y }))
      await new Promise(r => setTimeout(r, 16))
    }
    window.dispatchEvent(new PointerEvent('pointerup', { ...opts, buttons: 0, clientX: x, clientY: y }))
    await new Promise(r => setTimeout(r, 300))
    s.on = false
    return JSON.stringify({
      ms: Math.round(performance.now() - t0),
      moves: 40,
      constructed: s.constructed,
      observed: s.observed,
      callbacks: s.callbacks,
      entries: s.entries,
      distinctTargets: s.targets.size,
      userBubbles: document.querySelectorAll('[data-slot="aui_user-message-root"]').length
    })
  })()
`

const CLEANUP = `
  (() => {
    if (window.__R__) {
      for (const { sid, rid } of window.__R__.ids) {
        const s = window.__HERMES_SESSION_TILES__.states()
        window.__HERMES_SESSION_TILES__.publish(rid, { ...s[rid], busy: false, streamId: null })
        window.__HERMES_SESSION_TILES__.close(sid)
      }
      window.__R__ = null
    }
    return 'cleaned'
  })()
`

const { cdp, teardown } = await attach({ port })

try {
  await cdp.send('Runtime.enable')
  await cdp.eval(INSTRUMENT)

  const ok = await cdp.eval(setup)

  if (ok !== 'ok') {
    throw new Error(`setup failed: ${ok}`)
  }

  for (let n = 1; n <= TILES; n++) {
    await cdp.eval(reveal(`ro-tile-${n}`))
    await sleep(300)
  }

  await sleep(1500)

  const r = JSON.parse(await cdp.eval(DRAG))
  await cdp.eval(CLEANUP)

  console.log(JSON.stringify(r, null, 2))

  if (r.moves) {
    console.log(`\nper pointermove: ${(r.entries / r.moves).toFixed(1)} RO entries`)
    console.log(`distinct elements resized: ${r.distinctTargets} (user bubbles in DOM: ${r.userBubbles})`)
  }
} finally {
  teardown?.()
}
