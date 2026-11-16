// What is the drag actually spending time on?
//
// The render counters proved React is no longer the cost (commits 83 -> 12
// after the $layoutTree fix) yet drag fps stayed ~3 while p95 halved. That
// pattern says a fixed per-frame floor outside React. This takes a real CDP
// trace of one sash drag and prints the category split — Recalculate Style,
// Layout, Paint, Scripting — so the next fix targets the actual cost instead
// of the next plausible-looking thing.
//
//   node scripts/diag-drag-trace.mjs [--port 9222] [--tiles 5]

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
        parts: [{ type: 'text', text: 'Question ' + i }] },
      { id: sid + '-a' + i, role: 'assistant', timestamp: Date.now(), pending: false,
        parts: [{ type: 'text', text: '## Finding ' + i + '\\n\\nProse with **bold** and \`code\`.\\n' }] }
    ])
    window.__T__ = { ids: [] }
    for (let n = 1; n <= ${TILES}; n++) {
      const sid = 'trace-tile-' + n
      const rid = 'trace-rt-' + n
      const messages = []
      for (let i = 0; i < ${TURNS}; i++) messages.push(...turn(sid, i))
      messages.push({ id: sid + '-stream', role: 'assistant', timestamp: Date.now(), pending: true,
        parts: [{ type: 'text', text: 'Working.' }] })
      window.__T__.ids.push({ sid, rid })
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

// Drive the sash WITHOUT awaiting rAF: a slow app would stretch a rAF-paced
// loop and make the window itself a function of the slowness. Fixed wall-clock
// pacing keeps the trace window comparable run to run.
const DRAG = `
  (async () => {
    const handle = document.querySelector('[role="separator"]')
    if (!handle) return 'none'
    const box = handle.getBoundingClientRect()
    const y = box.top + box.height / 2
    const x0 = box.left + box.width / 2
    let x = x0
    const opts = { bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse', isPrimary: true, button: 0, buttons: 1 }
    handle.dispatchEvent(new PointerEvent('pointerdown', { ...opts, clientX: x, clientY: y }))
    for (let i = 0; i < 40; i++) {
      x += (i < 20 ? 3 : -3)
      window.dispatchEvent(new PointerEvent('pointermove', { ...opts, clientX: x, clientY: y }))
      await new Promise(r => setTimeout(r, 16))
    }
    window.dispatchEvent(new PointerEvent('pointerup', { ...opts, buttons: 0, clientX: x, clientY: y }))
    return 'dragged'
  })()
`

const CLEANUP = `
  (() => {
    if (window.__T__) {
      for (const { sid, rid } of window.__T__.ids) {
        const s = window.__HERMES_SESSION_TILES__.states()
        window.__HERMES_SESSION_TILES__.publish(rid, { ...s[rid], busy: false, streamId: null })
        window.__HERMES_SESSION_TILES__.close(sid)
      }
      window.__T__ = null
    }
    return 'cleaned'
  })()
`

const { cdp, teardown } = await attach({ port })

try {
  await cdp.send('Runtime.enable')

  const ok = await cdp.eval(setup)

  if (ok !== 'ok') {
    throw new Error(`setup failed: ${ok}`)
  }

  for (let n = 1; n <= TILES; n++) {
    await cdp.eval(reveal(`trace-tile-${n}`))
    await sleep(300)
  }

  await sleep(1500)

  // Collect trace events for the drag window only. `cdp.on` is the client's
  // only event API (no `once`), so completion is signalled through a flag.
  const events = []
  let complete = false
  cdp.on('Tracing.dataCollected', params => events.push(...(params.value ?? [])))
  cdp.on('Tracing.tracingComplete', () => {
    complete = true
  })

  await cdp.send('Tracing.start', {
    transferMode: 'ReportEvents',
    traceConfig: { includedCategories: ['devtools.timeline', 'blink.user_timing'] }
  })

  const dragged = await cdp.eval(DRAG)

  await cdp.send('Tracing.end')

  for (let waited = 0; !complete && waited < 10000; waited += 200) {
    await sleep(200)
  }

  await cdp.eval(CLEANUP)

  // Sum self-time per timeline category. Nested events would double-count, so
  // attribute each event's duration minus the duration of its direct children.
  const INTERESTING = new Set([
    'UpdateLayoutTree', // Recalculate Style
    'Layout',
    'Paint',
    'PaintImage',
    'Layerize',
    'UpdateLayer',
    'CompositeLayers',
    'FunctionCall',
    'EvaluateScript',
    'TimerFire',
    'EventDispatch',
    'HitTest',
    'ParseHTML',
    'CommitLoad'
  ])

  const totals = new Map()
  let traced = 0

  for (const e of events) {
    if (e.ph !== 'X' || typeof e.dur !== 'number') {
      continue
    }

    traced += 1
    const name = e.name

    if (!INTERESTING.has(name)) {
      continue
    }

    totals.set(name, (totals.get(name) ?? 0) + e.dur / 1000)
  }

  console.log(`drag=${dragged}  trace events=${events.length} (complete=${traced})\n`)
  console.log('TIMELINE COST (ms, total duration by event):')

  const rows = [...totals.entries()].sort((a, b) => b[1] - a[1])

  if (rows.length === 0) {
    console.log('  (no timeline events — category filter or tracing domain unavailable)')
  }

  for (const [name, ms] of rows) {
    console.log(`  ${name.padEnd(20)} ${ms.toFixed(1)}ms`)
  }

  const style = totals.get('UpdateLayoutTree') ?? 0
  const layout = totals.get('Layout') ?? 0
  const script = (totals.get('FunctionCall') ?? 0) + (totals.get('EvaluateScript') ?? 0) + (totals.get('TimerFire') ?? 0)

  console.log(`\nVERDICT: style=${style.toFixed(0)}ms layout=${layout.toFixed(0)}ms script=${script.toFixed(0)}ms`)

  // Script dominates -> name the functions. Timeline FunctionCall events carry
  // the callsite in args.data, so the top offenders can be attributed without
  // a separate CPU profile.
  const byFn = new Map()

  for (const e of events) {
    if (e.ph !== 'X' || e.name !== 'FunctionCall' || typeof e.dur !== 'number') {
      continue
    }

    const d = e.args?.data ?? {}
    const key = `${d.functionName || '(anonymous)'} @ ${(d.url || '?').split('/').pop()}:${d.lineNumber ?? '?'}`
    byFn.set(key, (byFn.get(key) ?? 0) + e.dur / 1000)
  }

  console.log('\nTOP SCRIPT CALLSITES (ms):')

  for (const [name, ms] of [...byFn.entries()].sort((a, b) => b[1] - a[1]).slice(0, 15)) {
    console.log(`  ${ms.toFixed(1).padStart(8)}  ${name}`)
  }
} finally {
  teardown?.()
}
