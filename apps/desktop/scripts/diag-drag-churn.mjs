// Who re-renders the transcript during a sash drag?
//
// Standalone probe, not a benchmark: seeds tiles, then drags the sash while
// recording (a) render attribution and (b) every nanostores atom that notifies
// during the gesture. The idle-cost scenario proved the transcript re-renders
// ~18x above baseline during a drag but that the sash HANDLER is not the cause
// (identical counts at 0px and 60px displacement) — so this names the store
// that actually fires.
//
//   node scripts/perf/diag-drag-churn.mjs [--port 9222]

import { attach } from './perf/lib/launch.mjs'
import { sleep } from './perf/lib/cdp.mjs'

const TILES = 5
const TURNS = 20

const setup = `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    if (!hook) return 'no-hook'
    const turn = (sid, i) => ([
      { id: sid + '-u' + i, role: 'user', timestamp: Date.now(),
        parts: [{ type: 'text', text: 'Question ' + i }] },
      { id: sid + '-a' + i, role: 'assistant', timestamp: Date.now(), pending: false,
        parts: [{ type: 'text', text: '## Finding ' + i + '\\n\\nSome prose with **bold** and \`code\`.\\n' }] }
    ])
    window.__D__ = { ids: [] }
    for (let n = 1; n <= ${TILES}; n++) {
      const sid = 'diag-tile-' + n
      const rid = 'diag-rt-' + n
      const messages = []
      for (let i = 0; i < ${TURNS}; i++) messages.push(...turn(sid, i))
      messages.push({ id: sid + '-stream', role: 'assistant', timestamp: Date.now(), pending: true,
        parts: [{ type: 'text', text: 'Working.' }] })
      window.__D__.ids.push({ sid, rid })
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

// Drag, recording renders AND atom notifications together.
const DRAG = `
  (async () => {
    const rc = window.__RENDER_COUNTS__
    const ac = window.__ATOM_CHURN__
    rc.start(); ac.start()

    const handle = document.querySelector('[role="separator"]')
    if (!handle) { rc.stop(); ac.stop(); return JSON.stringify({ error: 'no sash' }) }

    const box = handle.getBoundingClientRect()
    const y = box.top + box.height / 2
    const x0 = box.left + box.width / 2
    let x = x0
    const opts = { bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse', isPrimary: true, button: 0, buttons: 1 }
    handle.dispatchEvent(new PointerEvent('pointerdown', { ...opts, clientX: x, clientY: y }))
    for (let i = 0; i < 30; i++) {
      x += 2
      window.dispatchEvent(new PointerEvent('pointermove', { ...opts, clientX: x, clientY: y }))
      await new Promise(r => requestAnimationFrame(r))
    }
    window.dispatchEvent(new PointerEvent('pointerup', { ...opts, buttons: 0, clientX: x, clientY: y }))

    rc.stop(); ac.stop()
    const all = rc.report(400)
    const named = n => all.find(r => r.name === n) || null
    return JSON.stringify({
      moved: Math.round(x - x0),
      commits: rc.commits(),
      renders: rc.report(14),
      // The suspects: who at the TOP of the transcript tree re-rendered?
      chain: ['ChatView', 'ChatRuntimeBoundary', 'AuiProvider', 'Thread', 'SessionTile', 'TileChat',
              'LayoutTreeRoot', 'TreeNode', 'TreeSplit', 'TreeGroup', 'SessionView', 'PaneShell']
        .map(n => ({ name: n, hit: named(n) })).filter(x => x.hit),
      atoms: ac.report(30)
    })
  })()
`

const CLEANUP = `
  (() => {
    if (window.__D__) {
      for (const { sid, rid } of window.__D__.ids) {
        const s = window.__HERMES_SESSION_TILES__.states()
        window.__HERMES_SESSION_TILES__.publish(rid, { ...s[rid], busy: false, streamId: null })
        window.__HERMES_SESSION_TILES__.close(sid)
      }
      window.__D__ = null
    }
    return 'cleaned'
  })()
`

const port = Number(process.argv.includes('--port') ? process.argv[process.argv.indexOf('--port') + 1] : 9222)
const { cdp, teardown } = await attach({ port })

try {
  await cdp.send('Runtime.enable')

  const ok = await cdp.eval(setup)

  if (ok !== 'ok') {
    throw new Error(`setup failed: ${ok}`)
  }

  for (let n = 1; n <= TILES; n++) {
    await cdp.eval(reveal(`diag-tile-${n}`))
    await sleep(300)
  }

  await sleep(1500)

  const data = JSON.parse(await cdp.eval(DRAG))
  await cdp.eval(CLEANUP)

  console.log(`moved ${data.moved}px, ${data.commits} commits\n`)
  console.log('RENDERS during drag:')

  for (const r of data.renders) {
    console.log(
      `  ${r.name.padEnd(28)} r=${String(r.renders).padStart(6)} wasted=${String(r.wasted).padStart(6)} ` +
        `props=${String(r.propsChanged).padStart(5)} state=${String(r.stateChanged).padStart(5)} ` +
        `ctx=${String(r.contextChanged ?? 0).padStart(4)} ms=${r.totalMs}`
    )
  }

  console.log('\nTRANSCRIPT CHAIN (who above the messages re-rendered):')

  for (const { name, hit } of data.chain) {
    console.log(
      `  ${name.padEnd(24)} r=${String(hit.renders).padStart(6)} wasted=${String(hit.wasted).padStart(6)} ` +
        `props=${String(hit.propsChanged).padStart(5)} state=${String(hit.stateChanged).padStart(5)} ms=${hit.totalMs}`
    )
  }

  console.log('\nATOMS that notified during drag:')

  for (const a of data.atoms) {
    console.log(
      `  ${a.name.padEnd(26)} notifies=${String(a.notifies).padStart(5)} wasted=${String(a.wasted).padStart(5)} ` +
        `fanout=${String(a.fanout).padStart(6)} peakListeners=${a.peakListeners}`
    )
  }
} finally {
  teardown?.()
}
