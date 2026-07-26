// What does the app cost while a turn is running but NOTHING is arriving?
//
// The user-visible symptom: with a thread spinning, resizing the sidebar or
// typing in the composer feels slow. That is not streaming cost — the stream
// is idle. It is the app re-rendering on its own, competing with the
// interaction for the main thread.
//
// This scenario holds N tiles in a busy state, pushes NO tokens, and measures:
//   - idle_commits_per_s  the renderer's self-inflicted commit rate
//   - drag_fps            fps while dragging the sidebar splitter
//   - type_fps            fps while typing in the composer
//
// A perfectly idle app scores 0 idle commits and pins both interactions at the
// display's refresh rate. Every idle commit is main-thread time stolen from an
// interaction the user can feel.
//
//   node scripts/perf/run.mjs idle-cost --spawn [--tiles 5] [--seconds 6]

import { sleep } from '../lib/cdp.mjs'

/** Seed `tiles` busy session tiles. Same publish path as `multitab` /
 *  `render-churn`, but the driver never runs — the turn just stays open. */
const setup = (tiles, seedTurns) => `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    if (!hook) return 'no-hook'
    if (!window.__RENDER_COUNTS__) return 'no-render-counter'

    const turn = (sid, i) => ([
      { id: sid + '-u' + i, role: 'user', timestamp: Date.now(),
        parts: [{ type: 'text', text: 'Question ' + i + ' about the diff.' }] },
      { id: sid + '-a' + i, role: 'assistant', timestamp: Date.now(), pending: false,
        parts: [{ type: 'text', text: '## Finding ' + i + '\\n\\nThe handler swallows the rejection.\\n\\n- Point one.\\n- Point two.\\n' }] }
    ])

    window.__IDLE__ = { ids: [] }
    for (let n = 1; n <= ${tiles}; n++) {
      const sid = 'idle-tile-' + n
      const rid = 'idle-rt-' + n
      const messages = []
      for (let i = 0; i < ${seedTurns}; i++) messages.push(...turn(sid, i))
      // An OPEN assistant message: the turn is running, but no tokens arrive.
      messages.push({ id: sid + '-stream', role: 'assistant', timestamp: Date.now(), pending: true,
        parts: [{ type: 'text', text: 'Working on it.' }] })

      window.__IDLE__.ids.push({ sid, rid })
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

/** Measure the app's self-inflicted commit rate with nothing happening. */
const idleCost = seconds => `
  (async () => {
    const rc = window.__RENDER_COUNTS__
    rc.start()
    const t0 = performance.now()
    await new Promise(r => setTimeout(r, ${seconds} * 1000))
    const elapsed = (performance.now() - t0) / 1000
    rc.stop()
    return JSON.stringify({
      elapsed,
      commits: rc.commits(),
      top: rc.report(12),
      owners: rc.report(300).filter(r => r.stateChanged > 0 && r.propsChanged === 0).slice(0, 10)
    })
  })()
`

/** Record frame pacing across an interaction driven from the page, plus WHAT
 *  rendered during it — a slow interaction is almost never the handler, it is
 *  whatever re-renders underneath while the handler runs. */
const withFrames = body => `
  (async () => {
    const rc = window.__RENDER_COUNTS__
    rc.start()
    const frames = []
    let last = performance.now()
    let stop = false
    const tick = () => {
      if (stop) return
      const now = performance.now()
      frames.push(now - last)
      last = now
      requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
    ${body}
    stop = true
    rc.stop()
    const total = frames.reduce((a, b) => a + b, 0)
    const sorted = [...frames].sort((a, b) => a - b)
    const pct = p => sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] : 0
    return JSON.stringify({
      fps: total ? (frames.length / total) * 1000 : 0,
      p95: pct(0.95),
      worst: sorted.length ? sorted[sorted.length - 1] : 0,
      slow33: frames.filter(f => f > 33).length,
      n: frames.length,
      commits: rc.commits(),
      top: rc.report(10)
    })
  })()
`

/** Drag the sidebar splitter back and forth — the resize symptom. */
const DRAG = withFrames(`
  const handle = document.querySelector('[data-slot="pane-resize-handle"], [role="separator"], .resize-handle')
  if (handle) {
    const box = handle.getBoundingClientRect()
    const y = box.top + box.height / 2
    let x = box.left + box.width / 2
    handle.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: x, clientY: y, pointerId: 1 }))
    for (let i = 0; i < 60; i++) {
      x += (i % 2 === 0 ? 3 : -3)
      window.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: x, clientY: y, pointerId: 1 }))
      await new Promise(r => requestAnimationFrame(r))
    }
    window.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, clientX: x, clientY: y, pointerId: 1 }))
  } else {
    await new Promise(r => setTimeout(r, 1000))
  }
`)

/** Type into the composer — the keystroke symptom. */
const TYPE = withFrames(`
  const el = document.querySelector('[contenteditable="true"], textarea')
  if (el) {
    el.focus()
    for (let i = 0; i < 40; i++) {
      const ch = 'performance testing '[i % 20]
      el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: ch }))
      if (el.tagName === 'TEXTAREA') {
        el.value += ch
        el.dispatchEvent(new Event('input', { bubbles: true }))
      } else {
        el.textContent += ch
        el.dispatchEvent(new InputEvent('input', { bubbles: true, data: ch, inputType: 'insertText' }))
      }
      el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: ch }))
      await new Promise(r => setTimeout(r, 25))
    }
  } else {
    await new Promise(r => setTimeout(r, 1000))
  }
`)

const CLEANUP = `
  (() => {
    if (window.__IDLE__) {
      for (const { sid, rid } of window.__IDLE__.ids) {
        const states = window.__HERMES_SESSION_TILES__.states()
        window.__HERMES_SESSION_TILES__.publish(rid, { ...states[rid], busy: false, streamId: null })
        window.__HERMES_SESSION_TILES__.close(sid)
      }
      window.__IDLE__ = null
    }
    window.__RENDER_COUNTS__.clear()
    return 'cleaned'
  })()
`

const round = (n, places = 1) => Math.round(n * 10 ** places) / 10 ** places

export default {
  name: 'idle-cost',
  tier: 'ci',
  description: 'Busy-but-silent tiles: idle commit rate, and fps while resizing / typing.',
  async run(cdp, opts = {}) {
    const tiles = Number(opts.tiles ?? 5)
    const seedTurns = Number(opts.turns ?? 20)
    const seconds = Number(opts.seconds ?? 6)

    await cdp.send('Runtime.enable')

    const ok = await cdp.eval(setup(tiles, seedTurns))

    if (ok !== 'ok') {
      throw new Error(`idle-cost setup failed (${ok}) — needs a dev renderer with src/debug installed.`)
    }

    for (let n = 1; n <= tiles; n++) {
      await cdp.eval(reveal(`idle-tile-${n}`))
      await sleep(300)
    }

    await sleep(1500)

    const idle = JSON.parse(await cdp.eval(idleCost(seconds)))
    const drag = JSON.parse(await cdp.eval(DRAG))
    const type = JSON.parse(await cdp.eval(TYPE))

    await cdp.eval(CLEANUP)

    return {
      metrics: {
        // Commits per second with a turn open and nothing arriving. Should be 0.
        idle_commits_per_s: round(idle.commits / idle.elapsed),
        idle_renders: idle.top.reduce((a, r) => a + r.renders, 0),
        // Interaction smoothness while that churn competes for the main thread.
        // Reported as a deficit from 60fps so "lower is better" matches the
        // baseline gate's direction.
        drag_fps_deficit: round(Math.max(0, 60 - drag.fps)),
        drag_slow_frames: drag.slow33,
        type_fps_deficit: round(Math.max(0, 60 - type.fps)),
        type_slow_frames: type.slow33
      },
      detail: {
        tiles,
        idleSeconds: round(idle.elapsed),
        dragFps: round(drag.fps),
        dragP95: round(drag.p95),
        dragWorst: round(drag.worst),
        typeFps: round(type.fps),
        typeP95: round(type.p95),
        typeWorst: round(type.worst),
        // Components whose OWN state changed with no prop change: the roots.
        idleOwners: idle.owners,
        idleTop: idle.top,
        dragCommits: drag.commits,
        dragTop: drag.top,
        typeCommits: type.commits,
        typeTop: type.top
      }
    }
  }
}
