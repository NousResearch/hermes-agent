// Render churn during multi-tab streaming: WHAT re-rendered and WHY, and which
// store published the update. Frame pacing (see `multitab`) tells you the cost;
// this tells you the cause.
//
// Drives the same synthetic pipeline as `multitab` — publishSessionState per
// session per flush via `__HERMES_SESSION_TILES__`, no backend, no credits —
// then reads the dev-only counters installed by `src/debug/`:
//
//   window.__RENDER_COUNTS__  — per-component renders, attributed to
//                               props / hook state / parent-only ("wasted")
//   window.__ATOM_CHURN__     — per-store notifications, listener fan-out, and
//                               notifications whose value was deep-equal to the
//                               previous one ("wasted")
//
// The headline metric is `sidebar_renders`: how many times the sidebar tree
// re-rendered while agents were typing in other tabs. It should be 0.
//
//   node scripts/perf/run.mjs render-churn --spawn [--tiles 5] [--tokens 240]

import { sleep } from '../lib/cdp.mjs'

/** Components that make up the sidebar tree. A render of any of these while
 *  a background tab streams is work the user cannot see. */
const SIDEBAR_COMPONENTS = [
  'ChatSidebar',
  'SidebarSurface',
  'SessionRow',
  'SessionsSection',
  'CronJobsSection',
  'ProfileSwitcher',
  'VirtualSessionList',
  'WorkspaceGroup',
  'OverviewRow',
  'SessionStatusDot'
]

/** Page-side setup: open `tiles` session tiles, seed each with a transcript.
 *  Mirrors `multitab.mjs` so the two scenarios measure the same workload. */
const setup = (tiles, seedTurns) => `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    if (!hook) return 'no-hook'
    if (!window.__RENDER_COUNTS__) return 'no-render-counter'
    if (!window.__ATOM_CHURN__) return 'no-atom-churn'

    const turn = (sid, i) => ([
      { id: sid + '-u' + i, role: 'user', timestamp: Date.now(),
        parts: [{ type: 'text', text: 'Review question ' + i + ': does the diff handle the error path?' }] },
      { id: sid + '-a' + i, role: 'assistant', timestamp: Date.now(), pending: false,
        parts: [{ type: 'text', text: '## Finding ' + i + '\\n\\nThe handler swallows the rejection.\\n\\n- The catch block drops the error.\\n- Retries are unbounded.\\n' }] }
    ])

    const state = (sid) => {
      const messages = []
      for (let i = 0; i < ${seedTurns}; i++) messages.push(...turn(sid, i))
      messages.push({ id: sid + '-stream', role: 'assistant', timestamp: Date.now(), pending: true,
        parts: [{ type: 'text', text: '' }] })
      return {
        storedSessionId: sid, messages, branch: '', cwd: '', model: '', provider: '',
        reasoningEffort: '', serviceTier: '', fast: false, yolo: false, personality: '',
        busy: true, awaitingResponse: false, streamId: sid + '-stream', sawAssistantPayload: true,
        pendingBranchGroup: null, interrupted: false, interimBoundaryPending: false,
        needsInput: false, turnStartedAt: Date.now(), usage: null
      }
    }

    window.__RC__ = { ids: [], timer: null }
    for (let n = 1; n <= ${tiles}; n++) {
      const sid = 'churn-tile-' + n
      const rid = 'churn-rt-' + n
      window.__RC__.ids.push({ sid, rid })
      hook.open(sid, 'center')
      hook.patch(sid, { runtimeId: rid })
      hook.publish(rid, state(sid))
    }
    return 'ok'
  })()
`

const reveal = sid => `window.__HERMES_LAYOUT_TREE__.reveal(${JSON.stringify(`session-tile:${sid}`)})`

/** Grow every tile's streaming tail by `chunk` each `intervalMs`, through the
 *  same publish path the gateway's delta flush uses. */
const drive = (chunk, intervalMs, totalTokens) => `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    let pushed = 0
    const tick = () => {
      const states = hook.states()
      for (const { rid } of window.__RC__.ids) {
        const prev = states[rid]
        if (!prev) continue
        const messages = prev.messages.map(m => {
          if (m.id !== prev.streamId) return m
          const head = m.parts.slice(0, -1)
          const last = m.parts[m.parts.length - 1]
          return { ...m, parts: [...head, { type: 'text', text: last.text + ${JSON.stringify(chunk)} }] }
        })
        hook.publish(rid, { ...prev, messages })
      }
      pushed += 1
      if (pushed < ${totalTokens}) window.__RC__.timer = setTimeout(tick, ${intervalMs})
      else window.__RC__.done = true
    }
    window.__RC__.timer = setTimeout(tick, ${intervalMs})
    return 'driving'
  })()
`

/** Wait until the renderer stops committing on its own, so the recording window
 *  captures STREAMING cost and not whatever boot/hydration work happened to
 *  still be in flight. Returns `quiet:N` once commits hold still for `quietMs`.
 *
 *  If it returns `timeout:...` the app never went idle at all — with tiles
 *  marked busy and NO driver running, that means something is ticking on its
 *  own. The report of what rendered during the wait is attached so the culprit
 *  is named rather than guessed at. */
const quiesce = (quietMs, timeoutMs) => `
  (async () => {
    const rc = window.__RENDER_COUNTS__
    rc.start()
    const deadline = Date.now() + ${timeoutMs}
    const startedAt = Date.now()
    let last = -1
    let stableSince = Date.now()
    while (Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 100))
      const n = rc.commits()
      if (n !== last) { last = n; stableSince = Date.now(); continue }
      if (Date.now() - stableSince >= ${quietMs}) { rc.stop(); return 'quiet:' + n }
    }
    const idle = {
      commits: last,
      seconds: (Date.now() - startedAt) / 1000,
      top: rc.report(8),
      // Who OWNS the update? The component whose own hook state changed with
      // no changed props is the root of a churn cascade; everything under it
      // is collateral. Naming it is the difference between fixing the cause
      // and memoizing a symptom.
      owners: rc.report(200).filter(r => r.stateChanged > 0 && r.propsChanged === 0).slice(0, 8)
    }
    rc.stop()
    return 'timeout:' + JSON.stringify(idle)
  })()
`

const START = `
  (() => {
    window.__RENDER_COUNTS__.start()
    window.__ATOM_CHURN__.start()
    return 'recording'
  })()
`

const COLLECT = `
  (() => {
    window.__RENDER_COUNTS__.stop()
    window.__ATOM_CHURN__.stop()
    return JSON.stringify({
      commits: window.__RENDER_COUNTS__.commits(),
      renders: window.__RENDER_COUNTS__.report(200),
      atoms: window.__ATOM_CHURN__.report(200)
    })
  })()
`

const CLEANUP = `
  (() => {
    if (window.__RC__) {
      clearTimeout(window.__RC__.timer)
      for (const { sid, rid } of window.__RC__.ids) {
        const states = window.__HERMES_SESSION_TILES__.states()
        window.__HERMES_SESSION_TILES__.publish(rid, { ...states[rid], busy: false, streamId: null })
        window.__HERMES_SESSION_TILES__.close(sid)
      }
      window.__RC__ = null
    }
    window.__RENDER_COUNTS__.clear()
    window.__ATOM_CHURN__.clear()
    return 'cleaned'
  })()
`

export default {
  name: 'render-churn',
  tier: 'ci',
  description: 'N streaming tabs: per-component render attribution + store churn.',
  async run(cdp, opts = {}) {
    const tiles = Number(opts.tiles ?? 5)
    const seedTurns = Number(opts.turns ?? 20)
    const tokens = Number(opts.tokens ?? 240)
    // Matches STREAM_DELTA_FLUSH_MS — one publish per session per real flush.
    const intervalMs = Number(opts.intervalMs ?? 33)
    const chunk = opts.chunk ?? 'A streamed review sentence with **bold** and `code`.\n\n'

    await cdp.send('Runtime.enable')

    const ok = await cdp.eval(setup(tiles, seedTurns))

    if (ok !== 'ok') {
      throw new Error(
        `render-churn setup failed (${ok}) — needs a dev renderer with src/debug installed ` +
          '(the counters are aliased out of production builds unless VITE_PERF_PROBE=1).'
      )
    }

    // Mount every tab (keep-alive mounts on first activation), then settle.
    for (let n = 1; n <= tiles; n++) {
      await cdp.eval(reveal(`churn-tile-${n}`))
      await sleep(350)
    }

    // Let the app go quiet before recording, so boot/hydration commits that
    // happen to still be in flight don't land in the streaming window. This is
    // what makes runs comparable — a fixed sleep let 2-4x of hydration churn
    // leak in depending on machine load.
    const settle = await cdp.eval(quiesce(600, 15000))
    await cdp.eval(START)
    await cdp.eval(drive(chunk, intervalMs, tokens))
    await sleep(tokens * intervalMs + 1500)

    const data = JSON.parse(await cdp.eval(COLLECT))
    await cdp.eval(CLEANUP)

    const byName = new Map(data.renders.map(r => [r.name, r]))
    const sidebarRows = SIDEBAR_COMPONENTS.map(n => byName.get(n)).filter(Boolean)
    const sidebarRenders = sidebarRows.reduce((a, r) => a + r.renders, 0)
    const sidebarWasted = sidebarRows.reduce((a, r) => a + r.wasted, 0)
    const totalRenders = data.renders.reduce((a, r) => a + r.renders, 0)
    const totalWasted = data.renders.reduce((a, r) => a + r.wasted, 0)
    const atomWasted = data.atoms.reduce((a, r) => a + r.wasted, 0)

    return {
      metrics: {
        // The hypothesis, as a number: sidebar renders while background tabs
        // stream. Should be 0.
        sidebar_renders: sidebarRenders,
        sidebar_wasted: sidebarWasted,
        // Renders with no changed props and no changed hook state — pure
        // parent-driven work, across the whole tree.
        wasted_renders: totalWasted,
        total_renders: totalRenders,
        commits: data.commits,
        // Store notifications that published a value equal to the last one.
        wasted_notifies: atomWasted
      },
      detail: {
        tiles,
        tokens,
        // 'quiet:N' = the app went idle before recording (comparable run).
        // 'timeout:N' = it never did, so boot churn is mixed into the numbers.
        settle,
        sidebar: sidebarRows,
        topRenders: data.renders.slice(0, 15),
        topAtoms: data.atoms.slice(0, 15)
      }
    }
  }
}
