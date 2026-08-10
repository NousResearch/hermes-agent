// Multi-tab working sessions: N session tiles stacked as tabs in the main
// zone, EVERY tab mounted (keep-alive), all streaming concurrently — the
// "5 tabs doing PR review" workload. Measures frame pacing + longtasks while
// the whole stack streams, which is where multitab renderers crawl.
//
// --zones M splits the tiles across M VISIBLE split zones (a 2×2 grid for 4)
// instead of one tab stack — the "4 tiles with 4 sessions each" workload,
// where M transcripts stream on screen at once and the rest are mounted
// keep-alive tabs behind them. --streaming S caps how many sessions are
// actually mid-turn (zone leaders first, so S=zones means "every visible
// transcript streams, every hidden tab idles"); the rest sit settled.
// --sessions N seeds a populated recents list (a lived-in sessions DB).
//
// Drives the real pipeline synthetically (no backend, no credits): each tick
// routes one delta per streaming session through `hook.update` — the same
// wiring-cache write (journal + publish + view sync) the gateway's delta
// flush performs — via the __HERMES_SESSION_TILES__ hook.
//
//   node scripts/perf/run.mjs multitab --spawn [--tiles 5] [--tokens 240]
//   node scripts/perf/run.mjs multitab --spawn --tiles 16 --zones 4 --sessions 300

import { sleep } from '../lib/cdp.mjs'
import { frameHistogram, percentile } from '../lib/stats.mjs'

// Same recorder pattern as stream.mjs (generation-guarded rAF + longtasks).
const RECORDERS = `
  (() => {
    window.__FT_GEN__ = (window.__FT_GEN__ || 0) + 1
    const ftGen = window.__FT_GEN__
    window.__FT__ = { times: [], stop: false }
    let last = performance.now()
    const tick = () => {
      if (window.__FT_GEN__ !== ftGen || window.__FT__.stop) return
      const now = performance.now()
      window.__FT__.times.push(now - last)
      last = now
      requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)

    window.__LT__ = { entries: [], stop: false }
    try {
      const po = new PerformanceObserver((list) => {
        if (window.__LT__.stop) return
        for (const e of list.getEntries()) window.__LT__.entries.push({ duration: e.duration, startTime: e.startTime })
      })
      po.observe({ entryTypes: ['longtask'] })
      window.__LT__.po = po
    } catch {}
    return 'armed'
  })()
`

const COLLECT = `
  (() => {
    window.__FT__.stop = true
    window.__LT__.stop = true
    try { window.__LT__.po && window.__LT__.po.disconnect() } catch {}
    return JSON.stringify({ frames: window.__FT__.times, longtasks: window.__LT__.entries })
  })()
`

/** Page-side setup: open `tiles` session tiles — one tab stack in the main
 *  zone (zones=1), or spread across `zones` visible splits (a 2×2 grid for 4)
 *  — bind fake runtime ids, and seed each with a realistic transcript.
 *
 *  States are written through `hook.update` — the REAL gateway write path
 *  (wiring cache + in-flight journal + publish + view sync). Driving
 *  `hook.publish` alone under-models a stream: it skips the journal and the
 *  cache, which is exactly where multi-session cost used to hide. */
const setup = (tiles, seedTurns, streamSeed, zones, seedSessions, streaming, dead) => `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    if (!hook) return 'no-hook'
    if (!hook.update) return 'no-update-hook'

    const turn = (sid, i) => ([
      { id: sid + '-u' + i, role: 'user', timestamp: Date.now(),
        parts: [{ type: 'text', text: 'Review question ' + i + ': does the diff in module ' + i + ' handle the error path?' }] },
      { id: sid + '-a' + i, role: 'assistant', timestamp: Date.now(), pending: false,
        parts: [{ type: 'text', text: [
          '## Finding ' + i, '',
          'The handler swallows the rejection. Key points for hunk \\\`' + i + '\\\`:', '',
          '- The catch block drops the original error.',
          '- Retries are unbounded — see [the loop](https://example.com/loop).', '',
          '\\\`\\\`\\\`ts',
          'async function retry' + i + '(fn: () => Promise<void>) {',
          '  for (;;) { try { return await fn() } catch {} }',
          '}',
          '\\\`\\\`\\\`', '',
          '| path | covered |', '|---|---|', '| happy | yes |', '| error | no |', ''
        ].join('\\n') }] }
    ])

    const state = (sid, rid, isStreaming) => {
      const messages = []
      for (let i = 0; i < ${seedTurns}; i++) messages.push(...turn(sid, i))
      // Streaming tail the driver grows (--code seeds an open fence); a
      // non-streaming session sits settled — open, mounted, mid-nothing.
      if (isStreaming) {
        messages.push({ id: sid + '-stream', role: 'assistant', timestamp: Date.now(), pending: true,
          parts: [{ type: 'text', text: ${JSON.stringify(streamSeed)} }] })
      }
      return {
        storedSessionId: sid, messages, branch: '', cwd: '', model: '', provider: '',
        reasoningEffort: '', serviceTier: '', fast: false, yolo: false, personality: '',
        busy: isStreaming, awaitingResponse: false,
        streamId: isStreaming ? sid + '-stream' : null, sawAssistantPayload: true,
        pendingBranchGroup: null, interrupted: false, interimBoundaryPending: false,
        needsInput: false, turnStartedAt: isStreaming ? Date.now() : null, usage: null
      }
    }

    // A populated recents list (--sessions): every store publish re-runs the
    // busy/attention/draft projections against it, so an empty list hides
    // that scaling. Restored by CLEANUP.
    if (${seedSessions} > 0) {
      window.__MT_SAVED_SESSIONS__ = hook.sessions()
      const rows = []
      for (let i = 0; i < ${seedSessions}; i++) {
        rows.push({
          id: 'perf-row-' + i, title: 'Seeded session ' + i, ended_at: null,
          input_tokens: 1200, output_tokens: 800, is_active: false,
          last_active: Date.now() - i * 60000, message_count: 12,
          model: 'hermes-4', preview: 'seeded row', cwd: '/tmp/proj-' + (i % 7)
        })
      }
      hook.seedSessions(rows)
    }

    // Leaked residue (--dead): sessions that ran with no surface referencing
    // them and then settled — what a day of opening and closing tiles
    // accumulates. Modeled on the real path (insert while busy, then the
    // settle publish) so publish-time eviction, where present, engages.
    // CLEANUP drops whatever survives, for builds without eviction.
    window.__MT_DEAD__ = []
    for (let d = 0; d < ${dead}; d++) {
      const sid = 'perf-dead-' + d
      const rid = 'perf-dead-rt-' + d
      window.__MT_DEAD__.push(rid)
      const settled = state(sid, rid, false)
      hook.publish(rid, { ...settled, busy: true })
      hook.publish(rid, settled)
    }

    // Zone leaders open as visible splits (right of the workspace, then
    // subdividing that column into a grid); followers stack as tabs into
    // their zone. zones=1 keeps the classic one-stack workload.
    const perZone = Math.ceil(${tiles} / ${zones})
    const leaders = []

    // Streaming slots go to zone LEADERS first (rank orders round-robin across
    // zones), so --streaming ${'$'}{zones} means "every VISIBLE transcript streams,
    // every hidden tab idles" — the split the all-vs-visible snapshots diff.
    window.__MT__ = { ids: [], leaders, streaming: [], timer: null }
    for (let n = 1; n <= ${tiles}; n++) {
      const sid = 'perf-tile-' + n
      const rid = 'perf-rt-' + n
      window.__MT__.ids.push({ sid, rid })
      const zone = ${zones} > 1 ? Math.floor((n - 1) / perZone) : 0
      const posInZone = ${zones} > 1 ? (n - 1) % perZone : n - 1
      const rank = posInZone * ${zones} + zone
      const isStreaming = rank < ${streaming}
      if (isStreaming) window.__MT__.streaming.push(rid)
      const leader = leaders[zone]
      if (leader) {
        hook.open(sid, 'center', 'session-tile:' + leader)
      } else if (${zones} === 1) {
        hook.open(sid, 'center')
      } else {
        leaders[zone] = sid
        if (zone === 0) hook.open(sid, 'right')
        else if (zone === 1) hook.open(sid, 'bottom', 'session-tile:' + leaders[0])
        else hook.open(sid, 'right', 'session-tile:' + leaders[zone - 2])
      }
      hook.patch(sid, { runtimeId: rid })
      hook.update(rid, () => state(sid, rid, isStreaming))
    }
    return 'ok'
  })()
`

// Activate every tab once so keep-alive mounts the full stack (lazy mount:
// a never-activated tab stays unmounted, which would understate the cost).
const reveal = sid => `window.__HERMES_LAYOUT_TREE__.reveal(${JSON.stringify(`session-tile:${sid}`)})`

/** Page-side driver: grow every tile's streaming tail by `chunk` each
 *  `intervalMs`, through the same write path the gateway flush uses. */
const drive = (chunk, intervalMs, totalTokens) => `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    let pushed = 0
    const tick = () => {
      for (const rid of window.__MT__.streaming) {
        hook.update(rid, prev => {
          if (!prev.streamId) return prev
          const messages = prev.messages.map(m => {
            if (m.id !== prev.streamId) return m
            const head = m.parts.slice(0, -1)
            const last = m.parts[m.parts.length - 1]
            return { ...m, parts: [...head, { type: 'text', text: last.text + ${JSON.stringify(chunk)} }] }
          })
          return { ...prev, messages }
        })
      }
      pushed += 1
      if (pushed < ${totalTokens}) window.__MT__.timer = setTimeout(tick, ${intervalMs})
      else window.__MT__.done = true
    }
    window.__MT__.timer = setTimeout(tick, ${intervalMs})
    return 'driving'
  })()
`

const CLEANUP = `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    if (window.__MT_DEAD__) {
      for (const rid of window.__MT_DEAD__) hook.drop?.(rid)
      window.__MT_DEAD__ = null
    }
    if (window.__MT__) {
      clearTimeout(window.__MT__.timer)
      for (const { sid, rid } of window.__MT__.ids) {
        // Settle through the real path so the in-flight journal entry clears.
        hook.update(rid, prev => ({ ...prev, busy: false, streamId: null }))
        hook.close(sid)
      }
      window.__MT__ = null
    }
    if (window.__MT_SAVED_SESSIONS__) {
      hook.seedSessions(window.__MT_SAVED_SESSIONS__)
      window.__MT_SAVED_SESSIONS__ = null
    }
    return 'cleaned'
  })()
`

export default {
  name: 'multitab',
  tier: 'ci',
  description: 'N mounted session-tile tabs all streaming: frame pacing + longtasks.',
  async run(cdp, opts = {}) {
    const tiles = Number(opts.tiles ?? 5)
    const zones = Number(opts.zones ?? 1)
    const seedTurns = Number(opts.turns ?? 20)
    const seedSessions = Number(opts.sessions ?? 0)
    const streaming = Math.min(Number(opts.streaming ?? tiles), tiles)
    const dead = Number(opts.dead ?? 0)
    const tokens = Number(opts.tokens ?? 240)
    // Matches STREAM_DELTA_FLUSH_MS — one publish per session per real flush.
    const intervalMs = Number(opts.intervalMs ?? 33)
    // --code: every tile grows ONE giant fenced code block with no settle
    // boundaries — what a coding agent streams. The block re-parses and
    // re-renders fully every flush (block memoization can't settle it), the
    // documented worst case and the "5 tabs all coding" crawl.
    const chunk = opts.code
      ? '  const value = await resolve(ctx, { retry: true }) // step\n'
      : (opts.chunk ?? 'A streamed review sentence with **bold**, `code`, and ordinary prose.\n\n')
    const streamSeed = opts.code ? '```ts\n' : ''

    await cdp.send('Runtime.enable')

    const ok = await cdp.eval(setup(tiles, seedTurns, streamSeed, zones, seedSessions, streaming, dead))

    if (ok !== 'ok') {
      throw new Error(`multitab setup failed (${ok}) — dev hooks missing? (needs a dev/probe renderer)`)
    }

    // Mount every tab (keep-alive mounts on first activation), then settle.
    for (let n = 1; n <= tiles; n++) {
      await cdp.eval(reveal(`perf-tile-${n}`))
      await sleep(350)
    }

    // Front each zone's leader so the visible set is one transcript per zone
    // (the reveal loop above leaves each zone on its LAST tab).
    if (zones > 1) {
      const leaders = JSON.parse(await cdp.eval('JSON.stringify(window.__MT__.leaders)'))

      for (const sid of leaders) {
        await cdp.eval(reveal(sid))
        await sleep(150)
      }
    }

    await sleep(1000)
    await cdp.eval(RECORDERS)
    await cdp.eval(drive(chunk, intervalMs, tokens))
    await sleep(tokens * intervalMs + 1500)

    const data = JSON.parse(await cdp.eval(COLLECT))
    await cdp.eval(CLEANUP)

    // Drop the first 500ms (recorder install + settle).
    const frames = []
    let acc = 0

    for (const f of data.frames) {
      acc += f

      if (acc >= 500) {
        frames.push(f)
      }
    }

    const ltDurations = data.longtasks.map(e => e.duration)
    const windowS = frames.reduce((a, b) => a + b, 0) / 1000
    // The felt numbers: sustained fps over the window, and the fps of the
    // worst 1-second slice (a 333ms frame IS "3fps" even if the average looks
    // fine). Worst slice = max summed frame time in any sliding 1s window.
    const avgFps = windowS ? frames.length / windowS : 0
    let worstFps = avgFps

    for (let i = 0, j = 0, sum = 0; j < frames.length; j++) {
      sum += frames[j]

      while (sum > 1000) {
        sum -= frames[i++]
      }

      // Only a window that actually spans ~1s counts; short prefixes don't.
      if (sum >= 900) {
        worstFps = Math.min(worstFps, ((j - i + 1) / sum) * 1000)
      }
    }

    return {
      metrics: {
        longtasks_n: data.longtasks.length,
        longtask_max_ms: Math.round((ltDurations.length ? Math.max(...ltDurations) : 0) * 10) / 10,
        frame_p95_ms: Math.round(percentile(frames, 0.95) * 10) / 10,
        frame_p99_ms: Math.round(percentile(frames, 0.99) * 10) / 10,
        slow_frames_33: frames.filter(f => f > 33).length
      },
      detail: {
        tiles,
        zones,
        streaming,
        dead,
        sessions: seedSessions,
        code: Boolean(opts.code),
        windowS: Math.round(windowS * 10) / 10,
        avgFps: Math.round(avgFps * 10) / 10,
        worstSecondFps: Math.round(worstFps * 10) / 10,
        frameHistogram: frameHistogram(frames)
      }
    }
  }
}
