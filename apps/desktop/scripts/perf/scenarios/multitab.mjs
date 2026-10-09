// Multi-tab working sessions: N session tiles stacked as tabs in the main
// zone, EVERY tab mounted (keep-alive), all streaming concurrently — the
// "5 tabs doing PR review" workload. Measures frame pacing + longtasks while
// the whole stack streams, which is where multitab renderers crawl.
//
// Drives the real pipeline synthetically (no backend, no credits):
// publishSessionState per session per flush — exactly what the gateway's
// delta flush does — via the __HERMES_SESSION_TILES__ hook.
//
//   node scripts/perf/run.mjs multitab --spawn [--tiles 5] [--tokens 240]

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

/** Page-side setup: open `tiles` session tiles stacked into the main zone,
 *  bind fake runtime ids, and seed each with a realistic transcript. */
const setup = (tiles, seedTurns, streamSeed) => `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    if (!hook) return 'no-hook'

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

    const state = (sid, rid) => {
      const messages = []
      for (let i = 0; i < ${seedTurns}; i++) messages.push(...turn(sid, i))
      // Streaming tail the driver grows (--code seeds an open fence).
      messages.push({ id: sid + '-stream', role: 'assistant', timestamp: Date.now(), pending: true,
        parts: [{ type: 'text', text: ${JSON.stringify(streamSeed)} }] })
      return {
        storedSessionId: sid, messages, branch: '', cwd: '', model: '', provider: '',
        reasoningEffort: '', serviceTier: '', fast: false, yolo: false, personality: '',
        busy: true, awaitingResponse: false, streamId: sid + '-stream', sawAssistantPayload: true,
        pendingBranchGroup: null, interrupted: false, interimBoundaryPending: false,
        needsInput: false, turnStartedAt: Date.now(), usage: null
      }
    }

    window.__MT__ = { ids: [], timer: null }
    for (let n = 1; n <= ${tiles}; n++) {
      const sid = 'perf-tile-' + n
      const rid = 'perf-rt-' + n
      window.__MT__.ids.push({ sid, rid })
      hook.open(sid, 'center')
      hook.patch(sid, { runtimeId: rid })
      hook.publish(rid, state(sid, rid))
    }
    return 'ok'
  })()
`

// Activate every tab once so keep-alive mounts the full stack (lazy mount:
// a never-activated tab stays unmounted, which would understate the cost).
const reveal = sid => `window.__HERMES_LAYOUT_TREE__.reveal(${JSON.stringify(`session-tile:${sid}`)})`

/** Page-side driver: grow every tile's streaming tail by `chunk` each
 *  `intervalMs`, through the same publish path the gateway flush uses. */
const drive = (chunk, intervalMs, totalTokens) => `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    let pushed = 0
    const tick = () => {
      const states = hook.states()
      for (const { rid } of window.__MT__.ids) {
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
      if (pushed < ${totalTokens}) window.__MT__.timer = setTimeout(tick, ${intervalMs})
      else window.__MT__.done = true
    }
    window.__MT__.timer = setTimeout(tick, ${intervalMs})
    return 'driving'
  })()
`

const CLEANUP = `
  (() => {
    if (window.__MT__) {
      clearTimeout(window.__MT__.timer)
      for (const { sid, rid } of window.__MT__.ids) {
        window.__HERMES_SESSION_TILES__.publish(rid, {
          ...window.__HERMES_SESSION_TILES__.states()[rid], busy: false, streamId: null
        })
        window.__HERMES_SESSION_TILES__.close(sid)
      }
      window.__MT__ = null
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
    const seedTurns = Number(opts.turns ?? 20)
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

    const ok = await cdp.eval(setup(tiles, seedTurns, streamSeed))

    if (ok !== 'ok') {
      throw new Error(`multitab setup failed (${ok}) — dev hooks missing? (needs a dev/probe renderer)`)
    }

    // Mount every tab (keep-alive mounts on first activation), then settle.
    for (let n = 1; n <= tiles; n++) {
      await cdp.eval(reveal(`perf-tile-${n}`))
      await sleep(350)
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
        code: Boolean(opts.code),
        windowS: Math.round(windowS * 10) / 10,
        avgFps: Math.round(avgFps * 10) / 10,
        worstSecondFps: Math.round(worstFps * 10) / 10,
        frameHistogram: frameHistogram(frames)
      }
    }
  }
}
