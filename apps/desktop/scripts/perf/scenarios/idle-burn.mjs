// What does the app cost, in CPU, while nothing is happening?
//
// `idle-cost` counts React commits. That misses most of what an idle desktop
// actually burns: CSS animations, style recalc and layout, compositor frames
// in the GPU process, and timers that never reach React. Users report those
// as fans and battery (#53902, #88288, #89732, #121735, #122413), and no
// scenario measured them, so each regression was found from a user report
// weeks after it merged (tracker #127647).
//
// This scenario measures the real processes, per window state:
//   viewed-quiet   window shown, no session busy
//   viewed-busy    window shown, N sessions busy with no tokens arriving
//   hidden-busy    the same busy sessions with the window hidden. The
//                  scenario minimizes through the app's own window control;
//                  where the platform ignores a programmatic minimize (seen
//                  on macOS for a background-launched instance), it forces
//                  the hidden-window pause attribute instead, which exercises
//                  the CSS pause but not the JS loops that read window state.
//                  `detail.hidden_method` records which one ran.
//
// Per phase it reports CPU as % of one core for the renderer, GPU and browser
// (main) processes, from Chromium's cumulative per-process CPU time
// (`SystemInfo.getProcessInfo`); the renderer's style recalcs, layouts and
// main-thread task time per second (`Performance.getMetrics`); and which
// animations are running (`document.getAnimations()`), which names what is
// live when nothing should be.
//
// `--attribute` then pauses every running animation, and then each one by
// name, through the Web Animations API with the window shown and sessions
// busy, re-measures and resumes it, so each animation's cost is its own
// delta against viewed-busy (the #89732 method, without guessing
// selectors). The "every animation" row separates animation cost from
// everything else the renderer is doing.
//
//   node scripts/perf/run.mjs idle-burn --spawn [--tiles 3] [--seconds 30]
//        [--settle 10] [--attribute] [--runs 3]
//
// The harness launches Electron with its anti-throttle flags. That matches
// the app while any turn is streaming, when throttling is off for every
// window (#93904). Numbers are CPU on the machine that ran them: compare base
// and head on one machine in one session, never across machines.

import { CDP, sleep } from '../lib/cdp.mjs'

import { BUSY_TILES_CLEANUP, reveal, seedBusyTiles } from './idle-cost.mjs'

const round = (n, places = 1) => Math.round(n * 10 ** places) / 10 ** places

const PAUSED_ATTRIBUTE = 'data-renderer-animations-paused'
const IS_PAUSED = `document.documentElement.hasAttribute('${PAUSED_ATTRIBUTE}')`

/** The browser-level CDP session: per-process CPU lives there, not on the page. */
async function openBrowserSession(port) {
  const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()

  return CDP.open(version.webSocketDebuggerUrl)
}

/** Cumulative CPU seconds per process type, summed across processes of that type. */
async function cpuSeconds(browser) {
  const { processInfo } = await browser.send('SystemInfo.getProcessInfo')
  const byType = {}

  for (const { type, cpuTime } of processInfo) {
    byType[type] = (byType[type] ?? 0) + cpuTime
  }

  return byType
}

/** Cumulative renderer counters since `Performance.enable`. */
async function rendererCounters(cdp) {
  const { metrics } = await cdp.send('Performance.getMetrics')
  const value = name => metrics.find(m => m.name === name)?.value ?? 0

  return { layouts: value('LayoutCount'), recalcs: value('RecalcStyleCount'), taskSeconds: value('TaskDuration') }
}

// Running CSS animations grouped by name, with one example target.
const LIVE_ANIMATIONS = `
  (() => {
    const describe = (el, pseudo) => {
      if (!el) return ''
      const cls = typeof el.className === 'string' ? el.className.trim().split(/\\s+/).slice(0, 3).join('.') : ''
      return el.tagName.toLowerCase() + (cls ? '.' + cls : '') + (pseudo ?? '')
    }
    const byName = {}
    for (const a of document.getAnimations()) {
      if (a.playState !== 'running' || !a.animationName) continue
      byName[a.animationName] ??= { name: a.animationName, count: 0, example: describe(a.effect?.target, a.effect?.pseudoElement) }
      byName[a.animationName].count++
    }
    return JSON.stringify(Object.values(byName).sort((x, y) => y.count - x.count))
  })()
`

// `name` null means every running CSS animation. Resuming with `play()`
// detaches the animation from CSS `animation-play-state` for good, so
// attribution runs after every CSS-driven phase.
const setPlaying = (name, playing) => `
  (() => {
    let n = 0
    for (const a of document.getAnimations()) {
      if (!a.animationName || (${JSON.stringify(name)} !== null && a.animationName !== ${JSON.stringify(name)})) continue
      ${playing ? 'a.play()' : 'a.pause()'}
      n++
    }
    return n
  })()
`

const liveAnimations = async cdp => JSON.parse(await cdp.eval(LIVE_ANIMATIONS))

// Sidebar rows for the busy tiles: the running arc (`arc-border`) and the
// row's status motion render from the session list, so without rows a busy
// tile animates nothing the user would see. Restored by RESTORE_SESSIONS.
const seedSessionRows = tiles => `
  (() => {
    const hook = window.__HERMES_SESSION_TILES__
    window.__IDLE_BURN_SAVED_SESSIONS__ = hook.sessions()
    const rows = []
    for (let n = 1; n <= ${tiles}; n++) {
      rows.push({
        id: 'idle-tile-' + n, title: 'Busy session ' + n, ended_at: null, input_tokens: 1200,
        output_tokens: 800, is_active: true, last_active: Date.now(), message_count: 9,
        model: 'hermes-4', preview: 'working', cwd: '/home/perf/proj'
      })
    }
    hook.seedSessions(rows)
    return rows.length
  })()
`

const RESTORE_SESSIONS = `
  (() => {
    if (window.__IDLE_BURN_SAVED_SESSIONS__) {
      window.__HERMES_SESSION_TILES__.seedSessions(window.__IDLE_BURN_SAVED_SESSIONS__)
      window.__IDLE_BURN_SAVED_SESSIONS__ = null
    }
    return 'restored'
  })()
`

/** Measure one phase: `seconds` of wall clock with nothing driven. */
async function measurePhase(cdp, browser, seconds) {
  const cpu0 = await cpuSeconds(browser)
  const r0 = await rendererCounters(cdp)
  const t0 = performance.now()

  await sleep(seconds * 1000)

  const wall = (performance.now() - t0) / 1000
  const cpu1 = await cpuSeconds(browser)
  const r1 = await rendererCounters(cdp)
  const pct = type => round((((cpu1[type] ?? 0) - (cpu0[type] ?? 0)) / wall) * 100)

  return {
    renderer_cpu_pct: pct('renderer'),
    gpu_cpu_pct: pct('GPU'),
    browser_cpu_pct: pct('browser'),
    recalcs_per_s: round((r1.recalcs - r0.recalcs) / wall),
    layouts_per_s: round((r1.layouts - r0.layouts) / wall),
    task_ms_per_s: round(((r1.taskSeconds - r0.taskSeconds) / wall) * 1000)
  }
}

async function waitForPaused(cdp, paused, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if ((await cdp.eval(IS_PAUSED)) === paused) {
      return true
    }

    await sleep(200)
  }

  return false
}

/** A dev renderer can reload under the harness (HMR, auto-reload); wait for its debug hooks. */
async function waitForHooks(cdp, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if (await cdp.eval('!!(window.__HERMES_SESSION_TILES__ && window.__RENDER_COUNTS__)').catch(() => false)) {
      return
    }

    await sleep(500)
  }

  throw new Error('idle-burn: the renderer debug hooks never appeared; needs a dev renderer with src/debug installed.')
}

/** Bring the minimized window back: maximize restores it, the second toggle un-maximizes. */
async function restoreWindow(cdp) {
  await cdp.eval('window.hermesDesktop.windowControls.toggleMaximize()')
  await sleep(500)
  await cdp.eval('window.hermesDesktop.windowControls.toggleMaximize()')

  return waitForPaused(cdp, false)
}

/** Pause each running animation in turn and report what the process CPU drops by. */
async function attributeAnimations(cdp, browser, baseline, animations, seconds) {
  const rows = []

  const all = { name: null, count: animations.reduce((sum, a) => sum + a.count, 0), example: '(every animation)' }

  for (const { name, count, example } of animations.length > 1 ? [all, ...animations] : animations) {
    if ((await cdp.eval(setPlaying(name, false))) === 0) continue

    await sleep(2000)
    const paused = await measurePhase(cdp, browser, seconds)
    await cdp.eval(setPlaying(name, true))
    await sleep(2000)

    rows.push({
      name,
      count,
      example,
      renderer_saved_pct: round(baseline.renderer_cpu_pct - paused.renderer_cpu_pct),
      gpu_saved_pct: round(baseline.gpu_cpu_pct - paused.gpu_cpu_pct),
      recalcs_saved_per_s: round(baseline.recalcs_per_s - paused.recalcs_per_s)
    })
  }

  return rows
}

export default {
  name: 'idle-burn',
  // Report-only: absolute CPU % depends on the machine. Base/head pairs on
  // one machine are the unit of evidence (see the header).
  tier: 'report',
  description: 'Per-process CPU, style/layout work and live animations while idle: viewed, viewed busy, hidden.',
  async run(cdp, opts = {}) {
    const port = Number(opts.port ?? 9222)
    const tiles = Number(opts.tiles ?? 3)
    const seconds = Number(opts.seconds ?? 30)
    const settle = Number(opts.settle ?? 10)
    const browser = await openBrowserSession(port)
    const live = {}
    const phases = {}
    let attribution = []

    try {
      await cdp.send('Performance.enable')

      if (await cdp.eval(IS_PAUSED)) {
        throw new Error('idle-burn: the window starts hidden; restore it before measuring.')
      }

      await waitForHooks(cdp)
      await sleep(settle * 1000)
      live.viewed_quiet = await liveAnimations(cdp)
      phases.viewed_quiet = await measurePhase(cdp, browser, seconds)

      const seeded = await cdp.eval(seedBusyTiles(tiles, 4))

      if (seeded !== 'ok') {
        throw new Error(`idle-burn: busy-tile setup failed (${seeded}); needs a dev renderer with src/debug installed.`)
      }

      await cdp.eval(seedSessionRows(tiles))

      for (let n = 1; n <= tiles; n++) {
        await cdp.eval(reveal(`idle-tile-${n}`))
        await sleep(300)
      }

      await sleep(settle * 1000)
      live.viewed_busy = await liveAnimations(cdp)
      phases.viewed_busy = await measurePhase(cdp, browser, seconds)

      await cdp.eval('window.hermesDesktop.windowControls.minimize()')

      const minimized = await waitForPaused(cdp, true, 5000)
      const hiddenMethod = minimized ? 'minimize' : 'forced-attribute'

      if (!minimized) {
        await cdp.eval(`document.documentElement.setAttribute('${PAUSED_ATTRIBUTE}', '')`)
      }

      await sleep(settle * 1000)
      live.hidden_busy = await liveAnimations(cdp)
      phases.hidden_busy = await measurePhase(cdp, browser, seconds)

      if (!minimized) {
        await cdp.eval(`document.documentElement.removeAttribute('${PAUSED_ATTRIBUTE}')`)
      }

      const restored = minimized ? await restoreWindow(cdp) : true

      if (opts.attribute) {
        attribution = await attributeAnimations(cdp, browser, phases.viewed_busy, live.viewed_busy, seconds)
      }

      await cdp.eval(BUSY_TILES_CLEANUP)
      await cdp.eval(RESTORE_SESSIONS)

      const metrics = {}

      for (const [phase, values] of Object.entries(phases)) {
        for (const [key, value] of Object.entries(values)) {
          metrics[`${phase}__${key}`] = value
        }

        metrics[`${phase}__running_animations`] = live[phase].reduce((sum, a) => sum + a.count, 0)
      }

      return { metrics, detail: { tiles, seconds, settle, hidden_method: hiddenMethod, restored, live, attribution } }
    } finally {
      browser.close()
    }
  }
}
