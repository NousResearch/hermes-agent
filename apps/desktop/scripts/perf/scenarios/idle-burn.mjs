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
// (`SystemInfo.getProcessInfo`; summed over every process of the type, so
// "renderer" includes the app's other windows, and only processes alive for
// the whole window count); the renderer's style recalcs, layouts and
// main-thread task time per second (`Performance.getMetrics`); and which
// animations are running (`document.getAnimations()`), which names what is
// live when nothing should be.
//
// `--attribute` then re-measures a viewed-busy baseline and pauses every
// running animation, and then each one by name, through the Web Animations
// API, so each animation's cost is its own delta against that baseline (the
// #89732 method, without guessing selectors). The "every animation" row
// separates animation cost from everything else the renderer is doing.
// Resuming with `play()` detaches an animation from CSS
// `animation-play-state` for good, so attribution runs after every
// CSS-driven phase and the renderer reloads afterwards, which gives the next
// `--runs` iteration CSS-driven animations again.
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

// Mirrors src/lib/renderer-loop-pause.ts (an .mjs script cannot import it).
const RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE = 'data-renderer-animations-paused'
const IS_PAUSED = `document.documentElement.hasAttribute('${RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE}')`

// Forcing the pause for the hidden phase: the app re-syncs the attribute on
// every visibility or window-state event, so an observer holds it on.
const forcePaused = on => `
  (() => {
    const root = document.documentElement
    const attr = ${JSON.stringify(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)}
    window.__IDLE_BURN_FORCE__?.disconnect()
    window.__IDLE_BURN_FORCE__ = null
    root.toggleAttribute(attr, ${on})
    if (${on}) {
      const observer = new MutationObserver(() => root.hasAttribute(attr) || root.setAttribute(attr, ''))
      observer.observe(root, { attributes: true, attributeFilter: [attr] })
      window.__IDLE_BURN_FORCE__ = observer
    }
    return ${on}
  })()
`

/** The browser-level CDP session: per-process CPU lives there, not on the page. */
async function openBrowserSession(port) {
  const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()

  return CDP.open(version.webSocketDebuggerUrl)
}

/** Cumulative CPU seconds per process id, with its type. */
async function cpuSeconds(browser) {
  const { processInfo } = await browser.send('SystemInfo.getProcessInfo')

  return new Map(processInfo.map(({ id, type, cpuTime }) => [id, { type, cpuTime }]))
}

/** CPU seconds spent per type between two samples, over processes present in both. */
function cpuDelta(before, after) {
  const byType = {}

  for (const [id, { type, cpuTime }] of after) {
    const start = before.get(id)

    if (start) {
      byType[type] = (byType[type] ?? 0) + cpuTime - start.cpuTime
    }
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

// Pause the running CSS animations named `name` (null: all of them) and
// remember exactly those, so resuming touches nothing CSS had paused.
const pauseRunning = name => `
  (() => {
    const paused = document.getAnimations().filter(a => a.playState === 'running' && a.animationName &&
      (${JSON.stringify(name)} === null || a.animationName === ${JSON.stringify(name)}))
    for (const a of paused) a.pause()
    window.__IDLE_BURN_PAUSED__ = paused
    return paused.length
  })()
`

const RESUME_PAUSED = `
  (() => {
    for (const a of window.__IDLE_BURN_PAUSED__ ?? []) a.play()
    window.__IDLE_BURN_PAUSED__ = null
    return true
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
  const [cpu0, r0] = await Promise.all([cpuSeconds(browser), rendererCounters(cdp)])
  const t0 = performance.now()

  await sleep(seconds * 1000)

  const wall = (performance.now() - t0) / 1000
  const [cpu1, r1] = await Promise.all([cpuSeconds(browser), rendererCounters(cdp)])
  const cpu = cpuDelta(cpu0, cpu1)
  const pct = type => round(((cpu[type] ?? 0) / wall) * 100)

  return {
    renderer_cpu_pct: pct('renderer'),
    gpu_cpu_pct: pct('GPU'),
    browser_cpu_pct: pct('browser'),
    recalcs_per_s: round((r1.recalcs - r0.recalcs) / wall),
    layouts_per_s: round((r1.layouts - r0.layouts) / wall),
    task_ms_per_s: round(((r1.taskSeconds - r0.taskSeconds) / wall) * 1000)
  }
}

/** Poll `expression` until it is truthy; false on timeout. Eval errors (a reloading page) count as not yet. */
async function waitFor(cdp, expression, timeoutMs) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if (await cdp.eval(expression).catch(() => false)) {
      return true
    }

    await sleep(250)
  }

  return false
}

const waitForPaused = (cdp, paused, timeoutMs = 10000) => waitFor(cdp, paused ? IS_PAUSED : `!${IS_PAUSED}`, timeoutMs)

/** A dev renderer can reload under the harness (HMR, auto-reload); wait for its debug hooks. */
async function waitForHooks(cdp) {
  if (!(await waitFor(cdp, '!!(window.__HERMES_SESSION_TILES__ && window.__RENDER_COUNTS__)', 60000))) {
    throw new Error(
      'idle-burn: the renderer debug hooks never appeared; needs a dev renderer with src/debug installed.'
    )
  }
}

/** Bring the minimized window back: maximize restores it, the second toggle un-maximizes. */
async function restoreWindow(cdp) {
  await cdp.eval('window.hermesDesktop.windowControls.toggleMaximize()')
  await sleep(500)
  await cdp.eval('window.hermesDesktop.windowControls.toggleMaximize()')

  return waitForPaused(cdp, false)
}

/** Pause each running animation in turn and report what the process CPU drops by. */
async function attributeAnimations(cdp, browser, animations, seconds) {
  const rows = []
  const baseline = await measurePhase(cdp, browser, seconds)

  const all = { name: null, count: animations.reduce((sum, a) => sum + a.count, 0), example: '(every animation)' }

  for (const { name, count, example } of animations.length > 1 ? [all, ...animations] : animations) {
    if ((await cdp.eval(pauseRunning(name))) === 0) continue

    await sleep(2000)
    const paused = await measurePhase(cdp, browser, seconds)
    await cdp.eval(RESUME_PAUSED)

    rows.push({
      name,
      count,
      example,
      renderer_saved_pct: round(baseline.renderer_cpu_pct - paused.renderer_cpu_pct),
      gpu_saved_pct: round(baseline.gpu_cpu_pct - paused.gpu_cpu_pct),
      recalcs_saved_per_s: round(baseline.recalcs_per_s - paused.recalcs_per_s)
    })
  }

  return { baseline, rows }
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
    let attribution = null
    let hiddenMethod = null
    let minimized = false
    let restored = true
    let seeded = false

    try {
      await cdp.send('Performance.enable')

      if (await cdp.eval(IS_PAUSED)) {
        throw new Error('idle-burn: the window starts hidden; restore it before measuring.')
      }

      await waitForHooks(cdp)
      await sleep(settle * 1000)
      live.viewed_quiet = await liveAnimations(cdp)
      phases.viewed_quiet = await measurePhase(cdp, browser, seconds)

      const setup = await cdp.eval(seedBusyTiles(tiles, 4))

      if (setup !== 'ok') {
        throw new Error(`idle-burn: busy-tile setup failed (${setup}); needs a dev renderer with src/debug installed.`)
      }

      seeded = true
      await cdp.eval(seedSessionRows(tiles))

      for (let n = 1; n <= tiles; n++) {
        await cdp.eval(reveal(`idle-tile-${n}`))
        await sleep(300)
      }

      await sleep(settle * 1000)
      live.viewed_busy = await liveAnimations(cdp)
      phases.viewed_busy = await measurePhase(cdp, browser, seconds)

      await cdp.eval('window.hermesDesktop.windowControls.minimize()')
      minimized = await waitForPaused(cdp, true, 5000)
      hiddenMethod = minimized ? 'minimize' : 'forced-attribute'

      if (!minimized) {
        await cdp.eval(forcePaused(true))
      }

      await sleep(settle * 1000)
      live.hidden_busy = await liveAnimations(cdp)
      phases.hidden_busy = await measurePhase(cdp, browser, seconds)

      if (minimized) {
        restored = await restoreWindow(cdp)
        minimized = false
      } else {
        await cdp.eval(forcePaused(false))
      }

      if (opts.attribute && restored) {
        attribution = await attributeAnimations(cdp, browser, live.viewed_busy, seconds)
      }

      const metrics = {}

      for (const [phase, values] of Object.entries(phases)) {
        for (const [key, value] of Object.entries(values)) {
          metrics[`${phase}__${key}`] = value
        }

        metrics[`${phase}__running_animations`] = live[phase].reduce((sum, a) => sum + a.count, 0)
      }

      return { metrics, detail: { tiles, seconds, settle, hidden_method: hiddenMethod, restored, live, attribution } }
    } finally {
      // Leave the instance as found, even when a phase threw: the next
      // scenario (or --runs iteration) measures the same renderer.
      const quietly = expression => cdp.eval(expression).catch(() => undefined)
      await quietly(forcePaused(false))
      await quietly(RESUME_PAUSED)

      if (minimized) {
        await restoreWindow(cdp).catch(() => undefined)
      }

      if (seeded) {
        await quietly(BUSY_TILES_CLEANUP)
        await quietly(RESTORE_SESSIONS)
      }

      if (attribution) {
        // play() detached the attributed animations from CSS; a reload
        // recreates them CSS-driven for whatever runs next.
        await quietly('location.reload()')
        await sleep(1000)
        await waitForHooks(cdp).catch(() => undefined)
      }

      browser.close()
    }
  }
}
