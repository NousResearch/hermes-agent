// File-tree + terminal workspace stress. This is the regression scene for the
// desktop symptom where opening the project tree/terminal made the whole page
// hitch while chat and PTY output continued.
//
// It mounts a real project tree, one PTY plus multiple persistent xterm tabs,
// streams chat and terminal output together, mutates Git decoration state, and
// drags the terminal split. The debug probe records the specific work we care
// about rather than inferring it from CPU alone:
//   - fixed-overlay measurements
//   - active/hidden xterm fits
//   - ProjectTree + per-path row renders
//   - frame pacing / slow frames
//
//   npm run perf -- right-pane --spawn --prod --runs 3

import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { sleep } from '../lib/cdp.mjs'
import { frameHistogram, percentile } from '../lib/stats.mjs'

const DEFAULT_CWD = resolve(dirname(fileURLToPath(import.meta.url)), '../../..')

const RECORDERS = `
  (() => {
    window.__RP_FRAME_GEN__ = (window.__RP_FRAME_GEN__ || 0) + 1
    const generation = window.__RP_FRAME_GEN__
    window.__RP_FRAMES__ = { times: [], stop: false }
    let last = performance.now()
    const tick = () => {
      if (window.__RP_FRAME_GEN__ !== generation || window.__RP_FRAMES__.stop) return
      const now = performance.now()
      window.__RP_FRAMES__.times.push(now - last)
      last = now
      requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)

    window.__RP_LONG__ = { entries: [], stop: false }
    try {
      const observer = new PerformanceObserver(list => {
        if (window.__RP_LONG__.stop) return
        for (const entry of list.getEntries()) {
          window.__RP_LONG__.entries.push({ duration: entry.duration, startTime: entry.startTime })
        }
      })
      observer.observe({ entryTypes: ['longtask'] })
      window.__RP_LONG__.observer = observer
    } catch {}
    return 'armed'
  })()
`

const COLLECT_RECORDERS = `
  (() => {
    window.__RP_FRAMES__.stop = true
    window.__RP_LONG__.stop = true
    try { window.__RP_LONG__.observer && window.__RP_LONG__.observer.disconnect() } catch {}
    return JSON.stringify({ frames: window.__RP_FRAMES__.times, longtasks: window.__RP_LONG__.entries })
  })()
`

const START_COUNTERS = `window.__RIGHT_PANE_PERF__.start(); 'recording'`
const SNAPSHOT_COUNTERS = `
  (() => {
    window.__RIGHT_PANE_PERF__.stop()
    return JSON.stringify(window.__RIGHT_PANE_PERF__.snapshot())
  })()
`

const DRAG_TERMINAL_SPLIT = `
  (async () => {
    const slot = document.querySelector('[data-terminal-slot]')
    const overlay = document.querySelector('[data-persistent-terminal]')
    if (!slot || !overlay) return JSON.stringify({ target: 'none', drift: -1, moved: 0 })

    const slotBox = slot.getBoundingClientRect()
    const candidates = [...document.querySelectorAll('[role="separator"]')]
      .map(element => ({ element, box: element.getBoundingClientRect() }))
      .filter(item => item.box.width > item.box.height * 3)
      .sort((a, b) =>
        Math.abs((a.box.top + a.box.bottom) / 2 - slotBox.top) -
        Math.abs((b.box.top + b.box.bottom) / 2 - slotBox.top)
      )
    const target = candidates[0]
    if (!target) return JSON.stringify({ target: 'none', drift: -1, moved: 0 })

    const x = target.box.left + target.box.width / 2
    const y0 = target.box.top + target.box.height / 2
    let y = y0
    const pointer = {
      bubbles: true, cancelable: true, pointerId: 91, pointerType: 'mouse',
      isPrimary: true, button: 0, buttons: 1
    }
    target.element.dispatchEvent(new PointerEvent('pointerdown', { ...pointer, clientX: x, clientY: y }))

    for (let i = 0; i < 24; i += 1) {
      y -= 1
      window.dispatchEvent(new PointerEvent('pointermove', { ...pointer, clientX: x, clientY: y }))
      await new Promise(resolve => requestAnimationFrame(resolve))
    }
    for (let i = 0; i < 24; i += 1) {
      y += 1
      window.dispatchEvent(new PointerEvent('pointermove', { ...pointer, clientX: x, clientY: y }))
      await new Promise(resolve => requestAnimationFrame(resolve))
    }
    window.dispatchEvent(new PointerEvent('pointerup', { ...pointer, buttons: 0, clientX: x, clientY: y }))
    // Track-size transitions continue briefly after pointerup. Wait through
    // that animation, then give the overlay its normal two-frame calibration.
    await new Promise(resolve => setTimeout(resolve, 350))
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))

    const a = slot.getBoundingClientRect()
    const b = overlay.getBoundingClientRect()
    const drift = Math.max(
      Math.abs(a.top - b.top),
      Math.abs(a.left - b.left),
      Math.abs(a.width - b.width),
      Math.abs(a.height - b.height)
    )
    return JSON.stringify({ target: 'horizontal-separator', drift, moved: 24 })
  })()
`

async function waitFor(cdp, expression, label, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if (await cdp.eval(expression)) {
      return
    }

    await sleep(100)
  }

  throw new Error(`right-pane timed out waiting for ${label}`)
}

const trimWarmup = (frames, warmupMs = 300) => {
  const kept = []
  let elapsed = 0

  for (const frame of frames) {
    elapsed += frame

    if (elapsed >= warmupMs) {
      kept.push(frame)
    }
  }

  return kept
}

export default {
  name: 'right-pane',
  tier: 'report',
  description: 'Project tree + persistent terminal tabs under chat/terminal output and split dragging.',
  async run(cdp, opts = {}) {
    const cwd = resolve(String(opts.cwd ?? DEFAULT_CWD))
    const terminalCount = Math.max(2, Number(opts.terminals ?? 3))
    const tokens = Number(opts.tokens ?? 90)
    const outputChunks = Number(opts.outputChunks ?? 160)

    await cdp.send('Runtime.enable')

    const ready = await cdp.eval(
      `!!(window.__PERF_DRIVE__?.rightPaneSetup && window.__RIGHT_PANE_PERF__ && window.__HERMES_LAYOUT_TREE__)`
    )

    if (!ready) {
      throw new Error('right-pane needs a dev renderer or a production build with VITE_PERF_PROBE=1.')
    }

    let setup

    try {
      setup = await cdp.eval(
        `window.__PERF_DRIVE__.rightPaneSetup(${JSON.stringify({ cwd, terminals: terminalCount })})`
      )
      await cdp.eval(`window.__HERMES_LAYOUT_TREE__.reveal('files'); window.__HERMES_LAYOUT_TREE__.reveal('terminal')`)
      await waitFor(cdp, `!!document.querySelector('[data-project-tree]')`, 'project tree')
      await waitFor(
        cdp,
        `!!document.querySelector('[data-terminal-slot]') && !!document.querySelector('[data-persistent-terminal]')`,
        'persistent terminal'
      )
      await waitFor(
        cdp,
        `document.querySelectorAll('[data-terminal] .xterm').length >= ${terminalCount}`,
        `${terminalCount} mounted xterms`,
        30000
      )
      await sleep(1200)

      // Activate every keep-alive tab once, then return to the output tab.
      // Each activation should restore exactly one fit; inactive tabs must stay
      // at zero even while another tab resizes or writes output.
      await cdp.eval(START_COUNTERS)

      for (const id of setup.terminalIds) {
        await cdp.eval(`window.__PERF_DRIVE__.rightPaneSelect(${JSON.stringify(id)})`)
        await sleep(180)
      }

      const activation = JSON.parse(await cdp.eval(SNAPSHOT_COUNTERS))

      await cdp.eval(RECORDERS)

      // Chat DOM churn is deliberately measured in its own counter window:
      // terminal positioning should receive no wakeups from transcript changes.
      await cdp.eval(START_COUNTERS)
      await cdp.eval(
        `window.__PERF_DRIVE__.stream({
          chunk: 'Right pane streaming sentence with **bold** and \`code\`.\\n\\n',
          intervalMs: 16,
          totalTokens: ${tokens},
          flushMinMs: 33
        })`
      )
      await cdp.eval(`
        (() => {
          let n = 0
          window.__RP_OUTPUT_TIMER__ = setInterval(() => {
            window.__PERF_DRIVE__.rightPaneWrite(
              ${JSON.stringify(setup.procId)},
              'terminal output line ' + n + ' ........................................\\r\\n'
            )
            n += 1
            if (n >= ${outputChunks}) clearInterval(window.__RP_OUTPUT_TIMER__)
          }, 16)
          return 'writing'
        })()
      `)
      await sleep(Math.max(tokens, outputChunks) * 16 + 900)
      const stream = JSON.parse(await cdp.eval(SNAPSHOT_COUNTERS))

      // An unrelated Git status publication should render neither the tree root
      // nor any visible row. A status for one visible path should touch only it.
      await cdp.eval(START_COUNTERS)
      await cdp.eval(`window.__PERF_DRIVE__.rightPaneGit('__right_pane_unrelated__.txt', 'modified')`)
      await sleep(250)
      const unrelatedGit = JSON.parse(await cdp.eval(SNAPSHOT_COUNTERS))

      const visiblePath = await cdp.eval(
        `document.querySelector('[data-project-tree] [title]')?.getAttribute('title') || ''`
      )
      let affectedGit = { counts: { 'project-tree-render': 0, 'project-tree-row-render': 0 }, rows: {} }

      if (visiblePath) {
        const relative = String(visiblePath).startsWith(`${cwd}/`)
          ? String(visiblePath).slice(cwd.length + 1)
          : String(visiblePath)
        await cdp.eval(START_COUNTERS)
        await cdp.eval(`window.__PERF_DRIVE__.rightPaneGit(${JSON.stringify(relative)}, 'modified')`)
        await sleep(250)
        affectedGit = JSON.parse(await cdp.eval(SNAPSHOT_COUNTERS))
      }

      await cdp.eval(START_COUNTERS)
      const drag = JSON.parse(await cdp.eval(DRAG_TERMINAL_SPLIT))
      const dragCounters = JSON.parse(await cdp.eval(SNAPSHOT_COUNTERS))
      const recorded = JSON.parse(await cdp.eval(COLLECT_RECORDERS))
      const frames = trimWarmup(recorded.frames)
      const longtasks = recorded.longtasks.map(entry => entry.duration)
      const streamCounts = stream.counts
      const activationCounts = activation.counts
      const unrelatedCounts = unrelatedGit.counts
      const affectedRows = Object.values(affectedGit.rows).reduce((sum, count) => sum + count, 0)
      const affectedPaths = Object.keys(affectedGit.rows).length

      if (drag.target === 'none') {
        throw new Error('right-pane found no horizontal terminal split separator.')
      }

      return {
        metrics: {
          chat_terminal_measures: streamCounts['terminal-measure'],
          hidden_terminal_fits: activationCounts['terminal-fit-hidden'] + streamCounts['terminal-fit-hidden'],
          activation_fit_mismatch: Math.abs(activationCounts['terminal-fit-active'] - setup.terminalIds.length),
          unrelated_tree_renders: unrelatedCounts['project-tree-render'],
          unrelated_row_renders: unrelatedCounts['project-tree-row-render'],
          affected_tree_renders: affectedGit.counts['project-tree-render'],
          affected_row_path_excess: Math.max(0, affectedPaths - 1),
          terminal_drift_px: Math.round(drag.drift * 10) / 10,
          frame_p95_ms: Math.round(percentile(frames, 0.95) * 10) / 10,
          frame_p99_ms: Math.round(percentile(frames, 0.99) * 10) / 10,
          slow_frames_33: frames.filter(frame => frame > 33).length,
          longtask_max_ms: Math.round((longtasks.length ? Math.max(...longtasks) : 0) * 10) / 10
        },
        detail: {
          cwd,
          terminals: setup.terminalIds.length,
          activation,
          stream,
          unrelatedGit,
          affectedGit,
          affectedRows,
          drag,
          dragCounters,
          frameHistogram: frameHistogram(frames),
          frames: frames.length
        }
      }
    } finally {
      await cdp.eval(`
        (() => {
          clearInterval(window.__RP_OUTPUT_TIMER__)
          window.__RIGHT_PANE_PERF__?.stop()
          window.__PERF_DRIVE__?.reset()
          return 'cleaned'
        })()
      `)
    }
  }
}
