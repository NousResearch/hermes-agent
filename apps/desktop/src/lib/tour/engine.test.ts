import { describe, expect, it } from 'vitest'

import { collectTourTargets } from './collect-targets'
import { runTourEngine, type TourDriver, type TourHolder } from './engine'

/** A recording fake of the driver.js surface the engine touches. */
function makeFactory(calls: string[]) {
  return (config?: object) => {
    const cfg = (config ?? {}) as {
      onDestroyed?: () => void
      onNextClick?: () => void
      onPrevClick?: () => void
      steps?: { onHighlightStarted?: () => void; waitForElement?: number }[]
    }

    const steps = cfg.steps ?? []
    let active = false
    let index = 0

    const enter = (at: number) => {
      index = at
      steps[at]?.onHighlightStarted?.()
    }

    const instance: TourDriver & { clickNext: () => void; clickPrev: () => void } = {
      // Stands in for the popover's own Next/Prev buttons. driver.js runs a
      // configured handler if one exists, otherwise its built-in advance —
      // both end up entering the step, which is what must trigger the move.
      clickNext: () => (cfg.onNextClick ? cfg.onNextClick() : instance.moveNext()),
      clickPrev: () => (cfg.onPrevClick ? cfg.onPrevClick() : instance.movePrevious()),
      destroy() {
        calls.push('destroy')
        active = false
        cfg.onDestroyed?.()
      },
      drive(stepIndex?: number) {
        calls.push(`drive:${stepIndex ?? 0}`)
        active = true
        enter(stepIndex ?? 0)
      },
      getActiveIndex: () => index,
      highlight(step: object) {
        calls.push(`highlight:${JSON.stringify(step)}`)
        active = true
        ;(step as { onHighlightStarted?: () => void }).onHighlightStarted?.()
      },
      isActive: () => active,
      isLastStep: () => index >= steps.length - 1,
      moveNext() {
        calls.push('next')
        enter(index + 1)
      },
      movePrevious() {
        calls.push('prev')
        enter(index - 1)
      }
    }

    return instance
  }
}

/** A fake app: records where a tour sent it, and what it revealed. */
function makeHost(startRoute = '/session-1') {
  let route = startRoute
  const panes: string[] = []

  return {
    host: {
      currentRoute: () => route,
      navigate: (to: string) => {
        route = to
      },
      revealPane: (pane: string) => void panes.push(pane)
    },
    panes,
    route: () => route
  }
}

function seedDom() {
  document.body.innerHTML = `
    <nav aria-label="Primary"><button id="send-btn" aria-label="Send message">Send</button></nav>
    <main><h1>Inbox</h1><div data-tour="composer">Composer</div><section><p>Positional only</p></section></main>
  `
  // happy-dom lays nothing out — every rect is 0×0, which the collector's
  // visibility filter (rightly) drops. Give elements a real-looking box.
  Element.prototype.getBoundingClientRect = () =>
    ({ bottom: 40, height: 32, left: 8, right: 128, top: 8, width: 120, x: 8, y: 8 }) as DOMRect
}

describe('collectTourTargets', () => {
  it('reports resolving selectors, durable handles first', () => {
    seedDom()
    const targets = collectTourTargets(document, 150)

    expect(targets.length).toBeGreaterThan(0)
    expect(targets.some(t => t.selector === '[data-tour="composer"]')).toBe(true)

    for (const target of targets) {
      expect(document.querySelector(target.selector)).not.toBeNull()
      expect(target.label).toBeTruthy()
    }

    // Stable targets sort ahead of positional ones, so a caller taking the
    // first N gets selectors that survive a re-render.
    const firstPositional = targets.findIndex(t => !t.stable)

    if (firstPositional !== -1) {
      expect(targets.slice(firstPositional).every(t => !t.stable)).toBe(true)
    }
  })

  it('flags identity selectors stable and nth-child paths not', () => {
    seedDom()
    const targets = collectTourTargets(document, 150)
    const byTour = targets.find(t => t.selector === '[data-tour="composer"]')
    const positional = targets.find(t => t.selector.includes('nth-child'))

    expect(byTour?.stable).toBe(true)
    expect(positional?.stable ?? false).toBe(false)
  })

  it('honors the cap', () => {
    seedDom()
    expect(collectTourTargets(document, 2)).toHaveLength(2)
  })
})

describe('runTourEngine', () => {
  it('targets answers with the page identity and its targets', () => {
    seedDom()
    const result = runTourEngine(makeFactory([]), {}, { kind: 'targets' }, collectTourTargets, document)

    expect(result.success).toBe(true)
    expect(result.targets?.length).toBeGreaterThan(0)
  })

  it('show validates the selector before touching driver.js', () => {
    seedDom()
    const calls: string[] = []
    const holder: TourHolder = {}
    const factory = makeFactory(calls)
    const bad = runTourEngine(factory, holder, { kind: 'show', selector: '#nope' }, collectTourTargets, document)

    expect(bad.success).toBe(false)
    expect(bad.error).toContain('#nope')
    // The recovery hint points at the re-scan, not a dead end.
    expect(bad.hint).toContain('targets')
    expect(calls).toEqual([])

    const good = runTourEngine(
      factory,
      holder,
      { kind: 'show', selector: '#send-btn', text: 'Click here to send', title: 'Send' },
      collectTourTargets,
      document
    )

    expect(good.success).toBe(true)
    expect(calls.some(c => c.startsWith('highlight:'))).toBe(true)
    expect(holder.driver).toBeDefined()
  })

  it('start → next pages through and cleans up after the last step', () => {
    seedDom()
    const calls: string[] = []
    const holder: TourHolder = {}
    const factory = makeFactory(calls)

    const started = runTourEngine(
      factory,
      holder,
      {
        kind: 'start',
        steps: [
          { selector: '#send-btn', title: 'Send' },
          { text: 'That is the whole flow.', title: 'Done' }
        ]
      },
      collectTourTargets,
      document
    )

    expect(started).toMatchObject({ activeStep: 0, steps: 2, success: true })
    expect(runTourEngine(factory, holder, { kind: 'next' }, collectTourTargets, document)).toMatchObject({
      activeStep: 1,
      success: true
    })

    // Advancing past the last step ends the tour.
    expect(runTourEngine(factory, holder, { kind: 'next' }, collectTourTargets, document)).toMatchObject({
      done: true,
      success: true
    })
    expect(holder.driver).toBeUndefined()
  })

  it('start honors startAt and rejects unknown selectors, naming each', () => {
    seedDom()
    const holder: TourHolder = {}
    const factory = makeFactory([])

    expect(
      runTourEngine(
        factory,
        holder,
        { kind: 'start', startAt: 1, steps: [{ selector: '#send-btn' }, { title: 'Second' }] },
        collectTourTargets,
        document
      )
    ).toMatchObject({ activeStep: 1, success: true })

    const result = runTourEngine(
      factory,
      {},
      { kind: 'start', steps: [{ selector: '#ghost', title: 'x' }, { selector: '#send-btn' }] },
      collectTourTargets,
      document
    )

    expect(result.success).toBe(false)
    expect(result.error).toContain('#ghost')
    expect(result.error).not.toContain('#send-btn')
  })

  it('treats a malformed selector as unmatched instead of throwing', () => {
    seedDom()
    const result = runTourEngine(makeFactory([]), {}, { kind: 'show', selector: '<<<' }, collectTourTargets, document)

    expect(result.success).toBe(false)
    expect(result.error).toContain('<<<')
  })

  it('next without a running tour errors, stop is idempotent', () => {
    const holder: TourHolder = {}
    const factory = makeFactory([])

    expect(runTourEngine(factory, holder, { kind: 'next' }, collectTourTargets, document).success).toBe(false)
    expect(runTourEngine(factory, holder, { kind: 'stop' }, collectTourTargets, document).success).toBe(true)
    expect(runTourEngine(factory, holder, { kind: 'stop' }, collectTourTargets, document).success).toBe(true)
  })

  it('is self-contained source (injectable into a guest page)', () => {
    // The preview surface stringifies these functions into a webview. Any
    // captured import/closure reference would throw there — the source must
    // reference nothing but its own parameters and page globals.
    for (const source of [runTourEngine.toString(), collectTourTargets.toString()]) {
      expect(source).not.toContain('__vite')
      expect(source).not.toContain('import(')
      expect(source).not.toContain('require(')
    }
  })
})

describe('tours that move the app', () => {
  it('navigates and reveals panes as their steps are entered', () => {
    seedDom()
    const app = makeHost('/session-1')
    const holder: TourHolder = {}

    runTourEngine(
      makeFactory([]),
      holder,
      {
        kind: 'start',
        steps: [
          { pane: 'sessions', title: 'Sidebar' },
          { navigate: '/artifacts', title: 'Artifacts' }
        ]
      },
      collectTourTargets,
      document,
      undefined,
      app.host
    )

    // The engine moves the app itself, before driver.js sees the step — and
    // exactly once per step, not once here and once from a driver hook.
    expect(app.panes).toEqual(['sessions'])
    expect(app.route()).toBe('/session-1')

    runTourEngine(makeFactory([]), holder, { kind: 'next' }, collectTourTargets, document, undefined, app.host)
    expect(app.route()).toBe('/artifacts')
  })

  it('moves the app however the step is reached', () => {
    // The class of bug this pins: the move used to live on the paths (the API,
    // then the buttons), so whichever path wasn't wired advanced the text while
    // the app stayed put — and the step waited forever for an element on a page
    // it never opened. It now hangs off the step's own onHighlightStarted, the
    // one hook driver fires for every path, so a custom or agent-authored tour
    // behaves the same whether the user clicks Next, presses →, or code calls it.
    seedDom()

    for (const advance of ['api', 'button', 'driver'] as const) {
      const app = makeHost('/session-1')
      const holder: TourHolder = {}
      const factory = makeFactory([])

      runTourEngine(
        factory,
        holder,
        { kind: 'start', steps: [{ title: 'Intro' }, { navigate: '/artifacts', title: 'Away' }] },
        collectTourTargets,
        document,
        undefined,
        app.host
      )

      expect(app.route(), advance).toBe('/session-1')

      if (advance === 'api') {
        runTourEngine(factory, holder, { kind: 'next' }, collectTourTargets, document, undefined, app.host)
      } else if (advance === 'button') {
        // The popover's own Next button — driver's handler, not ours.
        ;(holder.driver as unknown as { clickNext: () => void }).clickNext()
      } else {
        // Arrow keys and any other internal advance land here.
        holder.driver?.moveNext()
      }

      expect(app.route(), advance).toBe('/artifacts')
    }
  })

  it('returns to where it started, however the tour ends', () => {
    seedDom()

    for (const end of ['lastStep', 'dismissed'] as const) {
      const app = makeHost('/session-1')
      const holder: TourHolder = {}

      runTourEngine(
        makeFactory([]),
        holder,
        { kind: 'start', steps: [{ navigate: '/artifacts', title: 'Away' }] },
        collectTourTargets,
        document,
        undefined,
        app.host
      )

      expect(app.route()).toBe('/artifacts')

      // 'next' past the last step vs. an Esc/✕ dismissal both destroy the tour.
      if (end === 'lastStep') {
        runTourEngine(makeFactory([]), holder, { kind: 'next' }, collectTourTargets, document, undefined, app.host)
      } else {
        holder.driver?.destroy()
      }

      expect(app.route(), end).toBe('/session-1')
      // Nothing lingers: the slot is empty and the tour owns no other state.
      expect(holder.driver, end).toBeUndefined()
    }
  })

  it('leaves the route alone when no step navigates', () => {
    seedDom()
    const app = makeHost('/session-1')
    const holder: TourHolder = {}

    runTourEngine(
      makeFactory([]),
      holder,
      { kind: 'start', steps: [{ selector: '#send-btn', title: 'Here' }] },
      collectTourTargets,
      document,
      undefined,
      app.host
    )
    holder.driver?.destroy()

    expect(app.route()).toBe('/session-1')
  })

  it('waits for targets that mount after a step moves the app', () => {
    seedDom()
    const app = makeHost()

    // #not-mounted-yet does not exist at start time — it appears after the
    // navigation. Neither the navigating step NOR the steps after it may fail
    // selector pre-validation, and each must carry a wait.
    const result = runTourEngine(
      makeFactory([]),
      {},
      {
        kind: 'start',
        steps: [
          { navigate: '/artifacts', selector: '#not-mounted-yet', title: 'Later' },
          { selector: '#also-not-here', title: 'Later still' }
        ]
      },
      collectTourTargets,
      document,
      undefined,
      app.host
    )

    expect(result.success).toBe(true)
  })

  it('still rejects a bad selector when nothing moves the app', () => {
    seedDom()

    const result = runTourEngine(
      makeFactory([]),
      {},
      { kind: 'start', steps: [{ selector: '#ghost', title: 'x' }] },
      collectTourTargets,
      document,
      undefined,
      makeHost().host
    )

    expect(result.success).toBe(false)
    expect(result.error).toContain('#ghost')
  })

  it('survives the surface re-rendering underneath it', async () => {
    // The bug this pins: sitting idle on a page whose data polls/refetches used
    // to kill the tour. driver.js holds the highlighted node by REFERENCE, so
    // when React swapped it for an equivalent one, driver kept measuring a
    // detached node (rect all zeros) and the spotlight vanished — looking like
    // the tour closed itself while the user did nothing. Any list that refetches
    // hits this, so it matters for curated and agent-authored tours alike.
    seedDom()
    const calls: string[] = []
    const holder: TourHolder = {}

    runTourEngine(
      makeFactory(calls),
      holder,
      { kind: 'start', steps: [{ selector: '#send-btn', title: 'Send' }] },
      collectTourTargets,
      document,
      undefined,
      makeHost().host
    )

    const drivesBefore = calls.filter(c => c.startsWith('drive:')).length

    // Simulate a refetch: the old node is replaced by an equivalent new one.
    const old = document.querySelector('#send-btn')!

    old.classList.add('driver-active-element')

    const fresh = old.cloneNode(true) as HTMLElement

    old.replaceWith(fresh)
    fresh.classList.remove('driver-active-element')

    // MutationObserver callbacks are microtask-scheduled, then rAF-coalesced.
    await new Promise(resolve => setTimeout(resolve, 50))

    expect(calls.filter(c => c.startsWith('drive:')).length).toBeGreaterThan(drivesBefore)
    expect(holder.driver?.isActive()).toBe(true)
  })

  it('stops watching once the tour ends', async () => {
    seedDom()
    const calls: string[] = []
    const holder: TourHolder = {}
    const factory = makeFactory(calls)

    runTourEngine(
      factory,
      holder,
      { kind: 'start', steps: [{ selector: '#send-btn', title: 'Send' }] },
      collectTourTargets,
      document,
      undefined,
      makeHost().host
    )
    runTourEngine(factory, holder, { kind: 'stop' }, collectTourTargets, document, undefined, makeHost().host)

    const after = calls.length
    const node = document.querySelector('#send-btn')!

    node.replaceWith(node.cloneNode(true))
    await new Promise(resolve => setTimeout(resolve, 50))

    // No re-drive attempts once it's over.
    expect(calls.length).toBe(after)
  })
})
