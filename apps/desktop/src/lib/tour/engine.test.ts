import { describe, expect, it } from 'vitest'

import { collectTourTargets } from './collect-targets'
import { runTourEngine, type TourDriver, type TourHolder } from './engine'

/** A recording fake of the driver.js surface the engine touches. */
function makeFactory(calls: string[]) {
  return (config?: object) => {
    const steps = (config as { steps?: unknown[] } | undefined)?.steps ?? []
    let active = false
    let index = 0

    const instance: TourDriver = {
      destroy() {
        calls.push('destroy')
        active = false
      },
      drive(stepIndex?: number) {
        calls.push(`drive:${stepIndex ?? 0}`)
        active = true
        index = stepIndex ?? 0
      },
      getActiveIndex: () => index,
      highlight(step: object) {
        calls.push(`highlight:${JSON.stringify(step)}`)
        active = true
      },
      isActive: () => active,
      isLastStep: () => index >= steps.length - 1,
      moveNext() {
        calls.push('next')
        index += 1
      },
      movePrevious() {
        calls.push('prev')
        index -= 1
      }
    }

    return instance
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
