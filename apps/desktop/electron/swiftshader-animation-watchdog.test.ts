import { describe, expect, it } from 'vitest'

import {
  decideStuckAnimationWarning,
  formatStuckAnimationWarning,
  parseStuckAnimationObserverLine
} from './swiftshader-animation-watchdog'

// Exact signature from the #124255 incident report (chromium desktop-chromium.log).
const STUCK_BUTTON_LINE =
  '[345851:0926/073122.013735:ERROR:ui/compositor/compositor_animation_observer.cc:65] ' +
  'CompositorAnimationObserver is active for too long (182.893s) ' +
  'location=Button@ui/views/controls/button/button.cc:667'

describe('parseStuckAnimationObserverLine', () => {
  it('parses the stuck Button observer line', () => {
    expect(parseStuckAnimationObserverLine(STUCK_BUTTON_LINE)).toEqual({
      activeSeconds: 182.893,
      location: 'Button@ui/views/controls/button/button.cc:667'
    })
  })

  it('returns null for unrelated chromium log lines', () => {
    expect(parseStuckAnimationObserverLine('')).toBeNull()
    expect(
      parseStuckAnimationObserverLine('[123:0926/073122.013735:ERROR:ui/compositor/other.cc:65] something else')
    ).toBeNull()
    expect(parseStuckAnimationObserverLine('CompositorAnimationObserver is active for too long')).toBeNull()
  })
})

describe('decideStuckAnimationWarning', () => {
  it('warns when the fallback is active and a stuck observer exceeds the threshold', () => {
    const decision = decideStuckAnimationWarning({
      tail: `some earlier log line\n${STUCK_BUTTON_LINE}\n`,
      fallbackActive: true
    })

    expect(decision.warn).toBe(true)
    expect(decision.activeSeconds).toBe(182.893)
    expect(decision.location).toContain('Button@')
  })

  it('stays silent without the fallback (GPU compositing absorbs the stuck observer)', () => {
    const decision = decideStuckAnimationWarning({
      tail: `${STUCK_BUTTON_LINE}\n`,
      fallbackActive: false
    })

    expect(decision.warn).toBe(false)
  })

  it('stays silent below the threshold', () => {
    const decision = decideStuckAnimationWarning({
      tail: 'CompositorAnimationObserver is active for too long (12.5s) location=Button@x\n',
      fallbackActive: true,
      thresholdSeconds: 60
    })

    expect(decision.warn).toBe(false)
  })

  it('warns only once per process', () => {
    const decision = decideStuckAnimationWarning({
      tail: `${STUCK_BUTTON_LINE}\n`,
      fallbackActive: true,
      alreadyWarned: true
    })

    expect(decision.warn).toBe(false)
  })

  it('stays silent when the tail has no stuck-observer line', () => {
    expect(
      decideStuckAnimationWarning({ tail: 'ordinary chromium error noise\n', fallbackActive: true }).warn
    ).toBe(false)
  })
})

describe('formatStuckAnimationWarning', () => {
  it('names the observer, the duration, and the opt-out', () => {
    const message = formatStuckAnimationWarning({ activeSeconds: 182.893, location: 'Button@x' })

    expect(message).toContain('182.893')
    expect(message).toContain('Button@x')
    expect(message).toContain('HERMES_DESKTOP_NVIDIA_SWIFTSHADER=0')
  })
})
