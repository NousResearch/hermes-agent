import { act, cleanup, render, screen } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider, type Locale } from '@/i18n'
import { publishMicLevel, resetMicLevel } from '@/store/voice-level'

import { JarvisCore } from './core'

function renderCore(ui: ReactElement, locale: Locale = 'pl') {
  return render(
    <I18nProvider configClient={null} initialLocale={locale}>
      {ui}
    </I18nProvider>
  )
}

function mockReducedMotion(matches: boolean) {
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: query === '(prefers-reduced-motion: reduce)' ? matches : false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

afterEach(() => {
  cleanup()
  resetMicLevel()
  vi.restoreAllMocks()
})

describe('Agent CzesiekCore', () => {
  it('exposes semantic state for assistive technology', () => {
    renderCore(<JarvisCore audioLevel={0.6} taskPhase="running" voice="listening" />)

    expect(screen.getByRole('status').getAttribute('aria-label')).toBe('Agent Czesiek słucha i wykonuje zadanie')
    expect(screen.getByTestId('jarvis-core').getAttribute('data-voice')).toBe('listening')
    expect(screen.getByTestId('jarvis-core').getAttribute('data-task')).toBe('running')
  })

  it('describes itself in the active locale rather than a hardcoded language', () => {
    renderCore(<JarvisCore taskPhase="running" voice="listening" />, 'en')

    expect(screen.getByRole('status').getAttribute('aria-label')).toBe(
      'Agent Czesiek is listening and running the task'
    )

    cleanup()
    renderCore(<JarvisCore taskPhase="idle" voice="idle" />, 'en')

    // An idle task contributes no clause — no dangling "and".
    expect(screen.getByRole('status').getAttribute('aria-label')).toBe('Agent Czesiek is waiting')
  })

  it('clamps audio and only lets it drive visual amplitude during audio states', () => {
    const { rerender } = renderCore(<JarvisCore audioLevel={1.8} taskPhase="idle" voice="listening" />)
    const core = screen.getByTestId('jarvis-core')

    const activeEnergy = Number(core.style.getPropertyValue('--jarvis-energy-scale'))
    const activeHalo = Number(core.style.getPropertyValue('--jarvis-halo-opacity'))

    expect(core.style.getPropertyValue('--jarvis-audio-level')).toBe('1')
    expect(activeEnergy).toBeGreaterThan(1)
    expect(activeHalo).toBeGreaterThan(0)

    rerender(
      <I18nProvider configClient={null} initialLocale="pl">
        <JarvisCore audioLevel={0.8} taskPhase="running" voice="idle" />
      </I18nProvider>
    )

    expect(core.style.getPropertyValue('--jarvis-audio-level')).toBe('0')
    expect(Number(core.style.getPropertyValue('--jarvis-energy-scale'))).toBeLessThan(activeEnergy)
    expect(Number(core.style.getPropertyValue('--jarvis-task-signal'))).toBeGreaterThan(0)
    expect(Number(core.style.getPropertyValue('--jarvis-current-opacity'))).toBeGreaterThan(0)
  })

  it('marks compact layout and reduced motion without changing the state contract', () => {
    mockReducedMotion(true)

    renderCore(<JarvisCore audioLevel={0.4} compact taskPhase="approval" voice="speaking" />)
    const core = screen.getByTestId('jarvis-core')

    expect(core.getAttribute('data-compact')).toBe('true')
    expect(core.getAttribute('data-motion')).toBe('reduced')
    expect(core.style.getPropertyValue('--jarvis-core-size')).toBe('184px')
    expect(core.getAttribute('data-voice')).toBe('speaking')
    expect(core.getAttribute('data-task')).toBe('approval')
  })

  it('keeps SVG paint definitions scoped to each rendered instance', () => {
    const { container } = renderCore(
      <>
        <JarvisCore audioLevel={0.4} taskPhase="running" voice="listening" />
        <JarvisCore audioLevel={0.2} taskPhase="verified" voice="speaking" />
      </>
    )

    const cores = screen.getAllByTestId('jarvis-core')
    expect(cores).toHaveLength(2)

    const ids = Array.from(container.querySelectorAll('radialGradient, linearGradient, filter'), element =>
      element.getAttribute('id')
    )

    expect(ids).toHaveLength(6)
    expect(new Set(ids).size).toBe(ids.length)
    expect(ids.every(id => id?.startsWith('jarvis-core-'))).toBe(true)

    for (const core of cores) {
      const localIds = Array.from(core.querySelectorAll('radialGradient, linearGradient, filter'), element =>
        element.getAttribute('id')
      )

      const glass = core.querySelector<SVGCircleElement>('.jarvis-core__glass')
      const liquid = core.querySelector<SVGPathElement>('.jarvis-core__liquid')

      expect(localIds).toHaveLength(3)
      expect(glass?.getAttribute('fill')).toBe(`url(#${localIds[0]})`)
      expect(liquid?.getAttribute('fill')).toBe(`url(#${localIds[1]})`)
      expect(liquid?.getAttribute('filter')).toBe(`url(#${localIds[2]})`)
    }
  })

  it('tracks the real microphone store when live, and rests at zero when it is not listening', () => {
    const { rerender } = renderCore(<JarvisCore live taskPhase="idle" voice="listening" />)
    const core = screen.getByTestId('jarvis-core')

    act(() => publishMicLevel(0.75))

    expect(Number(core.style.getPropertyValue('--jarvis-audio-level'))).toBeCloseTo(0.75, 1)
    expect(Number(core.style.getPropertyValue('--jarvis-pulse-opacity'))).toBeGreaterThan(0)

    rerender(
      <I18nProvider configClient={null} initialLocale="pl">
        <JarvisCore live taskPhase="running" voice="idle" />
      </I18nProvider>
    )

    // A running task with the mic closed is silent, however loud the last
    // reading was.
    expect(core.style.getPropertyValue('--jarvis-audio-level')).toBe('0')
    expect(core.style.getPropertyValue('--jarvis-pulse-opacity')).toBe('0')
  })

  it('leaves the prop-driven core alone: an unbound core ignores the live store', () => {
    renderCore(<JarvisCore audioLevel={0.5} taskPhase="idle" voice="listening" />)
    const core = screen.getByTestId('jarvis-core')

    act(() => publishMicLevel(1))

    expect(Number(core.style.getPropertyValue('--jarvis-audio-level'))).toBeCloseTo(0.5, 2)
  })
})
