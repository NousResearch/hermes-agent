import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { JarvisCore } from './core'

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
  vi.restoreAllMocks()
})

describe('JarvisCore', () => {
  it('exposes semantic state for assistive technology', () => {
    render(<JarvisCore audioLevel={0.6} taskPhase="running" voice="listening" />)

    expect(screen.getByRole('status').getAttribute('aria-label')).toBe('Jarvis słucha i wykonuje zadanie')
    expect(screen.getByTestId('jarvis-core').getAttribute('data-voice')).toBe('listening')
    expect(screen.getByTestId('jarvis-core').getAttribute('data-task')).toBe('running')
  })

  it('clamps audio and only lets it drive visual amplitude during audio states', () => {
    const { rerender } = render(<JarvisCore audioLevel={1.8} taskPhase="idle" voice="listening" />)
    const core = screen.getByTestId('jarvis-core')

    const activeEnergy = Number(core.style.getPropertyValue('--jarvis-energy-scale'))
    const activeHalo = Number(core.style.getPropertyValue('--jarvis-halo-opacity'))

    expect(core.style.getPropertyValue('--jarvis-audio-level')).toBe('1')
    expect(activeEnergy).toBeGreaterThan(1)
    expect(activeHalo).toBeGreaterThan(0)

    rerender(<JarvisCore audioLevel={0.8} taskPhase="running" voice="idle" />)

    expect(core.style.getPropertyValue('--jarvis-audio-level')).toBe('0')
    expect(Number(core.style.getPropertyValue('--jarvis-energy-scale'))).toBeLessThan(activeEnergy)
    expect(Number(core.style.getPropertyValue('--jarvis-task-signal'))).toBeGreaterThan(0)
    expect(Number(core.style.getPropertyValue('--jarvis-current-opacity'))).toBeGreaterThan(0)
  })

  it('marks compact layout and reduced motion without changing the state contract', () => {
    mockReducedMotion(true)

    render(<JarvisCore audioLevel={0.4} compact taskPhase="approval" voice="speaking" />)
    const core = screen.getByTestId('jarvis-core')

    expect(core.getAttribute('data-compact')).toBe('true')
    expect(core.getAttribute('data-motion')).toBe('reduced')
    expect(core.style.getPropertyValue('--jarvis-core-size')).toBe('148px')
    expect(core.getAttribute('data-voice')).toBe('speaking')
    expect(core.getAttribute('data-task')).toBe('approval')
  })

  it('keeps SVG paint definitions scoped to each rendered instance', () => {
    const { container } = render(
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
})
