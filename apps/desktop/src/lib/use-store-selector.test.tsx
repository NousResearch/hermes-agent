import { render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { useStoreSelector } from './use-session-slice'

describe('useStoreSelector snapshots', () => {
  it('caches a structurally unchanged projection returned by an inline selector', () => {
    const source = atom({ names: ['Hermes'] })
    const renders = vi.fn()

    function Probe() {
      const names = useStoreSelector(source, value => [...value.names])
      renders()

      return <span data-testid="names">{names.join(',')}</span>
    }

    render(<Probe />)
    const baseline = renders.mock.calls.length

    // A new store object and a fresh array projection still describe the same
    // UI. The snapshot must stay referentially stable for React.
    act(() => source.set({ names: ['Hermes'] }))

    expect(renders.mock.calls.length).toBe(baseline)
    expect(screen.getByTestId('names').textContent).toBe('Hermes')

    act(() => source.set({ names: ['Hermes', 'Operations'] }))

    expect(renders.mock.calls.length).toBeGreaterThan(baseline)
    expect(screen.getByTestId('names').textContent).toBe('Hermes,Operations')
  })
})
