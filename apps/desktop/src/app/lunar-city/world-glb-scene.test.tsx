import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { WorldGlbScene } from './world-glb-scene'

describe('WorldGlbScene', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('does not start a render loop while disabled', () => {
    const frame = vi.spyOn(window, 'requestAnimationFrame')

    render(<WorldGlbScene enabled={false} />)

    expect(screen.getByTestId('lunar-city-glb-host')).toBeTruthy()
    expect(frame).not.toHaveBeenCalled()
  })
})
