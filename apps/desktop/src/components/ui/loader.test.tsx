import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setReducedEffects } from '@/store/reduced-effects'

import { Loader } from './loader'

describe('Loader', () => {
  beforeEach(() => {
    setReducedEffects(false)
    vi.stubGlobal(
      'matchMedia',
      vi.fn(() => ({ matches: false }))
    )
    vi.stubGlobal(
      'requestAnimationFrame',
      vi.fn(() => 1)
    )
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('schedules continuous frames in full-effects mode', () => {
    render(<Loader />)

    expect(requestAnimationFrame).toHaveBeenCalledTimes(1)
  })

  it('renders one static frame in reduced-effects mode', () => {
    setReducedEffects(true)

    render(<Loader />)

    expect(requestAnimationFrame).not.toHaveBeenCalled()
  })
})
