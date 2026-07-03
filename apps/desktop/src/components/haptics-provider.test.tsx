import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const useWebHaptics = vi.fn((options: unknown) => {
  void options

  return { trigger: vi.fn() }
})

vi.mock('web-haptics/react', () => ({
  useWebHaptics: (options: unknown) => useWebHaptics(options)
}))

describe('HapticsProvider', () => {
  beforeEach(() => {
    window.localStorage.clear()
    useWebHaptics.mockClear()
  })

  afterEach(() => {
    cleanup()
    window.localStorage.clear()
  })

  it('keeps web-haptics debug logging disabled by default', async () => {
    const { HapticsProvider } = await import('./haptics-provider')

    render(
      <HapticsProvider>
        <div>child</div>
      </HapticsProvider>
    )

    expect(useWebHaptics).toHaveBeenCalledWith({ debug: false, showSwitch: false })
  })

  it('enables web-haptics debug logging only behind the local debug flag', async () => {
    window.localStorage.setItem('hermes.desktop.hapticsDebug', 'true')
    const { HapticsProvider } = await import('./haptics-provider')

    render(
      <HapticsProvider>
        <div>child</div>
      </HapticsProvider>
    )

    expect(useWebHaptics).toHaveBeenCalledWith({ debug: true, showSwitch: false })
  })
})
