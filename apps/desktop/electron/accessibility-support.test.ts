import { describe, expect, it, vi } from 'vitest'

import { enableMacRendererAccessibility } from './accessibility-support'

describe('enableMacRendererAccessibility', () => {
  it('enables the renderer accessibility tree on macOS only', () => {
    const setAccessibilitySupportEnabled = vi.fn()

    enableMacRendererAccessibility('darwin', { setAccessibilitySupportEnabled })
    enableMacRendererAccessibility('linux', { setAccessibilitySupportEnabled })

    expect(setAccessibilitySupportEnabled).toHaveBeenCalledTimes(1)
    expect(setAccessibilitySupportEnabled).toHaveBeenCalledWith(true)
  })
})
