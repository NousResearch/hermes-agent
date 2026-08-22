// @vitest-environment jsdom
import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useColorblindMode } from './use-colorblind-mode'

const mocks = vi.hoisted(() => ({
  loadedConfig: {} as Record<string, unknown>
}))

vi.mock('./use-config-record', () => ({
  useHermesConfigRecord: () => ({ data: mocks.loadedConfig })
}))

describe('useColorblindMode', () => {
  beforeEach(() => {
    delete document.documentElement.dataset.colorblind
  })

  afterEach(() => {
    cleanup()
    delete document.documentElement.dataset.colorblind
  })

  it('leaves the attribute off when colorblind_mode is unset or false', () => {
    mocks.loadedConfig = {}
    renderHook(() => useColorblindMode())
    expect(document.documentElement.dataset.colorblind).toBeUndefined()

    mocks.loadedConfig = { desktop: { colorblind_mode: false } }
    renderHook(() => useColorblindMode())
    expect(document.documentElement.dataset.colorblind).toBeUndefined()
  })

  it('sets html[data-colorblind="true"] when colorblind_mode is on', () => {
    mocks.loadedConfig = { desktop: { colorblind_mode: true } }
    renderHook(() => useColorblindMode())
    expect(document.documentElement.dataset.colorblind).toBe('true')
  })

  it('removes the attribute when the toggle turns back off', () => {
    mocks.loadedConfig = { desktop: { colorblind_mode: true } }
    const { rerender } = renderHook(() => useColorblindMode())
    expect(document.documentElement.dataset.colorblind).toBe('true')

    mocks.loadedConfig = { desktop: { colorblind_mode: false } }
    rerender()
    expect(document.documentElement.dataset.colorblind).toBeUndefined()
  })
})
