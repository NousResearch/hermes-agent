import { act, renderHook } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
})

it('returns the live raw combos for an action and follows a rebind', async () => {
  const { useKeybindCombos } = await import('@/lib/keybinds/use-keybind-hint')
  const { setBinding } = await import('@/store/keybinds')
  const { result } = renderHook(() => useKeybindCombos('session.new'))

  expect(result.current.length).toBeGreaterThan(0)

  act(() => setBinding('session.new', ['ctrl+alt+k']))
  expect(result.current).toEqual(['ctrl+alt+k'])
})
