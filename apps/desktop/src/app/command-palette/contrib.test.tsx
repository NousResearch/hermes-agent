import '@/app/contrib/controller'

import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { setInterfaceMode } from '@/store/interface-mode'

import { usePaletteContributions } from './contrib'

describe('contributed ⌘K rows by interface mode', () => {
  afterEach(() => setInterfaceMode('advanced'))

  it('rests developer rows in Simple, keeps the way back, and restores them in Advanced on the open palette', () => {
    const { result } = renderHook(() => usePaletteContributions().map(row => row.label))

    expect(result.current).toEqual(expect.arrayContaining(['Toggle yolo', 'Toggle logs', 'Simple mode']))

    act(() => setInterfaceMode('simple'))

    expect(result.current).not.toContain('Toggle yolo')
    expect(result.current).not.toContain('Toggle logs')
    expect(result.current).toContain('Simple mode')

    act(() => setInterfaceMode('advanced'))

    expect(result.current).toContain('Toggle yolo')
  })
})
