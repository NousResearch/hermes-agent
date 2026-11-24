// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { $narrowViewport } from '@/components/pane-shell/tree/store'
import { $interfaceMode, type InterfaceMode, setInterfaceMode } from '@/store/interface-mode'
import { setSidebarOpen } from '@/store/layout'
import { $shellChoice } from '@/store/shell-variant'
import { titlebarAppActionsClusterCounts } from '@/store/titlebar-app-actions'

import { $widthClass } from '../redesign/breakpoints'

import { useBandHiddenTools } from './band-tools'

let previousMode: InterfaceMode

beforeEach(() => {
  previousMode = $interfaceMode.get()
  $shellChoice.set(null)
  setInterfaceMode('advanced')
  $widthClass.set('wide')
  $narrowViewport.set(false)
  setSidebarOpen(true)
})

afterEach(() => {
  setInterfaceMode(previousMode)
  $widthClass.set('wide')
  $narrowViewport.set(false)
  $shellChoice.set(null)
})

// --titlebar-controls-width comes from this count; a slot for a tool the band hides offsets
// everything keyed on it.
it('reserves the left cluster only for the band tools the window shows', () => {
  const { result } = renderHook(() => titlebarAppActionsClusterCounts('right', 0, 0, 'advanced', useBandHiddenTools()))

  // Back and Forward; the open sidebar needs no band button in wide Advanced.
  expect(result.current.left).toBe(2)

  act(() => setSidebarOpen(false))
  expect(result.current.left).toBe(3)

  // Compact drops Back and Forward; the sidebar button is the menu.
  act(() => {
    $widthClass.set('compact')
    $narrowViewport.set(true)
  })
  expect(result.current.left).toBe(1)
})

// HADESK-118. The layout editor holds the Simple/Advanced picker. Hidden in Simple, choosing Simple
// there took away the control that switches back.
it('keeps the layout editor in the band in Simple mode, where the way back to Advanced lives', () => {
  const { result } = renderHook(() => useBandHiddenTools())
  expect(result.current.has('layout')).toBe(false)

  act(() => setInterfaceMode('simple'))
  expect(result.current.has('layout')).toBe(false)

  // A compact band has no room for it; Settings carries the same switch.
  act(() => $widthClass.set('compact'))
  expect(result.current.has('layout')).toBe(true)
})
