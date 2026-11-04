import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $interfaceMode } from '@/store/interface-mode'
import { SIDEBAR_NAV_PREFS_AREA } from '@/store/sidebar-nav'

import { SIDEBAR_NAV_AREA } from '../../routes'

import { SimplifiedGlobalMenu } from './simplified-global-menu'

afterEach(cleanup)

it('opens the existing Plugins manager through normal sidebar navigation', () => {
  const onNavigate = vi.fn()
  render(<SimplifiedGlobalMenu onNavigate={onNavigate} />)
  fireEvent.keyDown(screen.getByRole('button', { name: 'Hermes' }), { key: 'Enter' })
  fireEvent.click(screen.getByRole('menuitem', { name: 'Plugins' }))
  expect(onNavigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/capabilities?tab=plugins' }))
})

it('keeps plugin pages reachable and respects visibility, preferences, and unloading', () => {
  const previousMode = $interfaceMode.get()
  $interfaceMode.set('simple')

  const dispose = registry.registerMany([
    { area: SIDEBAR_NAV_AREA, id: 'test-board', data: { label: 'Test board', path: '/board' } },
    { area: SIDEBAR_NAV_AREA, id: 'test-advanced', data: { label: 'Advanced page', path: '/advanced', tier: 'advanced' } },
    { area: SIDEBAR_NAV_AREA, id: 'test-hidden', data: { label: 'Hidden page', path: '/hidden' } },
    { area: SIDEBAR_NAV_AREA, id: 'test-invalid', data: { label: 'Invalid page', path: 'invalid' } },
    { area: SIDEBAR_NAV_PREFS_AREA, id: 'test-prefs', data: { hide: ['test-hidden'] } }
  ])

  try {
    const onNavigate = vi.fn()
    render(<SimplifiedGlobalMenu onNavigate={onNavigate} />)
    fireEvent.keyDown(screen.getByRole('button', { name: 'Hermes' }), { key: 'Enter' })
    expect(screen.queryByRole('menuitem', { name: 'Advanced page' })).toBeNull()
    expect(screen.queryByRole('menuitem', { name: 'Hidden page' })).toBeNull()
    expect(screen.queryByRole('menuitem', { name: 'Invalid page' })).toBeNull()
    fireEvent.click(screen.getByRole('menuitem', { name: 'Test board' }))
    expect(onNavigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/board' }))
    fireEvent.keyDown(screen.getByRole('button', { name: 'Hermes' }), { key: 'Enter' })
    act(dispose)
    expect(screen.queryByRole('menuitem', { name: 'Test board' })).toBeNull()
  } finally {
    cleanup()
    dispose()
    $interfaceMode.set(previousMode)
  }
})
