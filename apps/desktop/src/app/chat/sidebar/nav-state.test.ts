import { describe, expect, test } from 'vitest'

import type { AppView } from '../../routes'
import type { SidebarNavItem } from '../../types'

import { sidebarNavItemIsActive, sidebarVisibleView } from './nav-state'

const item = (id: string, route: string): SidebarNavItem => ({
  id,
  icon: () => null,
  label: id,
  route
})

describe('sidebar navigation selection', () => {
  test('reports chat when focus is on a session tile', () => {
    expect(sidebarVisibleView('extension', true)).toBe('chat')
  })

  test('preserves the current view when focus is on the workspace', () => {
    expect(sidebarVisibleView('extension', false)).toBe('extension')
  })

  test.each<[AppView, string, boolean]>([
    ['extension', '/kanban', true],
    ['extension', '/reports', false],
    ['chat', '/kanban', false]
  ])('selects a contributed route only when its page is visible', (currentView, pathname, expected) => {
    expect(
      sidebarNavItemIsActive({
        contributed: true,
        currentView,
        item: item('kanban', '/kanban'),
        pathname
      })
    ).toBe(expected)
  })

  test('applies the visible view rule to every contributed route', () => {
    expect(
      sidebarNavItemIsActive({
        contributed: true,
        currentView: 'chat',
        item: item('reports', '/reports'),
        pathname: '/reports'
      })
    ).toBe(false)
  })

  test.each<[string, string, AppView]>([
    ['skills', '/skills', 'skills'],
    ['messaging', '/messaging', 'messaging'],
    ['artifacts', '/artifacts', 'artifacts'],
    ['cron', '/cron', 'cron']
  ])('preserves built in %s selection', (id, route, currentView) => {
    expect(
      sidebarNavItemIsActive({
        contributed: false,
        currentView,
        item: item(id, route),
        pathname: route
      })
    ).toBe(true)

    expect(
      sidebarNavItemIsActive({
        contributed: false,
        currentView: 'chat',
        item: item(id, route),
        pathname: '/session'
      })
    ).toBe(false)
  })

  test.each<[string, string]>([
    ['skills', '/skills'],
    ['messaging', '/messaging'],
    ['artifacts', '/artifacts'],
    ['cron', '/cron']
  ])('clears stale built in %s route selection while a session tile is visible', (id, route) => {
    expect(
      sidebarNavItemIsActive({
        contributed: false,
        currentView: 'chat',
        item: item(id, route),
        pathname: route
      })
    ).toBe(false)
  })
})
