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
  test.each<{
    contributed: boolean
    currentView: AppView
    expected: boolean
    focusedSessionIsTile: boolean
    id: string
    pathname: string
    route: string
    visibleView: AppView
  }>([
    {
      contributed: true,
      currentView: 'extension',
      expected: true,
      focusedSessionIsTile: false,
      id: 'kanban',
      pathname: '/kanban',
      route: '/kanban',
      visibleView: 'extension'
    },
    {
      contributed: true,
      currentView: 'extension',
      expected: false,
      focusedSessionIsTile: false,
      id: 'kanban',
      pathname: '/reports',
      route: '/kanban',
      visibleView: 'extension'
    },
    ...['kanban', 'reports'].map(id => ({
      contributed: true,
      currentView: 'extension' as const,
      expected: false,
      focusedSessionIsTile: true,
      id,
      pathname: `/${id}`,
      route: `/${id}`,
      visibleView: 'chat' as const
    })),
    ...(['skills', 'messaging', 'artifacts', 'cron'] as const).flatMap(id => [
      {
        contributed: false,
        currentView: id,
        expected: true,
        focusedSessionIsTile: false,
        id,
        pathname: `/${id}`,
        route: `/${id}`,
        visibleView: id
      },
      {
        contributed: false,
        currentView: id,
        expected: false,
        focusedSessionIsTile: true,
        id,
        pathname: `/${id}`,
        route: `/${id}`,
        visibleView: 'chat' as const
      }
    ]),
    {
      contributed: false,
      currentView: 'skills',
      expected: false,
      focusedSessionIsTile: false,
      id: 'messaging',
      pathname: '/messaging',
      route: '/messaging',
      visibleView: 'skills'
    }
  ])('keeps route activity coherent for $id with tile focus $focusedSessionIsTile', input => {
    const visibleView = sidebarVisibleView(input.currentView, input.focusedSessionIsTile)

    expect(visibleView).toBe(input.visibleView)
    expect(
      sidebarNavItemIsActive({
        contributed: input.contributed,
        currentView: visibleView,
        item: item(input.id, input.route),
        pathname: input.pathname
      })
    ).toBe(input.expected)
  })
})
