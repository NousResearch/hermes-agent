/**
 * A route tile opened BEFORE its plugin route registers must pick up the
 * contribution's title once registration lands. `paneMirror` recomputes titles
 * only when one of its atoms changes; without a routes-area signal the tab
 * kept the humanized-path fallback forever while the pane content healed.
 */
import { render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { I18nProvider, setRuntimeI18nLocale } from '@/i18n'
import { $routeTiles, closeRouteTile, openRouteTile } from '@/store/route-tiles'

import { PROJECTS_ROUTE } from '../routes'

import { watchRouteTiles } from './route-tile'

const paneTitle = (path: string) => registry.getArea('panes').find(c => c.id === `route-tile:${path}`)?.title

beforeAll(() => {
  watchRouteTiles()
})

afterEach(() => {
  for (const tile of $routeTiles.get()) {
    closeRouteTile(tile.path)
  }
})

it('refreshes the tile title when its route registers after the tile opened', () => {
  openRouteTile('/late-atlas')
  expect(paneTitle('/late-atlas')).toBe('Late Atlas')

  const dispose = registry.register({
    area: 'routes',
    id: 'late-atlas:page',
    title: 'Atlas of Everything',
    data: { path: '/late-atlas' },
    render: () => null
  })

  expect(paneTitle('/late-atlas')).toBe('Atlas of Everything')
  dispose()
})

it('titles the Projects tile with the localized navigation label, live with the app language', () => {
  setRuntimeI18nLocale('de')

  try {
    openRouteTile(PROJECTS_ROUTE)
    expect(paneTitle(PROJECTS_ROUTE)).toBe('Projekte')

    const tabTitle = registry.getArea('panes').find(c => c.id === `route-tile:${PROJECTS_ROUTE}`)?.data as
      undefined | { tabTitle?: () => ReactNode }

    render(
      <I18nProvider configClient={null} initialLocale="fr">
        {tabTitle?.tabTitle?.()}
      </I18nProvider>
    )
    expect(screen.getByText('Projets')).toBeTruthy()
  } finally {
    setRuntimeI18nLocale('en')
  }
})
