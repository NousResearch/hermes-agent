/**
 * A route tile opened BEFORE its plugin route registers must pick up the
 * contribution's title once registration lands. `paneMirror` recomputes titles
 * only when one of its atoms changes; without a routes-area signal the tab
 * kept the humanized-path fallback forever while the pane content healed.
 */
import { act, cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, expect, it } from 'vitest'

import { paneChrome } from '@/components/pane-shell/tree/renderer/track-model'
import { registry } from '@/contrib/registry'
import { resolveTranslations, setRuntimeI18nLocale } from '@/i18n'
import { I18nProvider } from '@/i18n/context'
import { $routeTiles, closeRouteTile, openRouteTile } from '@/store/route-tiles'
import { $shellChoice } from '@/store/shell-variant'

import { ARTIFACTS_ROUTE } from '../routes'

import { watchRouteTiles } from './route-tile'

const pane = (path: string) => registry.getArea('panes').find(c => c.id === `route-tile:${path}`)
const paneTitle = (path: string) => pane(path)?.title
const chrome = (path: string) => paneChrome(pane(path))

beforeAll(() => {
  watchRouteTiles()
})

afterEach(() => {
  cleanup()

  for (const tile of $routeTiles.get()) {
    closeRouteTile(tile.path)
  }

  $shellChoice.set(null)
  setRuntimeI18nLocale('en')
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

it('names a built-in page as classic always has, and as the redesign nav does in the live language', () => {
  $shellChoice.set(false)
  openRouteTile(ARTIFACTS_ROUTE)

  const tab = (node: ReactNode) =>
    render(
      <I18nProvider configClient={null} initialLocale="de">
        {node}
      </I18nProvider>
    ).container

  const label = tab(chrome(ARTIFACTS_ROUTE).tabTitle?.())
  expect(paneTitle(ARTIFACTS_ROUTE)).toBe('Artifacts')
  expect(chrome(ARTIFACTS_ROUTE).tabTitleText?.()).toBe('Artifacts')
  expect(label.textContent).toBe('Artifacts')

  act(() => $shellChoice.set(true))
  const library = resolveTranslations('de', 'redesign').sidebar.nav.artifacts
  expect(library).not.toBe('Artifacts')
  expect(label.textContent).toBe(library)
  expect(chrome(ARTIFACTS_ROUTE).tabTitleText?.()).toBe(library)

  act(() => setRuntimeI18nLocale('fr'))
  expect(chrome(ARTIFACTS_ROUTE).tabTitleText?.()).toBe(resolveTranslations('fr', 'redesign').sidebar.nav.artifacts)
})
