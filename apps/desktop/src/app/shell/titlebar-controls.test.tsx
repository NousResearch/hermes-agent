// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { I18nProvider } from '@/i18n'

import { ROUTES_AREA } from '../routes'

import { TitlebarControls } from './titlebar-controls'

function renderControls(pathname: string) {
  return render(
    <MemoryRouter initialEntries={[pathname]}>
      <I18nProvider configClient={null} initialLocale="en">
        <TitlebarControls onOpenSettings={() => {}} />
      </I18nProvider>
    </MemoryRouter>
  )
}

const windowControls = () => screen.queryByLabelText('Window controls')
const appControls = () => screen.queryByLabelText('App controls')
const pluginChrome = () => screen.queryByText('plugin-chrome')

describe('TitlebarControls fixed clusters', () => {
  let dispose: () => void

  beforeEach(() => {
    dispose = registry.registerMany([
      {
        area: ROUTES_AREA,
        data: { path: '/kanban' },
        id: 'test-kanban-route',
        render: () => null
      },
      {
        area: 'titleBar.center',
        id: 'test-plugin-chrome',
        render: () => <span>plugin-chrome</span>
      }
    ])
  })

  afterEach(() => {
    dispose()
    cleanup()
  })

  it('hides the app clusters on a contributed full-page route', () => {
    renderControls('/kanban')

    expect(windowControls()).toBeNull()
    expect(appControls()).toBeNull()
  })

  it('keeps plugin titlebar contributions on a contributed full-page route', () => {
    renderControls('/kanban')

    expect(pluginChrome()).not.toBeNull()
    expect(windowControls()).toBeNull()
    expect(appControls()).toBeNull()
  })

  it('keeps the app clusters on chat', () => {
    renderControls('/')

    expect(windowControls()).not.toBeNull()
    expect(appControls()).not.toBeNull()
  })

  it('hides the app clusters on an overlay', () => {
    renderControls('/settings')

    expect(windowControls()).toBeNull()
    expect(appControls()).toBeNull()
  })

  it('hides plugin titlebar contributions on an overlay', () => {
    renderControls('/settings')

    expect(pluginChrome()).toBeNull()
  })

  it('keeps the app clusters on a first-party workspace page', () => {
    renderControls('/skills')

    expect(windowControls()).not.toBeNull()
    expect(appControls()).not.toBeNull()
  })
})
