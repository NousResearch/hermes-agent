// @vitest-environment jsdom
/**
 * The rendered switcher must key a row by (profile, id). Two twins sharing one
 * stored id are two rows (#92454); a bare-id key collapses them, so React
 * reconciles one twin's row in the other's place — the highlighted row can
 * render the wrong twin's state. The store-level tests pin the helper; this one
 * pins that the JSX actually uses it.
 */

import { cleanup, render } from '@testing-library/react'
import { act } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $switcherIndex, $switcherOpen, $switcherSessions, closeSwitcher } from '@/store/session-switcher'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'
import type { SessionInfo } from '@/types/hermes'

import { SessionSwitcher } from './session-switcher'

stubResizeObserver()
stubMenuDomApis()

const TWIN_ID = '20260101_shared'

const twin = (profile: string, title: string): SessionInfo =>
  ({ connection_id: `source-${profile}`, id: TWIN_ID, profile, title }) as SessionInfo

const rowOf = (title: string): HTMLElement => {
  const label = [...document.querySelectorAll('span')].find(node => node.textContent === title)

  if (!label) {
    throw new Error(`no rendered row for ${title}`)
  }

  return label.parentElement as HTMLElement
}

const mount = (rows: SessionInfo[], index: number) => {
  closeSwitcher()
  $switcherSessions.set(rows)
  $switcherIndex.set(index)
  $switcherOpen.set(true)

  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <SessionSwitcher />
      </I18nProvider>
    </MemoryRouter>
  )
}

afterEach(() => {
  cleanup()
  closeSwitcher()
  $switcherSessions.set([])
  $switcherIndex.set(0)
  vi.restoreAllMocks()
})

describe('the rendered switcher keys each row by (profile, id)', () => {
  it('carries two twins as two rows without collapsing one onto the other', () => {
    const errors = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    const { unmount } = mount([twin('alpha', 'Alpha planning'), twin('beta', 'Beta planning')], 1)

    expect(document.body.textContent).toContain('Alpha planning')
    expect(document.body.textContent).toContain('Beta planning')
    // The highlighted row is the one the index names, not the other twin.
    expect(rowOf('Beta planning').className).toContain('bg-accent')
    expect(rowOf('Alpha planning').className).not.toContain('bg-accent')
    // A bare-id key is exactly the duplicate React.jsx reports here.
    expect(errors.mock.calls.flat().join(' ')).not.toContain('same key')

    unmount()
  })

  it('moves the highlight between twins when the index moves', () => {
    mount([twin('alpha', 'Alpha planning'), twin('beta', 'Beta planning')], 0)

    expect(rowOf('Alpha planning').className).toContain('bg-accent')

    act(() => {
      $switcherIndex.set(1)
    })

    expect(rowOf('Beta planning').className).toContain('bg-accent')
    expect(rowOf('Alpha planning').className).not.toContain('bg-accent')
  })
})
