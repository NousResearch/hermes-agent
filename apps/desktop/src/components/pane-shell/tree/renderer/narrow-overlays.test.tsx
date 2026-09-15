import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { registry } from '@/contrib/registry'
import { I18nProvider, TRANSLATIONS, useI18n } from '@/i18n'
import { stubResizeObserver } from '@/test/jsdom'

import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree, $narrowViewport, declareDefaultTree } from '../store'

import { NarrowOverlays } from './narrow-overlays'

// Ground truth for "the Bots tab is still visible when the sessions sidebar
// collapses on a narrow window". A collapsible pane DOCKED into the sessions
// zone (SESSIONS | BOTS) must leave the grid with the zone, and the narrow
// edge overlay must mirror the zone's tab strip so the docked pane stays
// reachable — not just the zone's first pane.

beforeAll(() => {
  stubResizeObserver()
})

const disposers: (() => void)[] = []

const registerPane = (id: string, title: string, data: Record<string, unknown>, body: string) => {
  disposers.push(
    registry.register({
      area: 'panes',
      data,
      id,
      render: () => <div data-testid={`${id}-body`}>{body}</div>,
      title
    })
  )
}

beforeEach(() => {
  window.localStorage.clear()
  $hiddenTreePanes.set(new Set())

  registerPane('sessions', 'sessions', { collapsible: true, placement: 'left', width: '237px' }, 'session rows')
  registerPane('bots', 'Bots', { collapsible: true, placement: 'left', width: '260px' }, 'bot roster')
  registerPane('workspace', 'workspace', { placement: 'main', uncloseable: true }, 'chat')

  declareDefaultTree(split('row', [group(['sessions', 'bots']), group(['workspace'])]))
  $narrowViewport.set(true)
})

afterEach(() => {
  cleanup()
  $narrowViewport.set(false)
  $layoutTree.set(null)
  disposers.splice(0).forEach(dispose => dispose())
})

const revealPane = (id: string) => {
  act(() => {
    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id, mode: 'open' } }))
  })
}

const overlayTab = (paneId: string) =>
  globalThis.document.querySelector<HTMLElement>(`[data-narrow-overlay-tab="${paneId}"]`)

function LocalizedPaneTitle() {
  const { t } = useI18n()

  return t.sidebar.sessions
}

function LocaleToggle() {
  const { locale, setLocale } = useI18n()

  return <button onClick={() => void setLocale(locale === 'en' ? 'ko' : 'en')}>Change test language</button>
}

function StatefulPane() {
  const [value, setValue] = useState('')

  return <input aria-label="Session filter" onChange={event => setValue(event.target.value)} value={value} />
}

describe('narrow overlay of a stacked zone', () => {
  it('updates a live tab title without re-registering panes or resetting their state and preserves custom titles', () => {
    disposers.push(
      registry.register({
        area: 'panes',
        data: { collapsible: true, placement: 'left', tabTitle: () => <LocalizedPaneTitle /> },
        id: 'sessions',
        render: () => <StatefulPane />,
        title: TRANSLATIONS.en.sidebar.sessions
      })
    )
    registerPane('bots', 'My custom workspace', { collapsible: true, placement: 'left' }, 'bot roster')

    const contributions = registry.getArea('panes')
    const tree = $layoutTree.get()

    const { getByRole } = render(
      <I18nProvider configClient={null} initialLocale="en">
        <LocaleToggle />
        <NarrowOverlays />
      </I18nProvider>
    )

    revealPane('sessions')
    const input = getByRole('textbox') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'keep my filter' } })
    expect(overlayTab('sessions')?.textContent).toBe(TRANSLATIONS.en.sidebar.sessions)

    fireEvent.click(getByRole('button', { name: 'Change test language' }))

    expect(overlayTab('sessions')?.textContent).toBe(TRANSLATIONS.ko.sidebar.sessions)
    expect(overlayTab('bots')?.textContent).toBe('My custom workspace')
    expect(overlayTab('sessions')?.getAttribute('aria-selected')).toBe('true')
    expect(getByRole('textbox')).toBe(input)
    expect(input.value).toBe('keep my filter')
    expect(registry.getArea('panes')).toBe(contributions)
    expect($layoutTree.get()).toBe(tree)

    fireEvent.click(getByRole('button', { name: 'Change test language' }))
    expect(overlayTab('sessions')?.textContent).toBe(TRANSLATIONS.en.sidebar.sessions)
  })

  it('mirrors the zone tab strip so every stacked collapsible stays reachable', () => {
    const { getByTestId, queryByTestId } = render(<NarrowOverlays />)

    revealPane('sessions')

    // Both zone-mates surface as tabs; the revealed pane's body is on screen.
    expect(overlayTab('sessions')).toBeTruthy()
    expect(overlayTab('bots')).toBeTruthy()
    expect(getByTestId('sessions-body')).toBeTruthy()
    expect(queryByTestId('bots-body')).toBeNull()

    // Clicking the BOTS tab swaps the overlay to the docked pane.
    fireEvent.pointerDown(overlayTab('bots')!, { button: 0 })
    expect(getByTestId('bots-body')).toBeTruthy()
    expect(queryByTestId('sessions-body')).toBeNull()
  })

  it('keeps the stripless form for a zone with a single collapsible', () => {
    // Direct set: declareDefaultTree only ADOPTS into an existing tree — it
    // would keep the beforeEach zone (with bots) instead of replacing it.
    $layoutTree.set(split('row', [group(['sessions']), group(['workspace'])]))

    const { getByTestId } = render(<NarrowOverlays />)

    revealPane('sessions')

    expect(getByTestId('sessions-body')).toBeTruthy()
    expect(overlayTab('sessions')).toBeNull()
  })
})
