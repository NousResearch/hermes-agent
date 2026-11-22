import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, expect, it } from 'vitest'

import { stubMenuDomApis } from '@/test/jsdom'

import { PageHeader, type PageTab, PageTabs } from './page-header'

// A tapped tab scrolls itself into view; jsdom has no scrollIntoView.
stubMenuDomApis()

afterEach(cleanup)

const TABS: PageTab[] = [
  { id: 'skills', label: 'Skills', meta: 39 },
  { id: 'plugins', label: 'Plugins', meta: null },
  { id: 'tools', label: 'Tools' }
]

function Page() {
  const [tab, setTab] = useState('skills')

  return (
    <PageHeader
      actions={<button type="button">Refresh</button>}
      description="Everything Hermes can use."
      tabs={<PageTabs label="Capabilities" onChange={setTab} tabs={TABS} value={tab} />}
      title="Capabilities"
    />
  )
}

it('names the page and exposes its tabs as one tab list with a single selected tab', () => {
  render(<Page />)

  expect(screen.getByRole('heading', { level: 1, name: 'Capabilities' })).toBeTruthy()
  expect(screen.getByText('Everything Hermes can use.')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Refresh' })).toBeTruthy()
  expect(screen.getByRole('tablist', { name: 'Capabilities' })).toBeTruthy()

  const selected = screen.getAllByRole('tab').filter(tab => tab.getAttribute('aria-selected') === 'true')
  expect(selected.map(tab => tab.textContent)).toEqual(['Skills39'])

  fireEvent.click(screen.getByRole('tab', { name: /Plugins/ }))
  expect(screen.getByRole('tab', { name: /Plugins/ }).getAttribute('aria-selected')).toBe('true')
  expect(screen.getByRole('tab', { name: /Skills/ }).getAttribute('aria-selected')).toBe('false')
})

it('keeps only the selected tab in the tab order and moves selection with arrow, Home and End keys', () => {
  render(<Page />)
  const tab = (name: RegExp) => screen.getByRole('tab', { name })

  expect(tab(/Skills/).tabIndex).toBe(0)
  expect(tab(/Plugins/).tabIndex).toBe(-1)

  tab(/Skills/).focus()
  fireEvent.keyDown(tab(/Skills/), { key: 'ArrowRight' })
  expect(tab(/Plugins/).getAttribute('aria-selected')).toBe('true')
  expect(tab(/Plugins/).ownerDocument.activeElement).toBe(tab(/Plugins/))

  fireEvent.keyDown(tab(/Plugins/), { key: 'End' })
  expect(tab(/Tools/).getAttribute('aria-selected')).toBe('true')

  fireEvent.keyDown(tab(/Tools/), { key: 'ArrowRight' })
  expect(tab(/Skills/).getAttribute('aria-selected')).toBe('true')

  fireEvent.keyDown(tab(/Skills/), { key: 'ArrowLeft' })
  expect(tab(/Tools/).getAttribute('aria-selected')).toBe('true')

  fireEvent.keyDown(tab(/Tools/), { key: 'Home' })
  expect(tab(/Skills/).getAttribute('aria-selected')).toBe('true')
})

it('shows a loading placeholder in place of a count that has not arrived', () => {
  render(<Page />)

  expect(screen.getByRole('tab', { name: /Plugins/ }).querySelector('[data-slot="count-skeleton"]')).toBeTruthy()
  expect(screen.getByRole('tab', { name: /Tools/ }).querySelector('[data-slot="count-skeleton"]')).toBeNull()
})

it('names the page by its H1 alone, with an eyebrow above it and a badge beside it', () => {
  render(
    <PageHeader
      badge={<span>Hermes machine</span>}
      eyebrow={<span>~/.hermes</span>}
      tabs={<PageTabs label="Agent" onChange={() => {}} tabs={TABS} value="skills" variant="segmented" />}
      title="research"
    />
  )

  expect(screen.getByRole('heading', { level: 1, name: 'research' })).toBeTruthy()
  expect(screen.getByText('~/.hermes')).toBeTruthy()
  expect(screen.getByText('Hermes machine')).toBeTruthy()
})

it('keeps the tab roles and keys on the segmented tab bar', () => {
  function Segmented() {
    const [tab, setTab] = useState('skills')

    return <PageTabs label="Agent" onChange={setTab} tabs={TABS} value={tab} variant="segmented" />
  }

  render(<Segmented />)
  const tab = (name: RegExp) => screen.getByRole('tab', { name })

  expect(screen.getByRole('tablist', { name: 'Agent' })).toBeTruthy()
  tab(/Skills/).focus()
  fireEvent.keyDown(tab(/Skills/), { key: 'End' })
  expect(tab(/Tools/).getAttribute('aria-selected')).toBe('true')
  expect(tab(/Tools/).ownerDocument.activeElement).toBe(tab(/Tools/))
  expect(tab(/Tools/).tabIndex).toBe(0)
  expect(tab(/Skills/).tabIndex).toBe(-1)
})
