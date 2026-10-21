import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'

import { PageSearchShell } from '../../page-search-shell'

import { CatalogBrowser } from './catalog-browser'
import { type CatalogKind, parseCatalog } from './catalog-data'
import { $catalogCardView } from './store'

beforeEach(() => {
  $catalogCardView.set(true)
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  $catalogCardView.set(true)
})

function setup(kind: CatalogKind) {
  const entries = parseCatalog(
    kind,
    ['alpha', 'beta'].map(name => ({
      name,
      identifier: `official/${name}`,
      installIdentifier: `official/${name}`,
      source: 'official',
      tier: 'official',
      category: name === 'alpha' ? 'memory' : 'voice',
      description: `${name} workflow`,
      repo: `https://github.com/example/${name}`,
      sourceUrl: `https://github.com/example/${name}`,
      docsUrl: `https://example.com/${name}`
    }))
  )

  queryClient.setQueryData(['public-catalog', kind], entries)
  const onInstall = vi.fn()

  function Harness() {
    const [query, setQuery] = useState('')

    return (
      <QueryClientProvider client={queryClient}>
        <PageSearchShell onSearchChange={setQuery} searchPlaceholder="Search catalog" searchValue={query}>
          <CatalogBrowser
            isInstalled={entry => entry.name === 'beta'}
            kind={kind}
            onInstall={onInstall}
            onQueryChange={setQuery}
            query={query}
          />
        </PageSearchShell>
      </QueryClientProvider>
    )
  }

  return { entries, onInstall, ...render(<Harness />) }
}

const card = (name: string) =>
  screen.getAllByRole('article').find(article => within(article).queryByRole('button', { name }))!

describe.each(['skills', 'plugins'] as const)('%s catalog layouts', kind => {
  it('defaults to cards, with installed entries already switched on', () => {
    const { container } = setup(kind)

    expect(screen.getAllByRole('article')).toHaveLength(2)
    expect(container.querySelector('[data-catalog-list]')).toBeNull()
    expect(screen.getByRole('button', { name: 'List view' })).toBeTruthy()
    expect(
      within(card('beta')).getByRole<HTMLButtonElement>('switch', { name: 'Added beta' }).getAttribute('aria-checked')
    ).toBe('true')
  })

  it('installs from the card switch without opening its details', () => {
    const { entries, onInstall } = setup(kind)

    fireEvent.click(within(card('alpha')).getByRole('switch', { name: 'Add alpha' }))

    expect(onInstall).toHaveBeenCalledExactlyOnceWith(entries[0])
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('opens the shared detail dialog with source links and the same install target', async () => {
    const { entries, onInstall } = setup(kind)
    fireEvent.click(screen.getByRole('button', { name: 'alpha' }))
    const dialog = screen.getByRole('dialog', { name: 'alpha' })

    expect(within(dialog).getByRole('link', { name: 'Repository' }).getAttribute('href')).toBe(entries[0].sourceUrl)
    expect(within(dialog).getByRole('link', { name: 'Documentation' }).getAttribute('href')).toBe(entries[0].docsUrl)
    fireEvent.click(within(dialog).getByRole('switch', { name: 'Add alpha' }))
    expect(onInstall).toHaveBeenCalledExactlyOnceWith(entries[0])

    if (kind === 'plugins') {
      // Release the inspector before the existing plugin installation dialog opens.
      await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    } else {
      fireEvent.click(within(dialog).getByRole('button', { name: 'Close' }))
      await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    }
  })

  it('preserves the parent search and filter when switching to list and back', async () => {
    const { container } = setup(kind)
    const search = screen.getByRole<HTMLInputElement>('textbox', { name: 'Search catalog' })
    fireEvent.change(search, { target: { value: 'workflow' } })
    fireEvent.click(screen.getByRole('button', { name: /^Memory/, pressed: false }))
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(1))

    fireEvent.click(screen.getByRole('button', { name: 'List view' }))

    expect(container.querySelector(`[data-catalog-list="${kind}"]`)).not.toBeNull()
    expect(screen.queryByRole('article')).toBeNull()
    expect(screen.getByRole('heading', { name: 'alpha' })).toBeTruthy()
    expect(screen.queryByRole('heading', { name: 'beta' })).toBeNull()
    expect(screen.getByRole('button', { name: /^Memory/, pressed: true })).toBeTruthy()
    expect(screen.getByRole('textbox', { name: 'Search catalog' })).toBe(search)
    expect(search.value).toBe('workflow')

    fireEvent.click(screen.getByRole('button', { name: 'Card view' }))
    expect(screen.getAllByRole('article')).toHaveLength(1)
    expect(screen.getByRole('button', { name: /^Memory/, pressed: true })).toBeTruthy()
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})

it('shares the saved layout choice when moving between Skills and Plugins', () => {
  setup('skills')
  fireEvent.click(screen.getByRole('button', { name: 'List view' }))
  expect($catalogCardView.get()).toBe(false)
  cleanup()

  const { container } = setup('plugins')
  expect(container.querySelector('[data-catalog-list="plugins"]')).not.toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Card view' }))
  expect($catalogCardView.get()).toBe(true)
})
