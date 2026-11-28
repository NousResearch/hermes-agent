import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { CatalogCard } from './catalog-card'
import { type CatalogEntry, parseCatalog } from './catalog-data'
import { CatalogDetail } from './catalog-detail'
import { CatalogListRow } from './catalog-list-row'

// First row of the recorded Team search response, with no author supplied.
const row = {
  name: 'conference-contact-capture',
  source: 'plugin.9d52e3990755847056e40e39e61be6814336daa89ba2ebf4c42e1a56eeb068db.team',
  identifier: 'plugin.9d52e3990755847056e40e39e61be6814336daa89ba2ebf4c42e1a56eeb068db.team:090114f0-0bff-406c-afd8-87f80afe8112',
  provider: {
    id: 'team', plugin_id: 'collective-wisdom', label: 'Team skills · Nous Research',
    version: '1', author: null, managed_state: 'not_installed', scan_verdict: 'unknown', actions: ['install']
  }
}

const handlers = { onOpen: vi.fn(), onCategory: vi.fn(), onTag: vi.fn(), onSearch: vi.fn() }
const surfaces = ['card', 'hero', 'list', 'detail'] as const

function mount(surface: typeof surfaces[number], entry: CatalogEntry) {
  if (surface === 'list') {
    return render(<CatalogListRow {...handlers} entry={entry} installed={false} kind="skills" selected={false} />)
  }

  if (surface === 'detail') {
    return render(<CatalogDetail dialog={false} entry={entry} kind="skills" related={[]} />)
  }

  return render(<CatalogCard {...handlers} accentIndex={0} action={null} entry={entry} variant={surface === 'hero' ? 'hero' : 'default'} />)
}

afterEach(cleanup)

it.each(surfaces)('%s uses the provider label without inventing a missing author', surface => {
  const [entry] = parseCatalog('skills', [{ ...row, author: row.provider.author }])
  const { container } = mount(surface, entry)

  expect(container.textContent).not.toContain(row.source)
  expect(screen.getAllByText(row.provider.label).length).toBeGreaterThan(0)
  expect(entry.author).toBe('')
  expect(entry.provider?.author).toBeNull()

  if (surface === 'hero') {
    expect(screen.getByTitle(row.provider.label).textContent).toBe(row.provider.label)
  }
})

it.each(surfaces)('%s prefers a real author over the provider label', surface => {
  const [entry] = parseCatalog('skills', [{ ...row, author: 'A real author', provider: { ...row.provider, author: 'A real author' } }])
  mount(surface, entry)

  expect(screen.getByText('A real author')).toBeTruthy()
  expect(entry.author).toBe('A real author')
})

it.each(surfaces)('%s keeps stock attribution unchanged', surface => {
  const [entry] = parseCatalog('skills', [{ name: 'Stock skill', source: 'official' }])
  mount(surface, entry)

  expect(screen.getAllByText('Official')).toHaveLength(2)
  expect(entry.author).toBe('')
})

it.each(surfaces)('%s keeps stock authors unchanged', surface => {
  const [entry] = parseCatalog('skills', [{ name: 'Stock skill', source: 'official', author: 'Stock author' }])
  mount(surface, entry)

  expect(screen.getByText('Stock author')).toBeTruthy()
})
