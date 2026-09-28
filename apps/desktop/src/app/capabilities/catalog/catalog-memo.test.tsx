import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render } from '@testing-library/react'
import { useCallback, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'

import { CatalogBrowser } from './catalog-browser'
import { parseCatalog, type CatalogEntry } from './catalog-data'
import * as catalogQuery from './catalog-query'
import { $catalogCardView } from './store'

afterEach(() => {
  cleanup()
  queryClient.clear()
  $catalogCardView.set(true)
  vi.restoreAllMocks()
})

function seed() {
  const entries = parseCatalog(
    'skills',
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
  queryClient.setQueryData(['public-catalog', 'skills'], entries)
}

// Unrelated parent state changes must not re-derive the whole feed.
describe('catalog memo', () => {
  it('does not re-derive facets when an unrelated parent render changes nothing', () => {
    $catalogCardView.set(true)
    seed()
    const tagsSpy = vi.spyOn(catalogQuery, 'catalogTags')
    const categoriesSpy = vi.spyOn(catalogQuery, 'catalogCategories')
    const sourcesSpy = vi.spyOn(catalogQuery, 'catalogSources')

    function Harness() {
      const [query, setQuery] = useState('')
      const [tick, setTick] = useState(0)
      const isInstalled = useCallback((_entry: CatalogEntry) => false, [])
      const onInstall = useCallback((_entry: CatalogEntry) => {}, [])
      return (
        <QueryClientProvider client={queryClient}>
          <button data-testid="tick" onClick={() => setTick(t => t + 1)}>
            tick {tick}
          </button>
          <CatalogBrowser
            isInstalled={isInstalled}
            // Like skill-catalog.tsx: fresh closures/elements every owner render.
            isInstalling={(_entry: CatalogEntry) => false}
            kind="skills"
            notice={<span>tick {tick}</span>}
            onInstall={onInstall}
            onQueryChange={setQuery}
            query={query}
            renderInstalledAction={(_entry: CatalogEntry) => null}
          />
        </QueryClientProvider>
      )
    }

    const { getByTestId, rerender } = render(<Harness />)
    const baseline = { tags: tagsSpy.mock.calls.length, cats: categoriesSpy.mock.calls.length, srcs: sourcesSpy.mock.calls.length }
    expect(baseline.tags).toBeGreaterThan(0)

    // Same props, parent re-render only.
    rerender(<Harness />)
    expect(getByTestId('tick')).toBeTruthy()

    expect(tagsSpy.mock.calls.length).toBe(baseline.tags)
    expect(categoriesSpy.mock.calls.length).toBe(baseline.cats)
    expect(sourcesSpy.mock.calls.length).toBe(baseline.srcs)
  })
})
