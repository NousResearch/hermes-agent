import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { CatalogFilterBar } from './catalog-filter-bar'
import { EMPTY_FACETS } from './catalog-query'

const renderBar = (sources: string[], onSource = vi.fn()) => {
  render(
    <CatalogFilterBar
      actions={null}
      categories={[]}
      facets={{ ...EMPTY_FACETS, sources }}
      kind="plugins"
      onCategory={vi.fn()}
      onClear={vi.fn()}
      onInstalled={vi.fn()}
      onSource={onSource}
      onTag={vi.fn()}
      resultCount={0}
      sortControl={null}
      sources={['official', 'verified', 'community']}
      tags={[]}
    />
  )
  fireEvent.click(screen.getByRole('button', { name: /^tier$/i }))

  return onSource
}

describe('tier chips', () => {
  it('collapses to All when the last unselected tier is picked', () => {
    const onSource = renderBar(['official', 'verified'])

    fireEvent.click(screen.getByRole('button', { name: 'Community' }))
    expect(onSource).toHaveBeenCalledWith(null)
  })

  it('toggles a tier normally while others stay unselected', () => {
    const onSource = renderBar(['official'])

    fireEvent.click(screen.getByRole('button', { name: 'Verified' }))
    expect(onSource).toHaveBeenCalledWith('verified')
    fireEvent.click(screen.getByRole('button', { name: 'Official' }))
    expect(onSource).toHaveBeenCalledWith('official')
  })
})
