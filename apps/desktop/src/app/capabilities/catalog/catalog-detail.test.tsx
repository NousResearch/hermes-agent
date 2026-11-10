import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { parseCatalog } from './catalog-data'
import { CatalogDetail } from './catalog-detail'

const locale = { current: 'en' }

vi.mock('@/i18n', async importOriginal => {
  const actual = (await importOriginal()) as {
    useI18n: () => { locale: string }
  } & Record<string, unknown>

  return {
    ...actual,
    useI18n: () => ({
      ...actual.useI18n(),
      locale: locale.current
    })
  }
})

const ID_DESC = 'Teks Indonesia'
const EN_DESC = 'English'
const UNIQUE_OVERVIEW = 'A unique overview that is not the card description.'

const detailEntry = (overrides: Record<string, unknown> = {}) =>
  parseCatalog('plugins', [
    {
      name: 'stage',
      description: EN_DESC,
      overview: EN_DESC,
      tier: 'community',
      i18n: { id: { description: ID_DESC } },
      ...overrides
    }
  ])[0]

const renderDetail = (entry = detailEntry()) =>
  render(<CatalogDetail dialog={false} entry={entry} kind="plugins" related={[]} />)

describe('catalog detail copy', () => {
  it('hides English overview when it duplicates the English description and a localized description is shown', () => {
    locale.current = 'id'
    renderDetail()

    expect(screen.getByText(ID_DESC)).toBeTruthy()
    expect(screen.queryByText(EN_DESC)).toBeNull()
  })

  it('still shows a unique overview that is neither the English nor the localized description', () => {
    locale.current = 'id'
    renderDetail(detailEntry({ overview: UNIQUE_OVERVIEW }))

    expect(screen.getByText(ID_DESC)).toBeTruthy()
    expect(screen.getByText(UNIQUE_OVERVIEW)).toBeTruthy()
    expect(screen.queryByText(EN_DESC)).toBeNull()
  })
})
