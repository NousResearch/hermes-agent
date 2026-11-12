import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { CatalogCard } from './catalog-card'
import { parseCatalog } from './catalog-data'

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

const ID_DESC = 'Menahan pemanggilan write_file dan patch Hermes di salinan sementara.'
const EN_DESC = 'Stage Hermes write_file and patch calls in a scratch copy.'

const cardEntry = (overrides: Record<string, unknown> = {}) =>
  parseCatalog('plugins', [
    {
      name: 'stage',
      title: '',
      description: EN_DESC,
      tier: 'community',
      i18n: { id: { title: 'Stage', description: ID_DESC } },
      ...overrides
    }
  ])[0]

const namelessTitle = cardEntry()
const englishTitle = cardEntry({ title: 'Stage' })

const renderCard = (entry = namelessTitle) =>
  render(
    <CatalogCard
      accentIndex={0}
      action={null}
      entry={entry}
      onCategory={() => {}}
      onOpen={() => {}}
      onSearch={() => {}}
      onTag={() => {}}
    />
  )

describe('catalog card copy', () => {
  it('renders Indonesian title and description when the locale matches, not the English field', () => {
    locale.current = 'id'
    renderCard()

    expect(screen.getByText('Stage')).toBeTruthy()
    expect(screen.getByText(ID_DESC)).toBeTruthy()
    expect(screen.queryByText(EN_DESC)).toBeNull()
  })

  it('renders the English description when the locale has no catalog strings', () => {
    locale.current = 'ru'
    renderCard()

    expect(screen.getByText(EN_DESC)).toBeTruthy()
    expect(screen.queryByText(ID_DESC)).toBeNull()
  })

  it('renders the English title Stage, not the catalog key, and keeps aria-label as name', () => {
    locale.current = 'en'
    renderCard(englishTitle)

    expect(screen.getByText('Stage')).toBeTruthy()
    expect(screen.getByText(EN_DESC)).toBeTruthy()
    expect(screen.getByLabelText('stage')).toBeTruthy()
    expect(screen.queryByText(ID_DESC)).toBeNull()
  })
})
