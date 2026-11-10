import { describe, expect, it } from 'vitest'

import { catalogLocaleCandidates, parseCatalogI18n, resolveCatalogCopy } from './catalog-i18n'

const ID_TITLE = 'Stage'
const ID_DESC = 'Menahan pemanggilan write_file dan patch Hermes di salinan sementara.'
const EN_DESC = 'Stage Hermes write_file and patch calls in a scratch copy.'

const stage = {
  name: 'stage',
  title: '',
  description: EN_DESC,
  i18n: { id: { title: ID_TITLE, description: ID_DESC } }
}

describe('catalog card i18n', () => {
  it('returns Indonesian title and description for id and id-ID', () => {
    expect(resolveCatalogCopy(stage, 'id')).toEqual({ title: ID_TITLE, description: ID_DESC })
    expect(resolveCatalogCopy(stage, 'id-ID')).toEqual({ title: ID_TITLE, description: ID_DESC })
    expect(resolveCatalogCopy(stage, 'id_ID')).toEqual({ title: ID_TITLE, description: ID_DESC })
  })

  it('falls back to English when the locale is missing (ru with only id)', () => {
    expect(resolveCatalogCopy(stage, 'ru')).toEqual({ title: 'stage', description: EN_DESC })
  })

  it('still works with no i18n map', () => {
    expect(resolveCatalogCopy({ name: 'plain', description: EN_DESC }, 'id')).toEqual({
      title: 'plain',
      description: EN_DESC
    })
  })

  it('ignores invalid and empty i18n values and falls back to English', () => {
    const entry = {
      name: 'stage',
      description: EN_DESC,
      i18n: parseCatalogI18n({
        id: { title: '', description: '   ' },
        ru: 'not a mapping',
        'zh-hant': { title: null, description: 1 }
      })
    }

    expect(entry.i18n).toEqual({})
    expect(resolveCatalogCopy(entry, 'id').description).toBe(EN_DESC)
  })

  it('does not fall back zh-hant to zh', () => {
    const entry = {
      name: 'x',
      description: 'en',
      i18n: { zh: { description: 'simplified' } }
    }

    expect(resolveCatalogCopy(entry, 'zh-hant').description).toBe('en')
    expect(resolveCatalogCopy(entry, 'zh').description).toBe('simplified')
    expect(catalogLocaleCandidates('zh-hant')).toEqual(['zh-hant'])
    expect(catalogLocaleCandidates('id-ID')).toEqual(['id-id', 'id'])
    expect(catalogLocaleCandidates('es-419')).toEqual(['es'])
    expect(catalogLocaleCandidates('xx-419')).toEqual(['xx-419', 'xx'])
    expect(catalogLocaleCandidates('xx-12345')).toEqual(['xx-12345'])
    expect(catalogLocaleCandidates('en-gb')).toEqual(['en-gb', 'en'])
    expect(catalogLocaleCandidates('zh-sg')).toEqual(['zh-sg', 'zh'])
  })

  it('uses the English title when present, including for missing locales', () => {
    const withTitle = { ...stage, title: 'Stage' }

    expect(resolveCatalogCopy(withTitle, 'en')).toEqual({ title: 'Stage', description: EN_DESC })
    expect(resolveCatalogCopy(withTitle, 'ru')).toEqual({ title: 'Stage', description: EN_DESC })
    expect(resolveCatalogCopy(withTitle, 'id')).toEqual({ title: ID_TITLE, description: ID_DESC })
    expect(withTitle.name).toBe('stage')
  })

  it('never changes the catalog key name', () => {
    expect(stage.name).toBe('stage')
    expect(resolveCatalogCopy({ ...stage, title: 'English Title' }, 'id').title).toBe(ID_TITLE)
  })
})
