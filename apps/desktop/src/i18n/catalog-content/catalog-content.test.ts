import { beforeAll, describe, expect, it } from 'vitest'

import type { CatalogEntry } from '@/app/capabilities/catalog/catalog-data'
import { EMPTY_FACETS, filterCatalog } from '@/app/capabilities/catalog/catalog-query'
import { catalogByline, localizeCatalogEntries, localizeCatalogEntry } from '@/app/capabilities/catalog/localize-catalog'
import { localizeConnectorCards } from '@/app/capabilities/connectors/localize-cards'
import type { ConnectorCardModel } from '@/app/capabilities/connectors/types'
import { localizeToolsets, toolTitle } from '@/app/capabilities/toolsets/localize-toolsets'
import type { ToolsetInfo } from '@/types/hermes'

import {
  type ContentDictionary,
  contentKeys,
  dictionaryStats,
  loadContentDictionary,
  lookupCategory,
  lookupCategoryLabel,
  lookupContent,
  overlayFrom,
  pluginCategoryText
} from './index'

const entry = (over: Partial<CatalogEntry>): CatalogEntry => ({
  id: 'official:x',
  name: 'x',
  description: 'English description',
  overview: '',
  category: '',
  categoryLabel: '',
  source: 'official',
  author: 'Hermes Agent',
  identifier: 'x',
  repo: '',
  sha: '',
  subdir: '',
  version: '',
  requiresHermes: '',
  tags: [],
  platforms: [],
  requirements: [],
  tools: [],
  hooks: [],
  sourceUrl: null,
  docsUrl: null,
  imageUrl: null,
  stars: null,
  search: 'x english description',
  ...over
})

const card = (over: Partial<ConnectorCardModel>): ConnectorCardModel =>
  ({
    inCatalog: true,
    name: 'X',
    residency: 'local',
    slug: 'x',
    state: 'available',
    stateWord: 'available',
    ways: { hosted: null, local: {} },
    ...over
  }) as unknown as ConnectorCardModel

beforeAll(async () => {
  await loadContentDictionary('ru')
})

describe('catalog content dictionary (ru)', () => {
  it('has a Russian skill name and description', () => {
    const text = lookupContent('skill', contentKeys('codex'), 'ru')
    expect(text?.name).toMatch(/[а-яё]/i)
    expect(text?.description).toMatch(/[а-яё]/i)
  })

  it('has a Russian plugin name for mem0', () => {
    expect(lookupContent('plugin', contentKeys('mem0'), 'ru')?.name).toMatch(/Mem0/)
    expect(lookupContent('plugin', contentKeys('mem0'), 'ru')?.name).toMatch(/[а-яё]/i)
  })

  it('has a Russian MCP server description', () => {
    expect(lookupContent('mcp', contentKeys('airtable'), 'ru')?.description).toMatch(/таблиц/i)
    expect(lookupContent('mcp', contentKeys('airtable'), 'ru')?.description).toMatch(/[а-яё]/i)
  })

  it('has Russian tool and toolset names', () => {
    expect(lookupContent('tool', contentKeys('web_search'), 'ru')?.name).toMatch(/[а-яё]/i)
    expect(lookupContent('toolset', contentKeys('browser'), 'ru')?.name).toMatch(/браузер/i)
  })

  it('ships entries for every kind', () => {
    const stats = dictionaryStats('ru')
    expect(Object.values(stats).filter(count => count > 0).length).toBeGreaterThanOrEqual(5)
  })

  it('returns null for English and for unknown items (no fake translation)', () => {
    expect(lookupContent('skill', contentKeys('codex'), 'en')).toBeNull()
    expect(lookupContent('skill', contentKeys('brand-new-skill-2027'), 'ru')).toBeNull()
    expect(lookupCategory('brand-new-category-2027', 'ru')).toBeNull()
  })

  it('resolves region locales and owner/name ids', () => {
    expect(lookupContent('skill', contentKeys('openai/skills/codex'), 'ru-RU')).not.toBeNull()
  })

  it('lets a registry overlay win over the bundled text', () => {
    const overlay: ContentDictionary = { skill: { codex: { name: 'Своё название' } } }
    expect(lookupContent('skill', contentKeys('codex'), 'ru', overlay)?.name).toBe('Своё название')
    expect(overlayFrom({ catalogContent: overlay })).toBe(overlay)
    expect(overlayFrom({})).toBeNull()
  })

  it('translates plugin taxonomy headings', () => {
    expect(pluginCategoryText('memory', 'ru')?.label).toBe('Память')
    expect(pluginCategoryText('memory', 'en')).toBeNull()
  })
})

describe('catalog entries', () => {
  it('shows Russian title/description, keeps ids and the original', () => {
    const raw = entry({ id: 'official:codex', name: 'codex', identifier: 'codex', installIdentifier: 'official/codex' })
    const out = localizeCatalogEntry(raw, 'skills', 'ru')
    expect(out.name).toMatch(/[а-яё]/i)
    expect(out.originalName).toBe('codex')
    expect(out.originalDescription).toBe('English description')
    expect([out.id, out.identifier, out.installIdentifier]).toEqual(['official:codex', 'codex', 'official/codex'])
    expect(catalogByline(out)).toBe('Codex · Hermes Agent')
  })

  it('switches back to English without touching the entry', () => {
    const raw = entry({ name: 'codex', identifier: 'codex' })
    expect(localizeCatalogEntry(raw, 'skills', 'ru')).not.toBe(raw)
    expect(localizeCatalogEntry(raw, 'skills', 'en')).toBe(raw)
    expect(localizeCatalogEntries([raw], 'skills', 'en')[0]).toBe(raw)
  })

  it('leaves a new, untranslated entry exactly as delivered', () => {
    const raw = entry({ name: 'shiny-new-thing', identifier: 'shiny-new-thing' })
    expect(localizeCatalogEntry(raw, 'skills', 'ru')).toBe(raw)
  })

  it('finds entries by Russian words and by the original technical name', () => {
    const rows = localizeCatalogEntries(
      [entry({ id: 'p:mem0', name: 'mem0', identifier: 'mem0', search: 'mem0 memory provider' })],
      'plugins',
      'ru'
    )

    const facets = EMPTY_FACETS
    expect(filterCatalog(rows, facets, 'память', () => false, 'plugins')).toHaveLength(1)
    expect(filterCatalog(rows, facets, 'mem0', () => false, 'plugins')).toHaveLength(1)
  })

  it('translates category slugs and English display labels at render time', () => {
    expect(lookupCategoryLabel('productivity', 'ru')).toMatch(/[а-яё]/i)
    expect(lookupCategoryLabel('Software Development', 'ru')).toMatch(/[а-яё]/i)
    expect(lookupCategoryLabel('productivity', 'en')).toBeNull()
    expect(lookupCategoryLabel('totally-new-category', 'ru')).toBeNull()
  })

  it('does not copy rows that only have a category (performance)', () => {
    const raw = entry({ name: 'unknown-zz', identifier: 'unknown-zz', category: 'productivity' })
    expect(localizeCatalogEntry(raw, 'skills', 'ru')).toBe(raw)
  })
})

describe('connectors and MCP cards', () => {
  it('translates a local MCP server and keeps its slug', () => {
    const [out] = localizeConnectorCards([card({ slug: 'airtable', name: 'airtable' })], 'ru')
    expect(out.name).toMatch(/Airtable/)
    expect(out.slug).toBe('airtable')
    expect(out.originalName).toBe('airtable')
  })

  it('keeps the brand title of a hosted connector and adds a Russian description', () => {
    const [out] = localizeConnectorCards([card({ slug: 'notion', name: 'Notion', residency: 'hosted' })], 'ru')
    expect(out.name).toBe('Notion')
    expect(out.description).toMatch(/[а-яё]/i)
  })

  it('leaves unknown connectors untouched', () => {
    const raw = card({ slug: 'zz-new', name: 'ZZ' })
    expect(localizeConnectorCards([raw], 'ru')[0]).toBe(raw)
  })
})

describe('toolsets', () => {
  const browser: ToolsetInfo = { name: 'browser', label: 'Browser', description: 'Control a browser', enabled: true, configured: true, tools: ['web_search'] } as ToolsetInfo

  it('localizes label/description but never the toolset name', () => {
    const [out] = localizeToolsets([browser], 'ru')
    expect(out.name).toBe('browser')
    expect(out.label).toMatch(/браузер/i)
    expect(out.search).toContain('browser')
    expect(toolTitle('web_search', 'ru')).toMatch(/[а-яё]/i)
    expect(localizeToolsets([browser], 'en')[0]).toBe(browser)
  })
})

describe('interface strings and search', () => {
  it('has the Russian detail-view labels and keeps the English ones', async () => {
    const { TRANSLATIONS } = await import('@/i18n/catalog')
    expect((TRANSLATIONS.ru as any).catalog.contentOriginal).toMatch(/[а-яё]/i)
    expect((TRANSLATIONS.en as any).catalog.contentOriginal).toBe('Original (English)')
  })

  it('finds tools by Russian words and by technical names', () => {
    const toolset = { name: 'browser', label: 'Browser', description: 'Control a browser', enabled: true, configured: true, tools: ['browser_navigate', 'web_search'] } as ToolsetInfo
    const [out] = localizeToolsets([toolset], 'ru')
    expect(out.search).toContain('браузер')
    expect(out.search).toContain('поиск')
    expect(out.name).toBe('browser')
  })

  it('keeps stable ids across languages', () => {
    const raw = entry({ id: 'official:codex', name: 'codex', identifier: 'codex', installIdentifier: 'official/codex' })
    const ru = localizeCatalogEntry(raw, 'skills', 'ru')
    const en = localizeCatalogEntry(raw, 'skills', 'en')
    expect([ru.id, ru.identifier, ru.installIdentifier]).toEqual([en.id, en.identifier, en.installIdentifier])
  })
})
