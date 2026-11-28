import { describe, expect, it } from 'vitest'

import type { ConnectorAuthType, ConnectorCardModel } from '../types'

import {
  type ConnectionCardSection,
  EMPTY_CONNECTIONS_FILTER,
  groupCardsBySubgroup,
  groupCardsForCategory,
  groupCardsForMainPage
} from './filter-logic'
import {
  ADVANCED_CATEGORY_ORDER,
  cardCtaKind,
  categoryOfCard,
  categoryOrderForMode,
  categoryPreviewLimit,
  CONNECTION_TAXONOMY,
  CONNECTION_TAXONOMY_APPS,
  connectionRank,
  type ConnectionTaxonomyApp,
  isConnectedCard,
  SIMPLE_CATEGORY_ORDER,
  subgroupOrderForCategory,
  suggestedCardsForMode,
  taxonomyAppForCard
} from './taxonomy'

interface CardOptions {
  authType?: ConnectorAuthType
  bundledAuthType?: ConnectorAuthType
  connected?: boolean
  installed?: boolean
  local?: boolean
  name?: string
}

function card(slug: string, options: CardOptions = {}): ConnectorCardModel {
  const name = options.name ?? slug

  if (options.local === true) {
    const installed = options.installed ?? true
    const state = installed ? 'connected' : 'available'

    return {
      inCatalog: true,
      name,
      residency: 'local',
      slug,
      state,
      stateWord: installed ? 'serverOn' : 'available',
      ways: {
        hosted: null,
        local: {
          authType: options.authType,
          installed,
          state
        }
      }
    }
  }

  const connected = options.connected ?? false
  const state = connected ? 'connected' : 'available'
  const bundled =
    options.bundledAuthType === undefined
      ? null
      : { authType: options.bundledAuthType, installed: false, state: 'available' as const }

  return {
    inCatalog: true,
    name,
    residency: 'hosted',
    slug,
    state,
    stateWord: connected ? 'connected' : 'available',
    ways: {
      hosted: {
        accounts: [],
        connected,
        retiredAccounts: [],
        state
      },
      local: bundled
    }
  }
}

function taxonomyCard(entry: ConnectionTaxonomyApp, options: CardOptions = {}): ConnectorCardModel {
  return card(entry.slug, { name: entry.name, ...options })
}

function allCards(connectedSlugs: readonly string[] = []): ConnectorCardModel[] {
  return CONNECTION_TAXONOMY_APPS.map(entry =>
    taxonomyCard(entry, { connected: connectedSlugs.includes(entry.slug) })
  )
}

const CATEGORY_COUNTS = {
  'customers-growth': 16,
  'data-analytics': 8,
  'design-content': 10,
  engineering: 24,
  money: 11,
  reference: 6,
  'travel-fitness': 4,
  'work-productivity': 20
} as const

const section = (sections: readonly ConnectionCardSection[], id: string) =>
  sections.find(candidate => candidate.id === id)

describe('connection taxonomy', () => {
  it('covers all eight categories with the Figma app counts', () => {
    expect(CONNECTION_TAXONOMY.map(category => category.id)).toEqual([...SIMPLE_CATEGORY_ORDER])
    expect(CONNECTION_TAXONOMY_APPS).toHaveLength(99)

    for (const category of CONNECTION_TAXONOMY) {
      expect(category.subgroups.flatMap(subgroup => subgroup.apps)).toHaveLength(CATEGORY_COUNTS[category.id])
    }
  })

  it('preserves the Figma subgroup order inside each category', () => {
    expect(CONNECTION_TAXONOMY.find(category => category.id === 'work-productivity')?.subgroups.map(x => x.id)).toEqual(
      ['docs-files', 'communication', 'projects-tasks', 'meetings-scheduling']
    )
    expect(CONNECTION_TAXONOMY.find(category => category.id === 'engineering')?.subgroups.map(x => x.id)).toEqual([
      'hosting-databases',
      'automation-ai-apis',
      'monitoring-incidents',
      'code-review-ci'
    ])
    expect(CONNECTION_TAXONOMY.find(category => category.id === 'reference')?.subgroups.map(x => x.id)).toEqual([
      'docs-knowledge'
    ])
  })

  it('exposes each category’s Figma subgroup order for the See-All page', () => {
    expect(subgroupOrderForCategory('work-productivity')).toEqual([
      'docs-files',
      'communication',
      'projects-tasks',
      'meetings-scheduling'
    ])
    expect(subgroupOrderForCategory('customers-growth')).toEqual([
      'marketing-seo',
      'sales-crm',
      'customer-support',
      'hiring-people'
    ])
    expect(subgroupOrderForCategory('engineering')).toEqual([
      'hosting-databases',
      'automation-ai-apis',
      'monitoring-incidents',
      'code-review-ci'
    ])
    expect(subgroupOrderForCategory('reference')).toEqual(['docs-knowledge'])
  })

  it('uses the Figma preview limits', () => {
    expect(categoryPreviewLimit('work-productivity')).toBe(6)
    expect(categoryPreviewLimit('engineering')).toBe(6)
    expect(categoryPreviewLimit('design-content')).toBe(4)
    expect(categoryPreviewLimit('money')).toBe(4)
    expect(categoryPreviewLimit('customers-growth')).toBe(4)
    expect(categoryPreviewLimit('data-analytics')).toBe(4)
    expect(categoryPreviewLimit('travel-fitness')).toBe(4)
    expect(categoryPreviewLimit('reference')).toBe(4)
  })

  it('orders categories differently for Simple and Advanced', () => {
    expect(categoryOrderForMode('simple')).toEqual(SIMPLE_CATEGORY_ORDER)
    expect(categoryOrderForMode('advanced')).toEqual(ADVANCED_CATEGORY_ORDER)
    expect(SIMPLE_CATEGORY_ORDER[0]).toBe('work-productivity')
    expect(ADVANCED_CATEGORY_ORDER[0]).toBe('engineering')
    expect(ADVANCED_CATEGORY_ORDER.at(-1)).toBe('travel-fitness')
  })

  it('matches normalized slugs, aliases and display names', () => {
    expect(taxonomyAppForCard(card('stripe_mcp', { name: 'Stripe' }))).toMatchObject({
      category: 'money',
      name: 'Stripe',
      rank: 1,
      subgroup: 'payments-commerce'
    })
    expect(taxonomyAppForCard(card('hugging_face', { name: 'Hugging Face' }))).toMatchObject({
      category: 'engineering',
      name: 'Hugging Face',
      subgroup: 'automation-ai-apis'
    })
    expect(taxonomyAppForCard(card('unrecognised', { name: 'monday.com' }))).toMatchObject({
      category: 'work-productivity',
      name: 'monday.com',
      rank: 16,
      subgroup: 'projects-tasks'
    })
    expect(taxonomyAppForCard(card('atlassian'))).toMatchObject({ name: 'Jira', subgroup: 'projects-tasks' })
    expect(taxonomyAppForCard(card('wordpress', { name: 'WordPress.com' }))).toMatchObject({
      category: 'design-content',
      subgroup: 'websites-assets'
    })
  })

  it('falls back to "other" for an unmapped local server', () => {
    const custom = card('my-private-server', { local: true, name: 'My Private Server' })

    expect(taxonomyAppForCard(custom)).toBeUndefined()
    expect(categoryOfCard(custom)).toBe('other')
  })

  it('keeps the static rank order inside a category', () => {
    const money = CONNECTION_TAXONOMY.find(category => category.id === 'money')
    const apps = money?.subgroups.flatMap(subgroup => subgroup.apps) ?? []

    expect(apps.map(entry => entry.name)).toEqual([
      'Stripe',
      'Shopify',
      'Plaid',
      'Cashfree Payments',
      'PayPal',
      'Square',
      'Coinbase Agents',
      'Coinbase Wallet',
      'Ramp',
      'Robinhood',
      'Twelve Data'
    ])
    expect(apps.map(entry => entry.rank)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    expect(connectionRank(card('notion'))).toBe(1)
    expect(connectionRank(card('fireflies'))).toBe(20)
    expect(connectionRank(card('unknown-app'))).toBe(Number.MAX_SAFE_INTEGER)
  })

  it('marks only the no-sign-in apps from Figma', () => {
    const noSignIn = CONNECTION_TAXONOMY_APPS.filter(entry => entry.noSignIn).map(entry => entry.name)

    expect(noSignIn).toEqual([
      'Unreal Engine',
      'Kiwi',
      'Trivago',
      'AllTrails',
      'AWS Knowledge',
      'Context7',
      'DeepWiki',
      'Microsoft Learn',
      'Twilio Docs',
      'Wolfram'
    ])
  })
})

describe('connection call to action', () => {
  it('treats a connected hosted account or an installed local server as connected', () => {
    expect(isConnectedCard(card('notion', { connected: true }))).toBe(true)
    expect(isConnectedCard(card('notion', { installed: true, local: true }))).toBe(true)
    expect(isConnectedCard(card('notion'))).toBe(false)
    expect(cardCtaKind(card('notion', { connected: true }))).toBe('manage')
    expect(cardCtaKind(card('notion', { installed: true, local: true }))).toBe('manage')
  })

  it('adds apps that need no sign-in and connects the rest', () => {
    expect(cardCtaKind(card('notion'))).toBe('connect')
    expect(cardCtaKind(card('kiwi'))).toBe('add')
    expect(cardCtaKind(card('trivago'))).toBe('add')
    expect(cardCtaKind(card('notion', { bundledAuthType: 'none' }))).toBe('add')
    expect(cardCtaKind(card('notion', { bundledAuthType: 'oauth' }))).toBe('connect')
  })
})

describe('suggested cards', () => {
  it('suggests the Simple order and only unconnected apps', () => {
    expect(suggestedCardsForMode(allCards(), 'simple').map(entry => entry.slug)).toEqual([
      'notion',
      'canva',
      'higgsfield',
      'slack'
    ])
  })

  it('suggests the Advanced order', () => {
    expect(suggestedCardsForMode(allCards(), 'advanced').map(entry => entry.slug)).toEqual([
      'cloudflare',
      'vercel',
      'supabase',
      'notion'
    ])
  })

  it('drops a connected suggestion without replacing it', () => {
    expect(suggestedCardsForMode(allCards(['notion']), 'simple').map(entry => entry.slug)).toEqual([
      'canva',
      'higgsfield',
      'slack'
    ])
    expect(suggestedCardsForMode(allCards(['cloudflare', 'vercel']), 'advanced').map(entry => entry.slug)).toEqual([
      'supabase',
      'notion'
    ])
  })

  it('returns nothing when none of the suggested apps are present', () => {
    expect(suggestedCardsForMode([card('stripe'), card('kiwi')], 'simple')).toEqual([])
  })
})

describe('main page grouping', () => {
  it('puts Connected first, then Suggested, then capped category previews', () => {
    const sections = groupCardsForMainPage(
      allCards(['notion', 'linear', 'cloudflare']),
      'simple',
      EMPTY_CONNECTIONS_FILTER
    )

    expect(sections.map(entry => entry.id)).toEqual([
      'connected',
      'suggested',
      'work-productivity',
      'design-content',
      'money',
      'customers-growth',
      'data-analytics',
      'engineering',
      'travel-fitness',
      'reference'
    ])
    expect(section(sections, 'connected')?.cards.map(entry => entry.slug)).toEqual([
      'notion',
      'linear',
      'cloudflare'
    ])
    expect(section(sections, 'connected')?.totalCount).toBe(3)
    expect(section(sections, 'suggested')?.cards.map(entry => entry.slug)).toEqual([
      'canva',
      'higgsfield',
      'slack'
    ])
  })

  it('caps previews, omits already-shown cards and keeps the full total count', () => {
    const sections = groupCardsForMainPage(
      allCards(['notion', 'linear', 'cloudflare']),
      'simple',
      EMPTY_CONNECTIONS_FILTER
    )

    const work = section(sections, 'work-productivity')
    expect(work?.cards.map(entry => entry.slug)).toEqual([
      'airtable',
      'confluence',
      'box',
      'pandadoc',
      'craft',
      'dropbox'
    ])
    expect(work?.totalCount).toBe(20)

    const design = section(sections, 'design-content')
    expect(design?.cards.map(entry => entry.slug)).toEqual(['figma', 'miro', 'memelord', 'comfy-cloud'])
    expect(design?.totalCount).toBe(10)

    const engineering = section(sections, 'engineering')
    expect(engineering?.cards.map(entry => entry.slug)).toEqual([
      'vercel',
      'supabase',
      'neon',
      'railway',
      'turso',
      'netlify'
    ])
    expect(engineering?.totalCount).toBe(24)

    expect(section(sections, 'money')?.cards).toHaveLength(4)
    expect(section(sections, 'money')?.totalCount).toBe(11)
    expect(section(sections, 'travel-fitness')?.cards).toHaveLength(4)
  })

  it('shows every card only once across Connected, Suggested and category previews', () => {
    const sections = groupCardsForMainPage(
      allCards(['notion', 'linear', 'cloudflare']),
      'simple',
      EMPTY_CONNECTIONS_FILTER
    )
    const slugs = sections.flatMap(entry => entry.cards.map(current => current.slug))

    expect(new Set(slugs).size).toBe(slugs.length)
  })

  it('uses the Advanced category order and Advanced suggestions', () => {
    const sections = groupCardsForMainPage(allCards(['cloudflare']), 'advanced', EMPTY_CONNECTIONS_FILTER)

    expect(section(sections, 'suggested')?.cards.map(entry => entry.slug)).toEqual([
      'vercel',
      'supabase',
      'notion'
    ])
    expect(sections.slice(2).map(entry => entry.id)).toEqual([...ADVANCED_CATEGORY_ORDER])
  })

  it('drops Suggested and preview caps when filtering', () => {
    const sections = groupCardsForMainPage(allCards(['notion']), 'simple', {
      ...EMPTY_CONNECTIONS_FILTER,
      query: 'slack'
    })

    expect(sections.map(entry => entry.id)).toEqual(['work-productivity'])
    expect(section(sections, 'work-productivity')?.cards.map(entry => entry.slug)).toEqual(['slack'])
    expect(section(sections, 'work-productivity')?.totalCount).toBe(1)
    expect(section(sections, 'suggested')).toBeUndefined()
    expect(section(sections, 'connected')).toBeUndefined()
  })

  it('groups a filtered page by category without a preview cap', () => {
    const sections = groupCardsForMainPage(allCards(['notion', 'linear', 'cloudflare']), 'simple', {
      ...EMPTY_CONNECTIONS_FILTER,
      status: 'connected'
    })

    expect(sections.map(entry => entry.id)).toEqual(['work-productivity', 'engineering'])
    expect(section(sections, 'work-productivity')?.cards.map(entry => entry.slug)).toEqual(['notion', 'linear'])
    expect(section(sections, 'engineering')?.cards.map(entry => entry.slug)).toEqual(['cloudflare'])
    expect(section(sections, 'suggested')).toBeUndefined()
  })

  it('combines categories and status with AND', () => {
    const sections = groupCardsForMainPage(allCards(['stripe']), 'simple', {
      ...EMPTY_CONNECTIONS_FILTER,
      categories: ['money', 'engineering'],
      status: 'not_connected'
    })

    expect(sections.map(entry => entry.id)).toEqual(['money', 'engineering'])
    expect(section(sections, 'money')?.cards).toHaveLength(10)
    expect(section(sections, 'engineering')?.cards).toHaveLength(24)
    expect(section(sections, 'money')?.cards.some(entry => entry.slug === 'stripe')).toBe(false)
  })

  it('keeps a category searchable when every app is already connected or suggested', () => {
    const connectedOnly = [card('notion', { connected: true }), card('linear', { connected: true })]
    const sections = groupCardsForMainPage(connectedOnly, 'simple', EMPTY_CONNECTIONS_FILTER)
    const work = section(sections, 'work-productivity')

    expect(work?.cards).toEqual([])
    expect(work?.totalCount).toBe(2)
    expect(work?.kind).toBe('category')

    const suggestedOnly = [card('notion'), card('figma')]
    const suggestedSections = groupCardsForMainPage(suggestedOnly, 'simple', EMPTY_CONNECTIONS_FILTER)

    expect(section(suggestedSections, 'suggested')?.cards.map(entry => entry.slug)).toEqual(['notion'])
    expect(section(suggestedSections, 'work-productivity')?.cards).toEqual([])
    expect(section(suggestedSections, 'work-productivity')?.totalCount).toBe(1)

    const searched = groupCardsForMainPage(connectedOnly, 'simple', {
      ...EMPTY_CONNECTIONS_FILTER,
      query: 'linear'
    })

    expect(section(searched, 'work-productivity')?.cards.map(entry => entry.slug)).toEqual(['linear'])
  })

  it('places unmapped local servers in an Other section', () => {
    const custom = card('my-private-server', { installed: false, local: true, name: 'My Private Server' })
    const sections = groupCardsForMainPage([card('stripe'), custom], 'simple', EMPTY_CONNECTIONS_FILTER)

    expect(isConnectedCard(custom)).toBe(false)
    expect(sections.map(entry => entry.id)).toEqual(['money', 'other'])
    expect(section(sections, 'other')?.cards).toEqual([custom])
    expect(section(sections, 'other')?.totalCount).toBe(1)
  })
})

describe('subgroup page grouping', () => {
  it('groups a category page by Figma subgroup order, skipping empty subgroups', () => {
    const groups = groupCardsBySubgroup(allCards(), 'work-productivity', EMPTY_CONNECTIONS_FILTER)

    expect(groups.map(group => group.id)).toEqual([
      'docs-files',
      'communication',
      'projects-tasks',
      'meetings-scheduling'
    ])
    expect(groups.map(group => group.cards.map(entry => entry.slug))).toEqual([
      ['notion', 'airtable', 'confluence', 'box', 'pandadoc', 'craft', 'dropbox'],
      ['discord', 'slack', 'wispr-flow'],
      ['linear', 'jira', 'todoist', 'clickup', 'asana', 'monday'],
      ['granola', 'calendly', 'circleback', 'fireflies']
    ])
  })

  it('sorts every subgroup by static connectionRank', () => {
    const shuffled = [card('fireflies'), card('notion'), card('slack'), card('linear')]
    const groups = groupCardsBySubgroup(shuffled, 'work-productivity', EMPTY_CONNECTIONS_FILTER)

    expect(groups.map(group => group.id)).toEqual([
      'docs-files',
      'communication',
      'projects-tasks',
      'meetings-scheduling'
    ])
    expect(groups.flatMap(group => group.cards.map(entry => entry.slug))).toEqual([
      'notion',
      'slack',
      'linear',
      'fireflies'
    ])
  })

  it('applies the filter before grouping', () => {
    const groups = groupCardsBySubgroup(allCards(['slack']), 'work-productivity', {
      ...EMPTY_CONNECTIONS_FILTER,
      query: 'slack'
    })

    expect(groups.map(group => group.id)).toEqual(['communication'])
    expect(groups[0]?.cards.map(entry => entry.slug)).toEqual(['slack'])
  })

  it('returns nothing when the filter excludes every card in the category', () => {
    const groups = groupCardsBySubgroup(allCards(), 'work-productivity', {
      ...EMPTY_CONNECTIONS_FILTER,
      query: 'stripe'
    })

    expect(groups).toEqual([])
  })
})

describe('category page grouping', () => {
  it('returns every category card in subgroup and rank order', () => {
    const page = groupCardsForCategory(allCards(), 'work-productivity', EMPTY_CONNECTIONS_FILTER)

    expect(page.totalCount).toBe(20)
    expect(page.cards).toHaveLength(20)
    expect(page.cards[0]?.slug).toBe('notion')
    expect(page.cards.at(-1)?.slug).toBe('fireflies')
  })

  it('applies the filter without a preview cap', () => {
    const page = groupCardsForCategory(allCards(), 'work-productivity', {
      ...EMPTY_CONNECTIONS_FILTER,
      query: 'slack'
    })

    expect(page.cards.map(entry => entry.slug)).toEqual(['slack'])
    expect(page.totalCount).toBe(1)
  })

  it('supports the Other category', () => {
    const custom = card('my-private-server', { local: true, name: 'My Private Server' })
    const page = groupCardsForCategory([custom], 'other', EMPTY_CONNECTIONS_FILTER)

    expect(page.id).toBe('other')
    expect(page.cards).toEqual([custom])
    expect(page.totalCount).toBe(1)
  })
})
