import { describe, expect, it } from 'vitest'

import { UNCATEGORISED } from '../derive-tools'
import type { ConnectorCardModel, ToolRowModel } from '../types'

import {
  activeConnectionsChipCount,
  activeToolsChipCount,
  CONNECTION_TOOL_FACETS,
  type ConnectionToolsFilterState,
  countByCategory,
  countByStatus,
  EMPTY_CONNECTION_TOOLS_FILTER,
  EMPTY_CONNECTIONS_FILTER,
  filterConnectionCards,
  filterConnectionTools,
  groupToolsByArea,
  toggleCategoryFilter,
  toggleToolAreaFilter,
  toggleToolFacetFilter,
  type ToolFacet
} from './filter-logic'
import { SIMPLE_CATEGORY_ORDER } from './taxonomy'

function hosted(
  slug: string,
  name: string,
  connected: boolean,
  description?: string
): ConnectorCardModel {
  const state = connected ? 'connected' : 'available'

  return {
    description,
    inCatalog: true,
    name,
    residency: 'hosted',
    slug,
    state,
    stateWord: connected ? 'connected' : 'available',
    ways: {
      hosted: { accounts: [], connected, retiredAccounts: [], state },
      local: null
    }
  }
}

function local(slug: string, name: string): ConnectorCardModel {
  return {
    inCatalog: false,
    name,
    residency: 'local',
    slug,
    state: 'connected',
    stateWord: 'serverOn',
    ways: {
      hosted: null,
      local: { installed: true, state: 'connected' }
    }
  }
}

const NOTION = hosted('notion', 'Notion', true, 'Pages and databases')
const STRIPE = hosted('stripe_mcp', 'Stripe', false, 'Online payments')
const FIGMA = hosted('figma', 'Figma', false, 'Design files')
const CUSTOM = local('my-private-server', 'My Private Server')
const CONNECTIONS = [NOTION, STRIPE, FIGMA]

function tool(slug: string, overrides: Partial<ToolRowModel> = {}): ToolRowModel {
  return {
    categories: ['notes'],
    deprecated: false,
    description: '',
    facet: 'read',
    hints: [],
    lockedBy: null,
    name: slug,
    on: true,
    slug,
    ...overrides
  }
}

describe('connections filter', () => {
  it('shows every card for the empty filter, in input order', () => {
    expect(filterConnectionCards(CONNECTIONS, EMPTY_CONNECTIONS_FILTER).map(entry => entry.slug)).toEqual([
      'notion',
      'stripe_mcp',
      'figma'
    ])
  })

  it('filters by status', () => {
    expect(
      filterConnectionCards(CONNECTIONS, { ...EMPTY_CONNECTIONS_FILTER, status: 'connected' }).map(
        entry => entry.slug
      )
    ).toEqual(['notion'])
    expect(
      filterConnectionCards(CONNECTIONS, { ...EMPTY_CONNECTIONS_FILTER, status: 'not_connected' }).map(
        entry => entry.slug
      )
    ).toEqual(['stripe_mcp', 'figma'])
  })

  it('normalizes the search across name, slug and description', () => {
    expect(
      filterConnectionCards(CONNECTIONS, { ...EMPTY_CONNECTIONS_FILTER, query: '  NOTION ' }).map(
        entry => entry.slug
      )
    ).toEqual(['notion'])
    expect(
      filterConnectionCards(CONNECTIONS, { ...EMPTY_CONNECTIONS_FILTER, query: 'stripe mcp' }).map(
        entry => entry.slug
      )
    ).toEqual(['stripe_mcp'])
    expect(
      filterConnectionCards(CONNECTIONS, { ...EMPTY_CONNECTIONS_FILTER, query: 'payments' }).map(
        entry => entry.slug
      )
    ).toEqual(['stripe_mcp'])
    expect(filterConnectionCards(CONNECTIONS, { ...EMPTY_CONNECTIONS_FILTER, query: 'nothing' })).toEqual([])
  })

  it('ORs categories inside the group', () => {
    expect(
      filterConnectionCards(CONNECTIONS, {
        ...EMPTY_CONNECTIONS_FILTER,
        categories: ['money', 'design-content']
      }).map(entry => entry.slug)
    ).toEqual(['stripe_mcp', 'figma'])
  })

  it('ANDs categories with status and search', () => {
    expect(
      filterConnectionCards(CONNECTIONS, {
        ...EMPTY_CONNECTIONS_FILTER,
        categories: ['money'],
        query: 'stripe',
        status: 'not_connected'
      }).map(entry => entry.slug)
    ).toEqual(['stripe_mcp'])

    expect(
      filterConnectionCards(CONNECTIONS, {
        ...EMPTY_CONNECTIONS_FILTER,
        categories: ['money'],
        status: 'connected'
      })
    ).toEqual([])

    expect(
      filterConnectionCards(CONNECTIONS, {
        ...EMPTY_CONNECTIONS_FILTER,
        categories: ['design-content'],
        query: 'stripe'
      })
    ).toEqual([])
  })

  it('never matches an unmapped local server by category', () => {
    expect(
      filterConnectionCards([CUSTOM], {
        ...EMPTY_CONNECTIONS_FILTER,
        categories: ['engineering']
      })
    ).toEqual([])
    expect(filterConnectionCards([CUSTOM], EMPTY_CONNECTIONS_FILTER)).toEqual([CUSTOM])
  })

  it('counts chips without counting the search', () => {
    expect(activeConnectionsChipCount(EMPTY_CONNECTIONS_FILTER)).toBe(0)
    expect(
      activeConnectionsChipCount({
        ...EMPTY_CONNECTIONS_FILTER,
        categories: ['money', 'engineering'],
        query: 'stripe',
        status: 'connected'
      })
    ).toBe(3)
  })

  it('counts statuses and categories', () => {
    expect(countByStatus([NOTION, STRIPE, FIGMA, CUSTOM])).toEqual({
      any: 4,
      connected: 2,
      not_connected: 2
    })
    expect(countByCategory([NOTION, STRIPE, FIGMA, CUSTOM])).toEqual({
      'customers-growth': 0,
      'data-analytics': 0,
      'design-content': 1,
      engineering: 0,
      money: 1,
      other: 1,
      reference: 0,
      'travel-fitness': 0,
      'work-productivity': 1
    })
  })

  it('adds and removes category chips in the available order', () => {
    const available = SIMPLE_CATEGORY_ORDER

    expect(toggleCategoryFilter([], 'money', available)).toEqual(['money'])
    expect(toggleCategoryFilter(['money'], 'engineering', available)).toEqual(['money', 'engineering'])
    expect(toggleCategoryFilter(['money', 'engineering'], 'money', available)).toEqual(['engineering'])
  })

  it('snaps back to All when every category is selected', () => {
    const available = SIMPLE_CATEGORY_ORDER
    const allButLast = available.slice(0, -1)
    const last = available.at(-1)

    expect(last).toBeDefined()
    expect(toggleCategoryFilter(allButLast, last!, available)).toEqual([])
  })

  it('does not mutate the selected categories', () => {
    const selected: readonly ['money'] = ['money']

    expect(toggleCategoryFilter(selected, 'engineering', SIMPLE_CATEGORY_ORDER)).toEqual([
      'money',
      'engineering'
    ])
    expect(selected).toEqual(['money'])
  })
})

describe('tools filter', () => {
  const SEARCH = tool('search-pages', { categories: ['search'] })
  const WRITE = tool('create-page', { categories: ['notes'], facet: 'write', on: false })
  const DESTRUCTIVE = tool('delete-page', { categories: ['notes'], facet: 'destructive' })
  const UNCLASSIFIED = tool('mystery', { categories: [], facet: 'unclassified' })
  const TOOLS = [SEARCH, WRITE, DESTRUCTIVE, UNCLASSIFIED]

  const filter = (overrides: Partial<ConnectionToolsFilterState>) =>
    filterConnectionTools(TOOLS, { ...EMPTY_CONNECTION_TOOLS_FILTER, ...overrides })

  it('shows every tool for the empty filter', () => {
    expect(filter({}).map(entry => entry.slug)).toEqual([
      'search-pages',
      'create-page',
      'delete-page',
      'mystery'
    ])
  })

  it('filters by status', () => {
    expect(filter({ status: 'on' }).map(entry => entry.slug)).toEqual([
      'search-pages',
      'delete-page',
      'mystery'
    ])
    expect(filter({ status: 'off' }).map(entry => entry.slug)).toEqual(['create-page'])
  })

  it('ORs facets inside the group and never matches unclassified', () => {
    expect(filter({ facets: ['read', 'write'] }).map(entry => entry.slug)).toEqual([
      'search-pages',
      'create-page'
    ])
    expect(filter({ facets: ['destructive'] }).map(entry => entry.slug)).toEqual(['delete-page'])
    expect(filter({ facets: ['read'] })).not.toContain(UNCLASSIFIED)
  })

  it('ORs areas inside the group, including uncategorised', () => {
    expect(filter({ areas: ['search', 'notes'] }).map(entry => entry.slug)).toEqual([
      'search-pages',
      'create-page',
      'delete-page'
    ])
    expect(filter({ areas: [UNCATEGORISED] }).map(entry => entry.slug)).toEqual(['mystery'])
  })

  it('ANDs facets, status, areas and search', () => {
    expect(
      filter({
        areas: ['notes'],
        facets: ['write'],
        query: 'create',
        status: 'off'
      }).map(entry => entry.slug)
    ).toEqual(['create-page'])

    expect(filter({ areas: ['search'], facets: ['write'] })).toEqual([])
    expect(filter({ facets: ['write'], status: 'on' })).toEqual([])
  })

  it('searches names, slugs, descriptions and areas', () => {
    expect(filter({ query: 'CREATE PAGE' }).map(entry => entry.slug)).toEqual(['create-page'])
    expect(filter({ query: 'search-pages' }).map(entry => entry.slug)).toEqual(['search-pages'])
    expect(filter({ query: 'notes' }).map(entry => entry.slug)).toEqual(['create-page', 'delete-page'])
    expect(filter({ query: 'unknown' })).toEqual([])
  })

  it('counts chips without counting the search', () => {
    expect(activeToolsChipCount(EMPTY_CONNECTION_TOOLS_FILTER)).toBe(0)
    expect(
      activeToolsChipCount({
        areas: ['notes'],
        facets: ['read', 'write'],
        query: 'page',
        status: 'off'
      })
    ).toBe(4)
  })

  it('adds and removes facet chips in the available order', () => {
    expect(toggleToolFacetFilter([], 'write', CONNECTION_TOOL_FACETS)).toEqual(['write'])
    expect(toggleToolFacetFilter(['write'], 'read', CONNECTION_TOOL_FACETS)).toEqual(['read', 'write'])
    expect(toggleToolFacetFilter(['read', 'write'], 'write', CONNECTION_TOOL_FACETS)).toEqual(['read'])
  })

  it('snaps facets back to All when every facet is selected', () => {
    const selected: readonly ToolFacet[] = ['read', 'write']

    expect(toggleToolFacetFilter(selected, 'destructive', CONNECTION_TOOL_FACETS)).toEqual([])
  })

  it('adds, removes and snaps area chips', () => {
    const available = ['notes', 'search', UNCATEGORISED]

    expect(toggleToolAreaFilter([], 'notes', available)).toEqual(['notes'])
    expect(toggleToolAreaFilter(['notes'], 'search', available)).toEqual(['notes', 'search'])
    expect(toggleToolAreaFilter(['notes', 'search'], UNCATEGORISED, available)).toEqual([])
    expect(toggleToolAreaFilter(['notes', 'search', UNCATEGORISED], 'notes', available)).toEqual([
      'search',
      UNCATEGORISED
    ])
  })
})

describe('tools grouping', () => {
  it('groups by area, busiest first, uncategorised last', () => {
    const shared = tool('shared', { categories: ['notes', 'search'] })
    const searchOnly = tool('search-only', { categories: ['search'] })
    const untagged = tool('untagged', { categories: [] })
    const groups = groupToolsByArea([shared, searchOnly, untagged])

    expect(groups.map(group => group.area)).toEqual(['search', 'notes', UNCATEGORISED])
    expect(groups[0]?.tools.map(entry => entry.slug)).toEqual(['shared', 'search-only'])
    expect(groups[1]?.tools.map(entry => entry.slug)).toEqual(['shared'])
    expect(groups[2]?.tools.map(entry => entry.slug)).toEqual(['untagged'])
  })

  it('breaks ties alphabetically', () => {
    const zebra = tool('zebra', { categories: ['zebra'] })
    const alpha = tool('alpha', { categories: ['alpha'] })
    const groups = groupToolsByArea([zebra, alpha])

    expect(groups.map(group => group.area)).toEqual(['alpha', 'zebra'])
  })

  it('groups a filtered tool list', () => {
    const shared = tool('shared', { categories: ['notes', 'search'] })
    const write = tool('write', { categories: ['notes'], facet: 'write' })
    const filtered = filterConnectionTools([shared, write], {
      ...EMPTY_CONNECTION_TOOLS_FILTER,
      facets: ['read']
    })

    expect(groupToolsByArea(filtered).map(group => group.area)).toEqual(['notes', 'search'])
  })
})
