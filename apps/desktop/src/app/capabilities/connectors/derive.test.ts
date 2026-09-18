import { describe, expect, it } from 'vitest'

import {
  cardCategoryOptions,
  deriveCards,
  filterCards,
  groupCards,
  hostedCard,
  inventoryCounts,
  localCard,
  pillCounts
} from './derive'
import { HOSTED, LOCAL, TITLES } from './fixtures'
import type { ConnectorsFilter } from './types'

const cards = deriveCards({ hosted: HOSTED, local: LOCAL, titles: TITLES })
const cardFor = (slug: string) => cards.find(card => card.slug === slug)
const ALL: ConnectorsFilter = { category: null, pill: 'all', query: '', residency: null }

describe('card derivation', () => {
  it('reads the account status, the personal switch and the org rule in that order', () => {
    expect(cardFor('gmail')).toMatchObject({ state: 'connected', stateWord: 'connected' })
    expect(cardFor('notion')).toMatchObject({ state: 'expired', stateWord: 'accessExpired', verb: 'reconnect' })
    expect(cardFor('sentry')).toMatchObject({ state: 'broken', stateWord: 'couldNotConnect', verb: 'tryAgain' })
    expect(cardFor('slack')).toMatchObject({ state: 'available', stateWord: 'available', verb: 'connect' })
    expect(cardFor('shopify')).toMatchObject({ offBy: 'me', state: 'off', verb: 'turnBackOn' })
    expect(cardFor('stripe')).toMatchObject({ offBy: 'org', state: 'off', stateWord: 'offByYourOrganisation' })
  })

  it('leaves an org-locked app no verb, because none of them would work', () => {
    expect(cardFor('stripe')?.verb).toBeUndefined()
  })

  it('prefers the provider’s own sentence for a broken connection', () => {
    expect(cardFor('notion')?.reason).toEqual({
      key: 'reconnect',
      text: 'Authorization expired. Reconnect to continue.'
    })
  })

  it('carries at most one fact', () => {
    expect(cardFor('gmail')?.fact).toEqual({ count: 4, key: 'toolsOff' })
    expect(cardFor('slack')?.fact).toBeUndefined()
  })

  it('says how much of a local server is live', () => {
    expect(cardFor('github')?.fact).toEqual({ count: 31, key: 'toolsSomeOn', on: 29 })
    expect(cardFor('postgres')?.fact).toEqual({ count: 8, key: 'toolsOn' })
  })

  it('says a server is unused instead of counting it, so the word can be read', () => {
    const idle = localCard({ ...LOCAL[1], toolsOn: 8, toolsTotal: 8, unused: true }, 'Postgres')

    // The lane holds one fact. A count here would print over "On, unused" and
    // leave that state word unreachable on every server that reports a total.
    expect(idle.stateWord).toBe('serverOnUnused')
    expect(idle.fact).toBeUndefined()
  })

  it('treats a never-probed server as connecting, not as working', () => {
    expect(localCard({ ...LOCAL[1], status: 'unknown' }, 'Postgres')).toMatchObject({
      state: 'connecting',
      stateWord: 'serverConnecting'
    })
  })

  it('keeps the switch on the local card and off the dialog', () => {
    expect(localCard({ ...LOCAL[1], enabled: false }, 'Postgres')).toMatchObject({
      serverEnabled: false,
      state: 'off',
      stateWord: 'serverOff',
      verb: undefined
    })
  })

  it('reads a connected row with no status as active', () => {
    expect(hostedCard({ ...HOSTED[0], connectionStatus: undefined }, 'Gmail').state).toBe('connected')
  })
})

describe('the one-card merge', () => {
  it('shows one card for an app with both a local server and a hosted version', () => {
    expect(cards.filter(card => card.slug === 'github')).toHaveLength(1)
  })

  it('shows the backing in use and keeps the twin as a pill', () => {
    expect(cardFor('github')).toMatchObject({ hostedTwinAvailable: true, residency: 'local' })
  })

  it('keeps a switched-off server on the page, with its switch and its place', () => {
    const merged = deriveCards({
      hosted: HOSTED,
      local: [{ ...LOCAL[0], enabled: false }],
      titles: TITLES
    })

    const github = merged.filter(card => card.slug === 'github')

    // Pressing the card's own switch must not delete the card: it stays where
    // it was, reading Off, which is what the Turned off pill counts.
    expect(github).toHaveLength(1)
    expect(github[0]).toMatchObject({ residency: 'local', serverEnabled: false, state: 'off', target: LOCAL[0].target })
    expect(groupCards(merged).find(group => group.id === 'local')?.cards).toContain(github[0])
    expect(pillCounts(merged).find(pill => pill.id === 'off')?.count).toBe(3)
  })

  it('offers the hosted twin only when the person could actually fall back on it', () => {
    const lockedTwin = HOSTED.map(row => (row.slug === 'github' ? { ...row, orgLocked: true } : row))

    expect(
      deriveCards({ hosted: lockedTwin, local: LOCAL, titles: TITLES }).find(card => card.slug === 'github')
    ).toMatchObject({ hostedTwinAvailable: false })
  })

  it('leaves a local server with no hosted twin alone', () => {
    expect(cardFor('postgres')).toMatchObject({ hostedTwinAvailable: false, residency: 'local' })
  })
})

describe('grouping', () => {
  const groups = groupCards(cards)
  const group = (id: string) => groups.find(candidate => candidate.id === id)

  it('orders the groups and drops empty ones', () => {
    expect(groups.map(candidate => candidate.id)).toEqual(['connected', 'local', 'available', 'off'])
    expect(groupCards([])).toEqual([])
  })

  it('puts broken connections first inside Connected', () => {
    expect(group('connected')?.cards.map(card => card.slug)).toEqual(['sentry', 'notion', 'gmail'])
  })

  it('keeps every server on this Mac in its own group, whatever its state', () => {
    expect(group('local')?.cards.map(card => card.slug)).toEqual(['linear-local', 'github', 'postgres'])
  })
})

describe('the state pills', () => {
  const pills = pillCounts(cards)

  it('counts each state once', () => {
    expect(pills).toEqual([
      { count: 10, id: 'all' },
      { count: 3, id: 'attention' },
      { count: 3, id: 'connected' },
      { count: 3, id: 'local' },
      { count: 2, id: 'available' },
      { count: 2, id: 'off' }
    ])
  })

  it('omits a pill with a zero count', () => {
    const hostedConnected = cards.filter(card => card.state === 'connected' && card.residency === 'hosted')

    expect(pillCounts(hostedConnected).map(pill => pill.id)).toEqual(['all', 'connected'])
  })

  it('omits every pill when there is nothing to filter', () => {
    expect(pillCounts([])).toEqual([])
  })
})

describe('filtering the directory', () => {
  it('matches the query against the slug and the name', () => {
    expect(filterCards(cards, { ...ALL, query: 'GIT' }).map(card => card.slug)).toEqual(['github'])
    expect(filterCards(cards, { ...ALL, query: 'Postgres' }).map(card => card.slug)).toEqual(['postgres'])
    expect(filterCards(cards, { ...ALL, query: '   ' })).toHaveLength(cards.length)
  })

  it('combines the pill, the query, the residency and the category', () => {
    expect(
      filterCards(cards, { ...ALL, pill: 'attention' })
        .map(card => card.slug)
        .sort()
    ).toEqual(['linear-local', 'notion', 'sentry'])
    expect(filterCards(cards, { ...ALL, residency: 'local' })).toHaveLength(3)
    expect(
      filterCards(cards, { ...ALL, category: 'communication' })
        .map(card => card.slug)
        .sort()
    ).toEqual(['gmail', 'slack'])
    expect(filterCards(cards, { ...ALL, category: 'communication', pill: 'available' })).toHaveLength(1)
  })

  it('counts the category options under every other filter but its own', () => {
    const options = cardCategoryOptions(cards, { ...ALL, category: 'communication', pill: 'available' })

    expect(options).toEqual([
      { count: 1, name: 'communication' },
      { count: 1, name: 'design' }
    ])
  })
})

describe('the inventory line', () => {
  it('counts the two backings separately', () => {
    expect(inventoryCounts(cards)).toEqual({ hosted: 7, local: 3 })
  })
})
