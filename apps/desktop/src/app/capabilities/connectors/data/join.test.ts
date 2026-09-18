import type {
  ConnectorAccountRow,
  ConnectorCatalogRow,
  ConnectorRow as ConnectorListRow,
  ConnectorPolicyLayer,
  ConnectorToolsResult
} from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import type { McpCatalogEntry } from '@/hermes'

import { deriveCards } from '../derive'
import { TOOLS } from '../fixtures'

import {
  connectorTitles,
  connectorToolRows,
  joinHostedConnectors,
  joinLocalServers,
  memberRevision,
  readPolicy,
  toolsFreshness
} from './join'

const layer = (kind: ConnectorPolicyLayer['kind'], body: ConnectorPolicyLayer['body'], revision = `${kind}-1`) => ({
  body,
  kind,
  revision
})

const catalogRow = (slug: string, name: string, category = 'Work'): ConnectorCatalogRow => ({
  category,
  description: `${name} for this account.`,
  name,
  slug
})

const listRow = (slug: string, extra: Partial<ConnectorListRow> = {}): ConnectorListRow => ({
  connected: false,
  connector: slug,
  enabled: true,
  ...extra
})

const account = (slug: string, extra: Partial<ConnectorAccountRow> = {}): ConnectorAccountRow => ({
  active: true,
  connection_id: `conn-${slug}`,
  connector: slug,
  created_at: '2026-02-01T10:00:00Z',
  label: `${slug}@example.com`,
  status: 'active',
  updated_at: '2026-02-01T10:00:00Z',
  ...extra
})

// Only `name`, `connector` and `description` are read; the rest of the catalog
// entry is install machinery this join never touches.
const bundled = (name: string, connector?: string): McpCatalogEntry =>
  ({ connector, description: `${name} on this Mac`, name }) as McpCatalogEntry

describe('policy layers', () => {
  it('separates the layer the person writes from the layers that lock them', () => {
    const policy = readPolicy([
      layer('member', { disabled_connectors: ['shopify'], mode: 'deny', tools: {} }, 'member-7'),
      layer('role', { disabled_connectors: ['stripe'], mode: 'deny', tools: {} })
    ])

    const rows = joinHostedConnectors({
      accounts: [],
      catalog: [catalogRow('shopify', 'Shopify'), catalogRow('stripe', 'Stripe')],
      list: [listRow('shopify'), listRow('stripe')],
      policy
    })

    expect(rows.find(row => row.slug === 'shopify')).toMatchObject({ enabled: false, orgLocked: false })
    expect(rows.find(row => row.slug === 'stripe')).toMatchObject({ enabled: true, orgLocked: true })
    expect(memberRevision(policy)).toBe('member-7')
  })

  it('reads an allow layer as a list of the only connectors it permits', () => {
    const policy = readPolicy([layer('org', { connectors: ['gmail'], mode: 'allow', tools: {} })])

    const rows = joinHostedConnectors({
      accounts: [],
      catalog: [catalogRow('gmail', 'Gmail'), catalogRow('slack', 'Slack')],
      list: [],
      policy
    })

    expect(rows.find(row => row.slug === 'gmail')?.orgLocked).toBe(false)
    expect(rows.find(row => row.slug === 'slack')?.orgLocked).toBe(true)
  })
})

describe('tool rows', () => {
  it('turns off what the member named and locks what a role layer named', () => {
    const policy = readPolicy([
      layer('member', { disabled_connectors: [], mode: 'deny', tools: { linear: ['LINEAR_ADD_COMMENT'] } }),
      layer('role', { disabled_connectors: [], mode: 'deny', tools: { linear: ['LINEAR_DELETE_PROJECT'] } })
    ])

    const rows = connectorToolRows(policy, 'linear', TOOLS)
    const row = (slug: string) => rows.find(candidate => candidate.slug === slug)

    expect(row('LINEAR_ADD_COMMENT')).toMatchObject({ lockedBy: null, on: false })
    expect(row('LINEAR_DELETE_PROJECT')).toMatchObject({ lockedBy: 'org', on: false })
    expect(row('LINEAR_LIST_ISSUES')).toMatchObject({ lockedBy: null, on: true })
  })

  it('locks every tool carrying a tag the layer disabled', () => {
    const policy = readPolicy([
      layer('role', { disabled_connectors: [], mode: 'deny', tags: { disable: ['deleteHint'] }, tools: {} })
    ])

    const locked = connectorToolRows(policy, 'linear', TOOLS)
      .filter(row => row.lockedBy === 'org')
      .map(row => row.slug)

    expect(locked).toEqual(['LINEAR_ARCHIVE_ISSUE', 'LINEAR_DELETE_COMMENT', 'LINEAR_DELETE_PROJECT'])
  })

  it('locks every tool carrying NONE of the tags the layer enabled', () => {
    const policy = readPolicy([
      layer('role', { disabled_connectors: [], mode: 'deny', tags: { enable: ['readOnlyHint'] }, tools: {} })
    ])

    const open = connectorToolRows(policy, 'linear', TOOLS)
      .filter(row => row.lockedBy === null)
      .map(row => row.slug)

    expect(open).toEqual([
      'LINEAR_LIST_ISSUES',
      'LINEAR_GET_ISSUE',
      'LINEAR_SEARCH_ISSUES',
      'LINEAR_LIST_TEAMS'
    ])
  })
})

describe('hosted rows', () => {
  const policy = readPolicy([layer('member', { disabled_connectors: [], mode: 'deny', tools: { gmail: ['A', 'B'] } })])

  const rows = joinHostedConnectors({
    accounts: [account('gmail'), account('notion', { status: 'pending' })],
    catalog: [catalogRow('gmail', 'Gmail'), catalogRow('notion', 'Notion')],
    list: [listRow('gmail', { connected: true }), listRow('slack')],
    policy
  })

  const row = (slug: string) => rows.find(candidate => candidate.slug === slug)

  it('takes the account for the label, the date and the status', () => {
    expect(row('gmail')).toMatchObject({
      accountLabel: 'gmail@example.com',
      connectedAt: '2026-02-01T10:00:00Z',
      connectionStatus: 'active',
      connected: true
    })
  })

  it('treats an account that is still authorizing as connected, so the card can say so', () => {
    expect(row('notion')).toMatchObject({ connected: true, connectionStatus: 'pending' })
  })

  it('carries a connector the list knows but the catalog does not', () => {
    expect(row('slack')).toMatchObject({ connected: false, inCatalog: false })
  })

  it('counts the member rule as the tools that are off', () => {
    expect(row('gmail')?.toolsOff).toBe(2)
    expect(row('notion')?.toolsOff).toBeUndefined()
  })

  it('prefers the catalog name over the built-in title table', () => {
    expect(connectorTitles({ catalog: [catalogRow('gmail', 'Google Mail')], list: [] }).gmail).toBe('Google Mail')
    expect(connectorTitles({ catalog: [], list: [listRow('googlecalendar')] }).googlecalendar).toBe('Google Calendar')
  })
})

describe('local servers', () => {
  const servers = {
    notion: { command: 'npx', args: ['-y', 'notion-mcp'] },
    scratch: { url: 'https://example.invalid/mcp' }
  }

  it('reads the hosted slug off the bundled manifest and nowhere else', () => {
    const local = joinLocalServers({
      catalog: [bundled('notion', 'notion')],
      servers,
      status: { notion: 'ok', scratch: 'error' }
    })

    expect(local.find(server => server.name === 'notion')?.hostedSlug).toBe('notion')
    expect(local.find(server => server.name === 'scratch')?.hostedSlug).toBeUndefined()
  })

  it('reads the transport and its target off the entry', () => {
    const local = joinLocalServers({ catalog: [], servers, status: {} })

    expect(local.find(server => server.name === 'notion')).toMatchObject({
      target: 'npx -y notion-mcp',
      transport: 'program'
    })
    expect(local.find(server => server.name === 'scratch')).toMatchObject({
      target: 'https://example.invalid/mcp',
      transport: 'url'
    })
  })

  it('collapses a local server and its hosted twin into one card', () => {
    const cards = deriveCards({
      hosted: joinHostedConnectors({
        accounts: [account('notion')],
        catalog: [catalogRow('notion', 'Notion')],
        list: [listRow('notion', { connected: true })],
        policy: readPolicy([])
      }),
      local: joinLocalServers({
        catalog: [bundled('notion', 'notion')],
        servers: { notion: servers.notion },
        status: { notion: 'ok' }
      }),
      titles: { notion: 'Notion' }
    })

    expect(cards).toHaveLength(1)
    expect(cards[0]).toMatchObject({ hostedTwinAvailable: true, residency: 'local', slug: 'notion' })
  })
})

describe('tool freshness', () => {
  it('converts the wire timestamp from seconds to the milliseconds the cue subtracts', () => {
    const result = {
      connector: 'linear',
      etag: 'x',
      fetched_at: 1_772_000_000,
      source: 'cache',
      stale: false,
      tools: [],
      toolkit_version: '1'
    } satisfies ConnectorToolsResult

    expect(toolsFreshness(result)).toEqual({ fetchedAt: 1_772_000_000_000, source: 'cache', stale: false })
  })
})
