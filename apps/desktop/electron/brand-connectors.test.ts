import assert from 'node:assert/strict'

import { test } from 'vitest'

import { brandMcpServerName, mergeBrandConnectors } from './brand-connectors'

test('a brand key is not the shared test server name', () => {
  const name = brandMcpServerName('foundrly', 'firecrawl')

  assert.equal(name, 'ivx-foundrly-firecrawl')
  assert.notEqual(name, 'firecrawl')
})

test('merge keeps a website key and a desktop-only key on that brand', () => {
  const merged = mergeBrandConnectors(
    'foundrly',
    [
      {
        connectorId: 'postiz',
        label: 'Postiz',
        status: 'connected',
        mcpUrl: '',
        credential: 'postiz-brand-key'
      }
    ],
    { firecrawl: 'fc-brand-only' }
  )
  const postiz = merged.find(row => row.connectorId === 'postiz')
  const firecrawl = merged.find(row => row.connectorId === 'firecrawl')

  assert.equal(postiz?.credential, 'postiz-brand-key')
  assert.equal(postiz?.source, 'web')
  assert.equal(postiz?.mcpUrl, 'https://postiz-mcp.intelli-verse-x.ai/')
  assert.equal(postiz?.hermesName, 'ivx-foundrly-postiz')
  assert.equal(firecrawl?.credential, 'fc-brand-only')
  assert.equal(firecrawl?.hermesName, 'ivx-foundrly-firecrawl')
  assert.equal(firecrawl?.source, 'desktop')
  assert.equal(
    merged.some(row => row.hermesName === 'firecrawl'),
    false
  )
})

test('an empty brand does not invent a test Firecrawl key', () => {
  assert.deepEqual(mergeBrandConnectors('foundrly', [], {}), [])
})

test('inbox studio keeps the account id that came with the brand key', () => {
  const [row] = mergeBrandConnectors(
    'foundrly',
    [
      {
        connectorId: 'chatwoot',
        label: 'Inbox Studio',
        status: 'connected',
        mcpUrl: '',
        credential: 'inbox-brand-key',
        accountId: '233'
      }
    ],
    {}
  )

  assert.equal(row.accountId, '233')
  assert.equal(row.credential, 'inbox-brand-key')
  assert.equal(row.hermesName, 'ivx-foundrly-chatwoot')
  assert.equal(row.mcpUrl, 'https://chatwoot-mcp.intelli-verse-x.ai/')
})

test('a local key fills a website row that has no secret yet', () => {
  const [row] = mergeBrandConnectors(
    'foundrly',
    [{ connectorId: 'firecrawl', label: '', status: 'connected', mcpUrl: '', credential: '' }],
    { firecrawl: 'fc-local' }
  )

  assert.equal(row.credential, 'fc-local')
  assert.equal(row.source, 'desktop')
  assert.equal(row.mcpUrl, 'https://mcp.firecrawl.dev/v2/mcp')
  assert.equal(row.hermesName, 'ivx-foundrly-firecrawl')
})
