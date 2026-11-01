import type { McpRuntimeStatus, McpServerRuntimeRow, McpServerSummary } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { joinLocalServers, pluginServerRows } from './data/join'
import { deriveCards } from './derive'

// The card a stdio server with no probe renders as: stateWord picks the copy
// key the row card and dialog paint, state picks the dot colour.
function localCard(server: Parameters<typeof joinLocalServers>[0]['servers'][string], status: 'ok' | 'unknown') {
  const [row] = joinLocalServers({
    catalog: [],
    servers: { aihub: server },
    status: { aihub: status }
  })

  return { card: deriveCards({ hosted: [], local: [row] })[0], row }
}

describe('unprobed stdio servers read as on demand, not connecting', () => {
  it('shows an enabled stdio server nobody has probed as "On demand"', () => {
    const { card, row } = localCard(
      { args: ['-jar', '/opt/mcp-server.jar'], command: '/usr/bin/java', enabled: true },
      'unknown'
    )

    // Background health checks never probe stdio servers (probing would spawn
    // the user's command), so 'unknown' here is permanent, not progress.
    expect(row.onDemand).toBe(true)
    expect(card.stateWord).toBe('serverOnDemand')
    expect(card.state).toBe('unknown')
  })

  it('keeps "Connecting…" for url-shaped servers the background sweep is about to probe', () => {
    const { card, row } = localCard({ enabled: true, url: 'https://mcp.example/sse' }, 'unknown')

    expect(row.onDemand).toBe(false)
    expect(card.stateWord).toBe('serverConnecting')
    expect(card.state).toBe('connecting')
  })

  it('keeps the probed state once a stdio server has a real answer', () => {
    const { card, row } = localCard({ command: '/usr/bin/java', enabled: true }, 'ok')

    expect(row.onDemand).toBe(false)
    expect(card.stateWord).toBe('serverOn')
    expect(card.state).toBe('connected')
  })
})

describe('plugin servers without a live runtime row read the same way', () => {
  const server = (overrides: Partial<McpServerSummary> = {}): McpServerSummary => ({
    args: ['server.js'],
    command: 'node',
    enabled: true,
    env: [],
    name: 'plug__tools',
    plugin: 'plug',
    source: 'plugin',
    tools: null,
    transport: 'stdio',
    url: null,
    ...overrides
  })

  const runtime = (status: McpRuntimeStatus): McpServerRuntimeRow[] => [
    {
      connected: false,
      disabled: false,
      name: 'plug__tools',
      source: 'plugin',
      status,
      tools: 0,
      transport: 'stdio'
    }
  ]

  it('marks a lazy stdio plugin server on demand', () => {
    const [row] = pluginServerRows({ runtime: runtime('lazy'), servers: [server()] })

    expect(row.onDemand).toBe(true)
  })

  it('does not mark a plugin server the runtime is actually connecting', () => {
    const [row] = pluginServerRows({ runtime: runtime('connecting'), servers: [server()] })

    expect(row.onDemand).toBe(false)
    expect(row.status).toBe('probing')
  })
})
