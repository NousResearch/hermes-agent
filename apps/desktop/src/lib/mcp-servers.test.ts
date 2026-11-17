import { describe, expect, it } from 'vitest'

import cases from './mcp-enabled-cases.json'
import { getServers, serverEnabled } from './mcp-servers'

describe('getServers', () => {
  it('returns empty when the map is absent or not a map', () => {
    expect(getServers(null)).toEqual({})
    expect(getServers({})).toEqual({})
    expect(getServers({ mcp_servers: null })).toEqual({})
    expect(getServers({ mcp_servers: ['files'] })).toEqual({})
    expect(getServers({ mcp_servers: 'files' })).toEqual({})
  })

  it('passes object entries through untouched', () => {
    const servers = { docs: { url: 'https://example.com/mcp' }, files: { args: ['-y', 'pkg'], command: 'npx' } }

    expect(getServers({ mcp_servers: servers })).toEqual(servers)
  })

  // Field-level junk is the readers' problem, not this filter's — they coerce
  // and tolerate. Pinned so a later tightening of `isEntry` can't start
  // dropping entries that merely carry a bad field.
  it('keeps entries whose fields are junk', () => {
    const servers = { files: { command: 42, enabled: 'yes' } }

    expect(getServers({ mcp_servers: servers })).toEqual(servers)
  })

  // A key left without a value in config.yaml parses as `null`, and every
  // reader (the `enabled` gate, the transport label, the probe) reaches
  // straight into the entry — so one of these used to take the pane down.
  it('drops non-object entries and keeps their siblings', () => {
    const servers = { broken: null, list: ['a'], scalar: 3, working: { command: 'npx' } }

    expect(getServers({ mcp_servers: servers })).toEqual({ working: { command: 'npx' } })
  })
})

// The backend reads the same table (tests/tools/test_mcp_enabled_reader.py), so
// the MCP page and the runtime cannot disagree about whether a server is on.
describe('serverEnabled', () => {
  it.each(cases)('%j', ({ on, ...entry }) => {
    expect(serverEnabled({ command: 'x', ...entry })).toBe(on)
  })
})
