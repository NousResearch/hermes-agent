import { describe, expect, it } from 'vitest'

import { connectorDisplayName, connectorPrimaryActionKind } from './mcp-catalog'

const entry = (overrides = {}) => ({
  name: 'github-enterprise',
  description: 'GitHub connector',
  source: 'https://example.com',
  transport: 'http',
  auth_type: 'oauth',
  required_env: [],
  command: null,
  args: [],
  url: 'https://example.com/mcp',
  install_url: null,
  install_ref: null,
  bootstrap: [],
  default_enabled: null,
  post_install: '',
  display_name: '',
  category: '',
  icon: '',
  tags: [],
  capabilities: [],
  setup_steps: [],
  danger_notes: [],
  needs_install: false,
  installed: false,
  enabled: false,
  ...overrides
})

describe('connectorDisplayName', () => {
  it('uses manifest display_name when present', () => {
    expect(connectorDisplayName(entry({ display_name: 'GitHub Enterprise' }))).toBe('GitHub Enterprise')
  })

  it('falls back to a readable name for legacy manifests', () => {
    expect(connectorDisplayName(entry())).toBe('Github Enterprise')
  })
})

describe('connectorPrimaryActionKind', () => {
  it('treats uninstalled OAuth HTTP catalog entries as connect actions', () => {
    expect(connectorPrimaryActionKind(entry())).toBe('connect')
  })

  it('keeps local stdio catalog entries as install actions', () => {
    expect(connectorPrimaryActionKind(entry({ transport: 'stdio', auth_type: 'none', command: 'npx', url: null }))).toBe(
      'install'
    )
  })

  it('returns installed once the entry is already installed', () => {
    expect(connectorPrimaryActionKind(entry({ installed: true, enabled: true }))).toBe('installed')
  })
})
