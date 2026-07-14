import { describe, expect, it } from 'vitest'

import { connectorDisplayName, connectorIdentityKey, connectorPrimaryActionKind, connectorSetupSummary } from './mcp-catalog'

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

describe('connectorIdentityKey', () => {
  it('prefers the curated icon key over the config name', () => {
    expect(connectorIdentityKey(entry({ icon: 'unreal-engine', name: 'ue-local' }))).toBe('unreal-engine')
  })

  it('falls back to the entry name for legacy manifests', () => {
    expect(connectorIdentityKey(entry({ icon: '' }))).toBe('github-enterprise')
  })
})

describe('connectorPrimaryActionKind', () => {
  it('treats uninstalled OAuth HTTP catalog entries as connect actions', () => {
    expect(connectorPrimaryActionKind(entry())).toBe('connect')
  })

  it('treats uninstalled OAuth SSE catalog entries as connect actions', () => {
    expect(connectorPrimaryActionKind(entry({ transport: 'sse' }))).toBe('connect')
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

describe('connectorSetupSummary', () => {
  it('describes OAuth HTTP connectors as browser sign-in flows', () => {
    expect(connectorSetupSummary(entry({ setup_steps: ['Sign in', 'Pick tools'] }))).toBe(
      '2 setup steps · Browser OAuth'
    )
  })

  it('describes api-key connectors as credential setup flows', () => {
    expect(
      connectorSetupSummary(
        entry({ auth_type: 'api_key', required_env: [{ name: 'API_KEY', prompt: 'API key', required: true }] })
      )
    ).toBe('1 setup step · Requires credentials')
  })

  it('describes local build connectors without setup steps', () => {
    expect(connectorSetupSummary(entry({ auth_type: 'none', needs_install: true, setup_steps: [] }))).toBe('Local build')
  })
})
