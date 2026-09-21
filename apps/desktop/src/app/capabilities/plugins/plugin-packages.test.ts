import { describe, expect, it } from 'vitest'

import type { PluginRecord } from '@/contrib/plugins-store'
import type { AgentPluginRow } from '@/store/agent-plugins'

import { mergePluginPackages } from './plugin-packages'

const agent = (over: Partial<AgentPluginRow>): AgentPluginRow => ({
  description: '',
  key: over.name ?? 'x',
  name: 'x',
  source: 'git',
  status: 'enabled',
  version: '1.0.0',
  ...over
})

const desktop = (over: Partial<PluginRecord>): PluginRecord => ({
  id: 'x',
  kind: 'disk',
  name: 'x',
  status: 'loaded',
  ...over
})

describe('mergePluginPackages', () => {
  it('shows a unified package as ONE row with both halves, never two rows', () => {
    const rows = mergePluginPackages(
      [desktop({ id: 'media', name: 'Media Studio', packageName: 'hermes-media-studio' })],
      [agent({ name: 'hermes-media-studio', has_desktop_half: true, description: 'Generate media.' })]
    )

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({ kind: 'both', agentMissingInProfile: false, desktopMissing: false })
    expect(rows[0].desktop?.id).toBe('media')
    expect(rows[0].agent?.name).toBe('hermes-media-studio')
  })

  it('a desktop half whose agent half is absent from THIS profile offers the install-here affordance', () => {
    const rows = mergePluginPackages([desktop({ id: 'media', packageName: 'hermes-media-studio' })], [])

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({ kind: 'both', agent: null, agentMissingInProfile: true })
  })

  it('an agent package that declares a desktop half not yet copied to the app is flagged pending', () => {
    const rows = mergePluginPackages([], [agent({ name: 'pkg', has_desktop_half: true })])

    expect(rows[0]).toMatchObject({ kind: 'both', desktop: null, desktopMissing: true })
  })

  it('standalone desktop plugins and agent-only packages keep one empty side; unified rows sort first', () => {
    const rows = mergePluginPackages(
      [desktop({ id: 'bots', name: 'Bots', kind: 'bundled' }), desktop({ id: 'u', packageName: 'unified' })],
      [agent({ name: 'snapcompact' }), agent({ name: 'unified', has_desktop_half: true })]
    )

    expect(rows.map(r => [r.key, r.kind])).toEqual([
      ['unified', 'both'],
      ['snapcompact', 'agent'],
      ['desktop:bots', 'desktop']
    ])
  })
})
