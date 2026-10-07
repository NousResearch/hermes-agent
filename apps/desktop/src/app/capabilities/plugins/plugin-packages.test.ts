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
      [desktop({ id: 'media', name: 'Media Studio', packageName: 'rabbit-media-studio' })],
      [agent({ name: 'rabbit-media-studio', has_desktop_half: true, description: 'Generate media.' })]
    )

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({ kind: 'both', agentMissingInProfile: false, desktopMissing: false })
    expect(rows[0].desktop?.id).toBe('media')
    expect(rows[0].agent?.name).toBe('rabbit-media-studio')
  })

  it('a desktop half whose agent half is absent from THIS profile offers the install-here affordance', () => {
    const rows = mergePluginPackages([desktop({ id: 'media', packageName: 'rabbit-media-studio' })], [])

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({ kind: 'both', agent: null, agentMissingInProfile: true })
  })

  it('an agent package that declares a desktop half not yet copied to the app is flagged pending', () => {
    const rows = mergePluginPackages([], [agent({ name: 'pkg', has_desktop_half: true })])

    expect(rows[0]).toMatchObject({ kind: 'both', desktop: null, desktopMissing: true })
  })

  // A catalog install used to land the desktop half at
  // desktop-plugins/<name>/plugin.js with no .rabbit-package.json, and the row
  // sat on "copying…" beside a second, already-enabled desktop row. The join is
  // the marker, not the folder name: Electron stamps it on install and adopts
  // marker-less copies on reconcile (see desktop-plugins-root.ts), so the page
  // pairs on evidence instead of guessing from a path.
  it('pairs a desktop half with its agent row through the package marker', () => {
    const rows = mergePluginPackages(
      [
        desktop({
          id: 'rabbit-talk',
          name: 'Rabbit Talk',
          description: 'GPT-Live subscription or explicit API voice, with Rabbit task delegation.',
          packageName: 'rabbit-talk',
          file: '/Users/me/.rabbit/desktop-plugins/rabbit-talk/plugin.js'
        })
      ],
      [agent({ name: 'rabbit-talk', has_desktop_half: true, version: '0.21.0' })]
    )

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({
      key: 'rabbit-talk',
      name: 'Rabbit Talk',
      kind: 'both',
      desktopMissing: false,
      agentMissingInProfile: false
    })
    expect(rows[0].desktop?.id).toBe('rabbit-talk')
    expect(rows[0].agent?.name).toBe('rabbit-talk')
  })

  it('keeps an unmarked app-root copy its own row rather than guessing from the folder name', () => {
    const rows = mergePluginPackages(
      [
        desktop({
          id: 'rabbit-talk',
          name: 'Rabbit Talk',
          file: '/Users/me/.rabbit/desktop-plugins/rabbit-talk/plugin.js'
        })
      ],
      [agent({ name: 'rabbit-talk', has_desktop_half: true })]
    )

    expect(rows.map(row => [row.key, row.kind])).toEqual([
      ['rabbit-talk', 'both'],
      ['desktop:rabbit-talk', 'desktop']
    ])
  })

  it('leaves a same-named standalone desktop plugin alone when the agent package has no desktop half', () => {
    const rows = mergePluginPackages(
      [desktop({ id: 'clock', name: 'Clock', file: '/Users/me/.rabbit/desktop-plugins/clock/plugin.js' })],
      [agent({ name: 'clock' })]
    )

    expect(rows.map(row => [row.key, row.kind])).toEqual([
      ['clock', 'agent'],
      ['desktop:clock', 'desktop']
    ])
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
