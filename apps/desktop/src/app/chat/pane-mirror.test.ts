import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { contributesToWorkspace } from '@/components/pane-shell/workspace-scope'
import { registry } from '@/contrib/registry'

import { paneMirror } from './pane-mirror'

interface Tile {
  id: string
  owner?: string
}

const cleanupSources: Array<ReturnType<typeof atom<Tile[]>>> = []
let sequence = 0

function setup(options: {
  workspaceMode?: 'sessions' | 'bots' | ((tile: Tile) => 'sessions' | 'bots' | undefined)
  workspaceOwnerKey?: string | ((tile: Tile) => string | undefined)
}) {
  const source = atom<Tile[]>([])
  const prefix = `pane-mirror-scope-${sequence++}`
  cleanupSources.push(source)

  paneMirror<Tile>({
    source,
    key: tile => tile.id,
    prefix,
    minWidth: '10rem',
    title: key => key,
    render: () => null,
    close: () => undefined,
    ...options
  })()

  return {
    source,
    contribution: (id: string) => registry.getArea('panes').find(entry => entry.id === `${prefix}:${id}`)
  }
}

afterEach(() => {
  for (const source of cleanupSources.splice(0)) {
    source.set([])
  }
})

describe('paneMirror workspace scope', () => {
  it('forwards a static workspace mode', () => {
    const mirror = setup({ workspaceMode: 'sessions' })
    mirror.source.set([{ id: 'one' }])

    expect(mirror.contribution('one')).toMatchObject({
      workspaceMode: 'sessions',
      workspaceOwnerKey: undefined
    })
  })

  it('resolves owner callbacks per tile and refreshes an unchanged title', () => {
    const mirror = setup({
      workspaceMode: 'bots',
      workspaceOwnerKey: tile => tile.owner
    })

    mirror.source.set([{ id: 'one', owner: 'connection-a::default' }])
    expect(mirror.contribution('one')?.workspaceOwnerKey).toBe('connection-a::default')

    mirror.source.set([{ id: 'one', owner: 'connection-b::default' }])
    expect(mirror.contribution('one')?.workspaceOwnerKey).toBe('connection-b::default')
  })

  it('leaves existing callers unscoped when options are omitted', () => {
    const mirror = setup({})
    mirror.source.set([{ id: 'one' }])

    expect(mirror.contribution('one')).toMatchObject({
      workspaceMode: undefined,
      workspaceOwnerKey: undefined
    })
  })

  it('keeps an unscoped Browser tile visible in Bot Mode', () => {
    const mirror = setup({})
    mirror.source.set([{ id: 'url:browser' }])

    const pane = mirror.contribution('url:browser')

    expect(contributesToWorkspace(pane, 'sessions')).toBe(true)
    expect(contributesToWorkspace(pane, 'bots', 'bot:connection-a::default')).toBe(true)
  })

  it('hides a Sessions-only Browser tile from Bot Mode', () => {
    const mirror = setup({ workspaceMode: 'sessions' })
    mirror.source.set([{ id: 'url:browser' }])

    const pane = mirror.contribution('url:browser')

    expect(contributesToWorkspace(pane, 'sessions')).toBe(true)
    expect(contributesToWorkspace(pane, 'bots', 'bot:connection-a::default')).toBe(false)
  })
})
