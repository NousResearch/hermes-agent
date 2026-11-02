import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  backendsInterruptedByUpdate,
  normalizeWorkOwners,
  type OwnedActiveWork,
  type UpdateBackendKind,
  updateWorkSources,
  workInterruptedByUpdate,
  type WorkOwner
} from './update-work-owners'

const KINDS: Record<string, UpdateBackendKind> = { cloud: 'cloud', laptop: 'local', lab: 'ssh', vps: 'remote' }

const kindOf = (owner: WorkOwner): null | UpdateBackendKind => KINDS[owner.connectionId ?? owner.profile ?? ''] ?? null

test('a chat on a URL or cloud remote never holds a local update; local, SSH and unknown owners do', () => {
  const sessions = normalizeWorkOwners([
    { title: 'On the VPS', owner: { connectionId: 'vps' } },
    { title: 'Cloud agent', owner: { profile: 'cloud' } },
    { title: 'Local refactor', owner: { connectionId: 'laptop', profile: 'default' } },
    { title: 'Lab box', owner: { profile: 'lab' } },
    { title: 'Owner unknown', owner: null },
    { title: 'Unresolvable', owner: { connectionId: 'deleted' } }
  ])

  const work = workInterruptedByUpdate([{ count: 6, titles: sessions.map(s => s.title), sessions }], kindOf)

  assert.deepEqual(work, { count: 4, titles: ['Local refactor', 'Lab box', 'Owner unknown', 'Unresolvable'] })
  assert.deepEqual(
    workInterruptedByUpdate([{ count: 1, titles: ['On the VPS'], sessions: sessions.slice(0, 1) }], kindOf),
    { count: 0, titles: [] }
  )
})

test('a report without owners, or with owners out of step with its count, still counts in full', () => {
  const remoteOnly = normalizeWorkOwners([{ title: 'On the VPS', owner: { connectionId: 'vps' } }])

  assert.equal(workInterruptedByUpdate([{ count: 2, titles: ['A', 'B'] }], kindOf).count, 2)
  assert.equal(workInterruptedByUpdate([{ count: 2, titles: ['A'], sessions: remoteOnly }], kindOf).count, 2)
  assert.deepEqual(normalizeWorkOwners('junk'), [])
})

test('the update probes the local backend and SSH-managed ones, never a URL remote', () => {
  const local = { baseUrl: 'http://127.0.0.1:9119', mode: 'local' }
  const ssh = { baseUrl: 'http://127.0.0.1:42001', mode: 'remote', remoteKind: 'ssh' }
  const vps = { baseUrl: 'https://vps.example', mode: 'remote', remoteKind: 'url' }

  assert.deepEqual(backendsInterruptedByUpdate([local, null, ssh, vps, { ...local }]), [local, ssh])
})

test('a window that closes stops counting; the SSH backend it ran on is still probed', async () => {
  const windows = new Map<number, OwnedActiveWork>([
    [
      1,
      {
        count: 1,
        titles: ['Lab job'],
        sessions: normalizeWorkOwners([{ title: 'Lab job', owner: { profile: 'lab' } }])
      }
    ]
  ])

  const ssh = { baseUrl: 'http://127.0.0.1:42001', mode: 'remote', remoteKind: 'ssh' }

  const sources = updateWorkSources({
    connections: () => [
      Promise.resolve({ baseUrl: 'http://127.0.0.1:9119', mode: 'local' }),
      Promise.resolve(ssh),
      null
    ],
    reports: () => windows.values(),
    remoteOnly: () => false,
    registryConnections: () => [],
    profileRouteKind: profile => (profile === 'lab' ? 'ssh' : null)
  })

  assert.deepEqual(sources.activeWork(), { count: 1, titles: ['Lab job'] })
  windows.delete(1) // the secondary window closed (or its renderer crashed) mid-turn
  assert.deepEqual(sources.activeWork(), { count: 0, titles: [] })
  assert.deepEqual(
    (await sources.localBackends()).map(backend => backend.baseUrl),
    ['http://127.0.0.1:9119', 'http://127.0.0.1:42001']
  )
})
