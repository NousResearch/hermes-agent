import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $dismissedPreviewPublications,
  $previewStatusBySession,
  clearPreviewArtifacts,
  dismissPreviewArtifact,
  previewArtifactOwnerId,
  type PreviewArtifactPublication,
  previewArtifactPublicationKey,
  syncPreviewArtifacts
} from './preview-status'

const STORAGE_KEY = 'hermes.desktop.dismissedPreviewPublications.v1'

const publicationKey = (publicationId: string, ownerId = 'lineage-root', profile = 'default') =>
  previewArtifactPublicationKey({ mode: 'local', ownerId, profile, publicationId })

const publicationsBySession = new Map<string, PreviewArtifactPublication[]>()

const record = (runtimeId: string, target: string, publicationId: string, ownerId = 'lineage-root') => {
  const publications = publicationsBySession.get(runtimeId) ?? []
  publications.push({ cwd: '/work', publicationKey: publicationKey(publicationId, ownerId), target })
  publicationsBySession.set(runtimeId, publications)
  syncPreviewArtifacts(runtimeId, publications)
}

beforeEach(() => {
  window.localStorage.removeItem(STORAGE_KEY)
  publicationsBySession.clear()
  $dismissedPreviewPublications.set([])
  $previewStatusBySession.set({})
})

describe('syncPreviewArtifacts', () => {
  it('keys publications by an opaque compression-stable owner identity', () => {
    const sessions = [{ id: 'compression-tip', _lineage_root_id: 'lineage-root' }]
    const ownerId = previewArtifactOwnerId('compression-tip', sessions)!

    const identity = {
      baseUrl: 'https://user:secret@example.test/gateway?token=hidden',
      mode: 'remote' as const,
      ownerId,
      profile: 'work',
      publicationId: 'tool:call-1'
    }

    const keys = [
      identity,
      { ...identity, baseUrl: 'https://other.example.test' },
      { ...identity, mode: 'local' as const, profile: 'other' },
      { ...identity, mode: 'local' as const, ownerId: 'other-lineage' },
      { ...identity, mode: 'local' as const, publicationId: 'tool:call-2' }
    ].map(previewArtifactPublicationKey)

    const [base] = keys

    expect(ownerId).toBe('lineage-root')
    expect(base).toMatch(/^p:[a-z0-9]{7}-[a-z0-9]{7}$/)
    expect(base).not.toContain('secret')
    expect(base).not.toContain('lineage-root')
    expect(new Set(keys)).toHaveLength(5)
  })

  it('resolves the lineage root within the owning profile', () => {
    const sessions = [
      { id: 'same-tip', _lineage_root_id: 'legacy-unscoped-root' },
      { id: 'same-tip', _lineage_root_id: 'default-root', profile: 'default' },
      { id: 'same-tip', _lineage_root_id: 'work-root', profile: 'work' }
    ]

    expect(previewArtifactOwnerId('same-tip', sessions, 'work')).toBe('work-root')
    expect(previewArtifactOwnerId('legacy-tip', [{ id: 'legacy-tip', _lineage_root_id: 'legacy-root' }], 'work')).toBe(
      'legacy-root'
    )
    expect(previewArtifactOwnerId('unlisted-compression-tip', sessions, 'work')).toBeNull()
  })

  it('appends new targets newest-last and is idempotent', () => {
    record('s1', '/a/index.html', 'tool:index')
    record('s1', '/a/about.html', 'tool:about')
    record('s1', '/a/index.html', 'tool:index')

    expect($previewStatusBySession.get().s1.map(i => i.id)).toEqual(['/a/index.html', '/a/about.html'])
  })

  it('caps the list and derives a label', () => {
    for (const n of [1, 2, 3, 4, 5]) {
      record('s1', `/a/p${n}.html`, `tool:${n}`)
    }

    const list = $previewStatusBySession.get().s1
    expect(list).toHaveLength(4)
    expect(list[0].id).toBe('/a/p2.html')
    expect(list[3].label).toBe('p5.html')
  })

  it('dismiss and clear remove rows', () => {
    record('s1', '/a/index.html', 'tool:index')
    record('s1', '/a/about.html', 'tool:about')
    dismissPreviewArtifact('s1', '/a/index.html')
    expect($previewStatusBySession.get().s1.map(i => i.id)).toEqual(['/a/about.html'])

    clearPreviewArtifacts('s1')
    expect($previewStatusBySession.get().s1).toBeUndefined()
  })

  it('does not resurrect a dismissed publication after relaunch under a new runtime id', async () => {
    const target = '/a/architecture-diagram.html'
    const key = publicationKey('tool:call-1')
    syncPreviewArtifacts('runtime-1', [{ cwd: '/work', publicationKey: key, target }])

    dismissPreviewArtifact('runtime-1', target)

    expect(JSON.parse(String(window.localStorage.getItem(STORAGE_KEY)))).toEqual([key])
    vi.resetModules()
    const fresh = await import('./preview-status')
    fresh.syncPreviewArtifacts('runtime-2', [{ cwd: '/work', publicationKey: key, target }])
    expect(fresh.$previewStatusBySession.get()['runtime-2']).toBeUndefined()
  })

  it('keeps preserved history dismissed while allowing a fresh tool call to republish the same target', () => {
    record('runtime-1', '/a/architecture-diagram.html', 'tool:old-call')
    dismissPreviewArtifact('runtime-1', '/a/architecture-diagram.html')

    // Rewind/edit clears the derived feed before the backend operation. If it
    // fails, the preserved historical call remounts and must remain dismissed.
    clearPreviewArtifacts('runtime-1')
    record('runtime-1', '/a/architecture-diagram.html', 'tool:old-call')
    expect($previewStatusBySession.get()['runtime-1']).toBeUndefined()

    record('runtime-1', '/a/architecture-diagram.html', 'tool:new-call')
    expect($previewStatusBySession.get()['runtime-1'].map(i => i.id)).toEqual(['/a/architecture-diagram.html'])
    expect($dismissedPreviewPublications.get()).toEqual([publicationKey('tool:old-call')])
  })

  it('dismisses every historical tool call represented by a deduplicated target row', () => {
    const target = '/a/architecture-diagram.html'
    record('runtime-1', target, 'tool:call-1')
    const visibleState = $previewStatusBySession.get()
    record('runtime-1', target, 'tool:call-2')

    expect($previewStatusBySession.get()).toBe(visibleState)
    expect($previewStatusBySession.get()['runtime-1']).toHaveLength(1)

    dismissPreviewArtifact('runtime-1', target)
    $previewStatusBySession.set({})
    record('runtime-2', target, 'tool:call-1')
    record('runtime-2', target, 'tool:call-2')
    expect($previewStatusBySession.get()['runtime-2']).toBeUndefined()

    record('runtime-2', target, 'tool:call-3')
    expect($previewStatusBySession.get()['runtime-2']).toHaveLength(1)
  })

  it('retains duplicate provenance when a target is evicted by the visible-row cap', () => {
    record('runtime-1', '/A.html', 'tool:A-old')

    for (const name of ['B', 'C', 'D', 'E']) {
      record('runtime-1', `/${name}.html`, `tool:${name}`)
    }

    record('runtime-1', '/A.html', 'tool:A-new')
    expect($previewStatusBySession.get()['runtime-1'].map(item => item.id)).toEqual([
      '/C.html',
      '/D.html',
      '/E.html',
      '/A.html'
    ])

    dismissPreviewArtifact('runtime-1', '/A.html')
    $previewStatusBySession.set({})
    syncPreviewArtifacts('runtime-2', publicationsBySession.get('runtime-1') ?? [])

    expect($previewStatusBySession.get()['runtime-2'].map(item => item.id)).toEqual([
      '/B.html',
      '/C.html',
      '/D.html',
      '/E.html'
    ])
  })

  it('scopes dismissals to the owning profile and stored session', () => {
    const target = '/a/architecture-diagram.html'
    syncPreviewArtifacts('runtime-1', [
      { cwd: '/work', publicationKey: publicationKey('tool:call-1', 'stored-1'), target }
    ])
    dismissPreviewArtifact('runtime-1', target)

    syncPreviewArtifacts('runtime-2', [
      { cwd: '/work', publicationKey: publicationKey('tool:call-1', 'stored-1', 'other'), target }
    ])

    expect($previewStatusBySession.get()['runtime-1']).toBeUndefined()
    expect($previewStatusBySession.get()['runtime-2'].map(i => i.id)).toEqual([target])
  })

  it('hydrates a validated fixed-size dismissal list within a small storage budget', async () => {
    const oversized = `p:${'a'.repeat(100_000)}-b`

    const valid = Array.from({ length: 5_000 }, (_, index) => publicationKey(`tool:${index}`))

    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(['', 'raw-sensitive-value', ...valid, valid.at(-1), oversized])
    )
    vi.resetModules()

    const fresh = await import('./preview-status')
    const hydrated = fresh.$dismissedPreviewPublications.get()
    const encoded = JSON.stringify(hydrated)

    expect(hydrated.length).toBeLessThan(valid.length)
    expect(hydrated.at(-1)).toBe(valid.at(-1))
    expect(hydrated).not.toContain(oversized)
    expect(new Set(hydrated)).toHaveLength(hydrated.length)
    expect(hydrated.every(key => /^p:[a-z0-9]+-[a-z0-9]+$/.test(key))).toBe(true)
    expect(encoded.length).toBeLessThan(100_000)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(encoded)

    const oldest = hydrated[0]

    const newest = publicationKey('tool:newest')

    fresh.syncPreviewArtifacts('runtime', [{ cwd: '/work', publicationKey: newest, target: '/a/new.html' }])
    fresh.dismissPreviewArtifact('runtime', '/a/new.html')

    expect(fresh.$dismissedPreviewPublications.get()).toHaveLength(hydrated.length)
    expect(fresh.$dismissedPreviewPublications.get()).not.toContain(oldest)
    expect(fresh.$dismissedPreviewPublications.get().at(-1)).toBe(newest)
  })
})
