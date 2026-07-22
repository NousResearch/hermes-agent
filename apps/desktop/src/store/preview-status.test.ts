import { beforeEach, describe, expect, it } from 'vitest'

import {
  $previewStatusBySession,
  clearPreviewArtifacts,
  dismissPreviewArtifact,
  getPreviewClearGeneration,
  recordPreviewArtifact,
  recordPreviewArtifacts
} from './preview-status'

beforeEach(() => $previewStatusBySession.set({}))

describe('recordPreviewArtifact', () => {
  it('appends new targets newest-last and is idempotent', () => {
    recordPreviewArtifact('s1', '/a/index.html', '/work')
    recordPreviewArtifact('s1', '/a/about.html', '/work')
    recordPreviewArtifact('s1', '/a/index.html', '/work')

    expect($previewStatusBySession.get().s1.map(i => i.id)).toEqual(['/a/index.html', '/a/about.html'])
  })

  it('caps the list and derives a label', () => {
    for (const n of [1, 2, 3, 4, 5]) {
      recordPreviewArtifact('s1', `/a/p${n}.html`, '/work')
    }

    const list = $previewStatusBySession.get().s1
    expect(list).toHaveLength(4)
    expect(list[0].id).toBe('/a/p2.html')
    expect(list[3].label).toBe('p5.html')
  })

  it('records a large batch atomically and keeps only its newest unique targets', () => {
    let emissions = 0

    const unsubscribe = $previewStatusBySession.subscribe(() => {
      emissions += 1
    })

    recordPreviewArtifacts(
      's1',
      [...Array.from({ length: 100 }, (_, index) => `/a/p${index}.html`), '/a/p99.html'],
      '/work'
    )

    expect($previewStatusBySession.get().s1.map(item => item.id)).toEqual([
      '/a/p96.html',
      '/a/p97.html',
      '/a/p98.html',
      '/a/p99.html'
    ])
    expect(emissions).toBe(2)
    unsubscribe()
  })

  it('preserves sequential eviction semantics within one batch', () => {
    recordPreviewArtifacts('s1', ['/a/a.html', '/a/b.html', '/a/c.html', '/a/d.html'], '/work')
    recordPreviewArtifacts('s1', ['/a/e.html', '/a/a.html'], '/work')

    expect($previewStatusBySession.get().s1.map(item => item.id)).toEqual([
      '/a/c.html',
      '/a/d.html',
      '/a/e.html',
      '/a/a.html'
    ])
  })

  it('dismiss and clear remove rows', () => {
    recordPreviewArtifact('s1', '/a/index.html', '/work')
    recordPreviewArtifact('s1', '/a/about.html', '/work')
    dismissPreviewArtifact('s1', '/a/index.html')
    expect($previewStatusBySession.get().s1.map(i => i.id)).toEqual(['/a/about.html'])

    clearPreviewArtifacts('s1')
    expect($previewStatusBySession.get().s1).toBeUndefined()
  })

  it('advances the reconciliation generation only when the timeline is cleared', () => {
    const before = getPreviewClearGeneration('s1')
    recordPreviewArtifact('s1', '/a/index.html', '/work')
    dismissPreviewArtifact('s1', '/a/index.html')

    expect(getPreviewClearGeneration('s1')).toBe(before)
    clearPreviewArtifacts('s1')
    expect(getPreviewClearGeneration('s1')).toBe(before + 1)
  })
})
