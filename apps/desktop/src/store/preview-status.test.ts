import { beforeEach, describe, expect, it } from 'vitest'

import {
  $dismissedPreviewArtifacts,
  $previewStatusBySession,
  clearPreviewArtifacts,
  dismissPreviewArtifact,
  isPreviewArtifactDismissed,
  recordPreviewArtifact,
  undismissPreviewArtifact
} from './preview-status'

beforeEach(() => {
  $previewStatusBySession.set({})
  $dismissedPreviewArtifacts.set([])
  window.localStorage.clear()
})

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

  it('dismiss and clear remove rows', () => {
    recordPreviewArtifact('s1', '/a/index.html', '/work')
    recordPreviewArtifact('s1', '/a/about.html', '/work')
    dismissPreviewArtifact('s1', '/a/index.html')
    expect($previewStatusBySession.get().s1.map(i => i.id)).toEqual(['/a/about.html'])

    clearPreviewArtifacts('s1')
    expect($previewStatusBySession.get().s1).toBeUndefined()
  })
})

describe('dismissal durability', () => {
  it('a dismissed target does not come back when its tool row re-registers', () => {
    recordPreviewArtifact('s1', '/a/one-pager.html', '/work')
    dismissPreviewArtifact('s1', '/a/one-pager.html')

    // Transcript virtualization / session switch / reconnect remounts the row.
    recordPreviewArtifact('s1', '/a/one-pager.html', '/work')

    expect($previewStatusBySession.get().s1).toBeUndefined()
    expect(isPreviewArtifactDismissed('s1', '/a/one-pager.html')).toBe(true)
  })

  it('is scoped per session', () => {
    dismissPreviewArtifact('s1', '/a/one-pager.html')
    recordPreviewArtifact('s2', '/a/one-pager.html', '/work')

    expect($previewStatusBySession.get().s2.map(i => i.id)).toEqual(['/a/one-pager.html'])
  })

  it('persists the dismissal to storage', () => {
    dismissPreviewArtifact('s1', '/a/one-pager.html')

    const stored = window.localStorage.getItem('hermes.desktop.previewStatusDismissed.v1') ?? ''
    expect(JSON.parse(stored)).toEqual(['s1\u0000/a/one-pager.html'])
  })

  it('undismiss lets the artifact surface again', () => {
    dismissPreviewArtifact('s1', '/a/one-pager.html')
    undismissPreviewArtifact('s1', '/a/one-pager.html')
    recordPreviewArtifact('s1', '/a/one-pager.html', '/work')

    expect($previewStatusBySession.get().s1.map(i => i.id)).toEqual(['/a/one-pager.html'])
  })

  it('clear does not forget dismissals', () => {
    recordPreviewArtifact('s1', '/a/one-pager.html', '/work')
    dismissPreviewArtifact('s1', '/a/one-pager.html')
    clearPreviewArtifacts('s1')
    recordPreviewArtifact('s1', '/a/one-pager.html', '/work')

    expect($previewStatusBySession.get().s1).toBeUndefined()
  })

  it('bounds remembered dismissals, evicting the oldest', () => {
    for (let n = 0; n < 505; n++) {
      dismissPreviewArtifact('s1', `/a/p${n}.html`)
    }

    const remembered = $dismissedPreviewArtifacts.get()
    expect(remembered).toHaveLength(500)
    expect(remembered[0]).toBe('s1\u0000/a/p5.html')
    expect(isPreviewArtifactDismissed('s1', '/a/p4.html')).toBe(false)
    expect(isPreviewArtifactDismissed('s1', '/a/p504.html')).toBe(true)
  })

  it('re-dismissing moves the key to newest without duplicating it', () => {
    dismissPreviewArtifact('s1', '/a/x.html')
    dismissPreviewArtifact('s1', '/a/y.html')
    dismissPreviewArtifact('s1', '/a/x.html')

    expect($dismissedPreviewArtifacts.get()).toEqual(['s1\u0000/a/y.html', 's1\u0000/a/x.html'])
  })
})
