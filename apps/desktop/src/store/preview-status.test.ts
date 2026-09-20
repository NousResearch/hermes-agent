import { beforeEach, describe, expect, it } from 'vitest'

import { rescopeConnectionScopedStores } from '@/lib/connection-scoped'

import {
  $previewStatusBySession,
  clearPreviewArtifacts,
  dismissPreviewArtifact,
  recordPreviewArtifact
} from './preview-status'
import { recordSessionEventScope } from './session-states'

beforeEach(() => {
  window.localStorage.clear()
  $previewStatusBySession.set({})
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

  it('suppresses a dismissed historical target when the row mounts again', () => {
    for (const session_id of ['runtime-a', 'runtime-b']) {
      recordSessionEventScope({ session_id, connectionId: 'owner', profile: 'default' })
    }

    recordPreviewArtifact('runtime-a', '/a/index.html', '/work', 'stored-a')
    dismissPreviewArtifact('runtime-a', '/a/index.html', 'stored-a')
    $previewStatusBySession.set({})

    recordPreviewArtifact('runtime-b', '/a/index.html', '/work', 'stored-a')

    expect($previewStatusBySession.get()['runtime-b']).toBeUndefined()
    expect(window.localStorage.getItem('hermes.desktop.previewDismissals.v1')).toContain('stored-a')
  })

  it('keeps dismissal with the session owner across foreground changes and runtime rebinds', () => {
    const record = (runtime: string, connectionId: string, profile: string) => {
      recordSessionEventScope({ session_id: runtime, connectionId, profile })
      recordPreviewArtifact(runtime, '/work/report.html', '/work', 'same-stored-id')
    }

    record('owner-a', 'local', 'alpha')
    dismissPreviewArtifact('owner-a', '/work/report.html', 'same-stored-id')
    $previewStatusBySession.set({})
    rescopeConnectionScopedStores({ mode: 'remote', baseUrl: 'https://other.invalid', profile: 'other' })
    record('owner-a-rebound', 'local', 'alpha')
    record('other-profile', 'local', 'beta')
    record('other-connection', 'remote', 'alpha')
    expect($previewStatusBySession.get()['owner-a-rebound']).toBeUndefined()
    expect($previewStatusBySession.get()['other-profile']).toHaveLength(1)
    expect($previewStatusBySession.get()['other-connection']).toHaveLength(1)
    rescopeConnectionScopedStores({ mode: 'local' })
  })

  it('deduplicates equivalent file URLs without hiding distinct same-named files', () => {
    recordPreviewArtifact('files', '/work/one/index.html', '/work')
    recordPreviewArtifact('files', 'file:///work/one/index.html', '/work')
    recordPreviewArtifact('files', './one/index.html', '/work')
    recordPreviewArtifact('files', '/work/two/index.html', '/work')

    const items = $previewStatusBySession.get().files
    expect(items).toHaveLength(2)
    expect(new Set(items.map(item => item.label)).size).toBe(2)
    dismissPreviewArtifact('files', items[0].id)
    recordPreviewArtifact('files', 'file:///work/one/index.html', '/work')
    expect($previewStatusBySession.get().files).toHaveLength(1)
  })
})
