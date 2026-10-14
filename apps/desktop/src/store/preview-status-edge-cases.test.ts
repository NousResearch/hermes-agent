import { afterEach, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { makeSessionInfo } from '@/test/session-info'

import { $previewStatusBySession, dismissPreviewArtifact, recordPreviewArtifact } from './preview-status'
import { setSessions } from './session'
import {
  clearAllSessionStates,
  migrateTilesForProfile,
  publishSessionState,
  recordSessionEventScope
} from './session-states'

afterEach(() => {
  $previewStatusBySession.set({})
  setSessions([])
  clearAllSessionStates()
  window.localStorage.clear()
})

it('retains a close when the same runtime owner is refined from profile-only to an exact route', () => {
  setSessions([makeSessionInfo({ id: 'refined-stored', profile: 'probe' })])
  publishSessionState('refined-runtime', createClientSessionState('refined-stored'))
  recordPreviewArtifact('refined-runtime', '/work/refined.html', '/work', 'refined-stored')
  recordSessionEventScope({ session_id: 'refined-runtime', connectionId: 'local', profile: 'probe' })
  dismissPreviewArtifact('refined-runtime', '/work/refined.html', 'refined-stored')
  recordPreviewArtifact('refined-runtime', '/work/refined.html', '/work', 'refined-stored')
  expect($previewStatusBySession.get()['refined-runtime']).toBeUndefined()

  setSessions([makeSessionInfo({ id: 'refined-append-stored', profile: 'probe-append' })])
  publishSessionState('refined-append', createClientSessionState('refined-append-stored'))
  recordPreviewArtifact('refined-append', '/work/first.html', '/work', 'refined-append-stored')
  recordSessionEventScope({ session_id: 'refined-append', connectionId: 'local', profile: 'probe-append' })
  recordPreviewArtifact('refined-append', '/work/second.html', '/work', 'refined-append-stored')
  dismissPreviewArtifact('refined-append', '/work/first.html', 'refined-append-stored')
  recordPreviewArtifact('refined-append', '/work/first.html', '/work', 'refined-append-stored')
  expect($previewStatusBySession.get()['refined-append'].map(item => item.id)).toEqual(['/work/second.html'])
})

it('tolerates malformed persisted scope keys and keeps Windows aliases identical', () => {
  window.localStorage.setItem('hermes.desktop.previewDismissals.v1', '{"__proto__":["x"],"constructor":["y"]}')
  expect(() => migrateTilesForProfile('old-profile', 'new-profile')).not.toThrow()
  recordPreviewArtifact('windows-alias', './report.html', 'C:\\work')
  const item = $previewStatusBySession.get()['windows-alias'][0]
  dismissPreviewArtifact('windows-alias', item.id)
  recordPreviewArtifact('windows-alias', 'file:///C:/work/report.html', 'C:\\work')
  expect($previewStatusBySession.get()['windows-alias']).toBeUndefined()
})
