import { afterEach, expect, test, vi } from 'vitest'

import { SESSION_DRAFTS_STORAGE_KEY, stashSessionDraft, takeSessionDraft } from './composer'

afterEach(() => { localStorage.clear(); sessionStorage.clear(); vi.restoreAllMocks() })

test('the persisted draft hydrates its exact submission provenance after all renderer state is discarded', async () => {
  const attachment = { id: 'image', occurrenceId: 'occurrence', kind: 'image' as const, label: 'image.png',
    path: '/cache/retained.png', previewUrl: 'blob:renderer-only' }

  stashSessionDraft('saved-session', 'pending message', [attachment], {
    id: 'original-operation', text: 'pending message', attachmentIds: ['occurrence'], fromQueue: true, pending: true
  })
  const persisted = localStorage.getItem(SESSION_DRAFTS_STORAGE_KEY)!
  expect(persisted).toContain('original-operation')
  expect(persisted).not.toContain('blob:renderer-only')
  sessionStorage.clear()
  vi.resetModules()
  const restarted = await import('./composer')
  expect(restarted.takeSessionDraft('saved-session')).toMatchObject({
    text: 'pending message', attachments: [{ id: 'image', occurrenceId: 'occurrence', path: '/cache/retained.png' }],
    retry: { id: 'original-operation', fromQueue: true, pending: false }
  })
  // Fresh typing records a new draft, even if another window's text is equal.
  restarted.stashSessionDraft('saved-session', 'pending message', [])
  expect(restarted.takeSessionDraft('saved-session')).not.toHaveProperty('retry')
})

test('a pending operation must persist successfully before its caller can send', () => {
  const set = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('disk full') })
  expect(() => stashSessionDraft('unsaved', 'keep text', [], {
    id: 'unsaved-operation', text: 'keep text', attachmentIds: [], pending: true
  })).toThrow('disk full')
  expect(takeSessionDraft('unsaved').text).toBe('keep text')
  set.mockRestore()
})
