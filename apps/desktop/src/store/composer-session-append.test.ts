import { afterEach, describe, expect, it } from 'vitest'

import {
  appendComposerToSessionDraft,
  clearSessionDraft,
  onComposerSessionAppendRequest,
  takeSessionDraft
} from '@/store/composer'
import { setSessions } from '@/store/session'

describe('session-scoped composer append', () => {
  afterEach(() => {
    // Keep the test scope isolated from the module-level draft map.
    clearSessionDraft('session-test')
    clearSessionDraft('root-id')
    clearSessionDraft('tip-id')
    setSessions([])
  })

  it('appends text and attachments to the requested durable session draft', () => {
    appendComposerToSessionDraft('session-test', 'web annotation text', [
      {
        id: 'browser-annotation-1',
        kind: 'image',
        label: 'browser-annotation-1.png',
        path: '/tmp/browser-annotation-1.png',
        refText: '@image:/tmp/browser-annotation-1.png',
        attachedSessionId: 'session-test'
      }
    ])

    expect(takeSessionDraft('session-test')).toEqual({
      text: 'web annotation text',
      attachments: [expect.objectContaining({ id: 'browser-annotation-1', kind: 'image' })]
    })
  })

  it('notifies only the matching mounted composer', () => {
    const received: string[] = []
    const dispose = onComposerSessionAppendRequest(detail => {
      detail.handled = true
      received.push(detail.sessionKey)
    })
    appendComposerToSessionDraft('session-test', 'annotation', [])
    dispose()
    expect(received).toEqual(['session-test'])
  })

  it('normalizes a compressed tip id to the durable lineage root', () => {
    setSessions([{ id: 'tip-id', _lineage_root_id: 'root-id' }] as never)
    appendComposerToSessionDraft('tip-id', 'post-compaction annotation', [])
    expect(takeSessionDraft('root-id').text).toBe('post-compaction annotation')
    expect(takeSessionDraft('tip-id').text).toBe('')
  })

  it('upserts repeated attachment ids in the stashed draft', () => {
    const first = { id: 'same', kind: 'image', label: 'old.png' } as const
    const replacement = { id: 'same', kind: 'image', label: 'new.png' } as const
    appendComposerToSessionDraft('session-test', 'one', [first])
    appendComposerToSessionDraft('session-test', 'two', [replacement])
    expect(takeSessionDraft('session-test').attachments).toEqual([replacement])
  })
})
