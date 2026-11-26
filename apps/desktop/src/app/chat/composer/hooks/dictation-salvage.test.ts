import { beforeEach, describe, expect, it, vi } from 'vitest'

import { salvageDictation } from './dictation-salvage'

const stashSessionDraft = vi.fn()

let draft = { attachments: [] as { id: string }[], text: '' }

vi.mock('@/store/composer', () => ({
  takeSessionDraft: () => ({ attachments: draft.attachments, text: draft.text }),
  stashSessionDraft: (...args: unknown[]) => stashSessionDraft(...args)
}))

describe('salvageDictation', () => {
  beforeEach(() => {
    stashSessionDraft.mockReset()
    draft = { attachments: [{ id: 'file-1' }], text: '' }
  })

  it('parks the transcript in the session it was dictated into', () => {
    expect(salvageDictation('session-a', '  明天下午三点  ')).toBe(true)

    expect(stashSessionDraft).toHaveBeenCalledExactlyOnceWith('session-a', '明天下午三点', [{ id: 'file-1' }])
  })

  it('appends to a draft the user already had, keeping its attachments', () => {
    draft = { attachments: [{ id: 'file-1' }], text: '先写一半' }

    salvageDictation('session-a', '再补一句')

    expect(stashSessionDraft).toHaveBeenCalledExactlyOnceWith('session-a', '先写一半 再补一句', [{ id: 'file-1' }])
  })

  it('writes nothing for an empty transcript', () => {
    expect(salvageDictation('session-a', '   ')).toBe(false)

    expect(stashSessionDraft).not.toHaveBeenCalled()
  })
})
