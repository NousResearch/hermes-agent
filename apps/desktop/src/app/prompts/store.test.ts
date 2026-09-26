import { describe, expect, it, vi } from 'vitest'

import { promptStore, readPrompts, removePrompt, savePrompt } from './store'

describe('personal prompt library', () => {
  it('persists edits and removal without crossing profile boundaries', () => {
    const scope = crypto.randomUUID()
    const other = crypto.randomUUID()
    const saved = savePrompt(scope, { title: 'Plan', body: 'Ask me about my goals' })
    savePrompt(scope, { ...saved, body: 'Make a weekly plan' })
    expect(readPrompts(scope)).toEqual(promptStore(scope).get())
    expect(readPrompts(scope)).toHaveLength(1)
    expect(readPrompts(scope)[0].body).toBe('Make a weekly plan')
    expect(readPrompts(other)).toEqual([])
    removePrompt(scope, saved.id)
    expect(readPrompts(scope)).toEqual([])
  })

  it('keeps the saved prompt when storage rejects an edit', () => {
    const scope = crypto.randomUUID()
    const saved = savePrompt(scope, { title: 'Original', body: 'Original content' })

    const write = vi.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new Error('Quota')
    })

    try {
      expect(() => savePrompt(scope, { ...saved, body: 'Unsaved' })).toThrow('Quota')
      expect(promptStore(scope).get()).toEqual([saved])
    } finally {
      write.mockRestore()
    }
  })
})
