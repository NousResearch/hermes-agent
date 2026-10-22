import { describe, expect, it } from 'vitest'

import { onComposerSpellcheckRequest, requestComposerSpellcheck } from './focus'

describe('composer spellcheck bus', () => {
  it('delivers the request to the addressed surface', async () => {
    const seen: { enabled: boolean; target: string }[] = []

    const off = onComposerSpellcheckRequest(detail => {
      seen.push({ enabled: detail.enabled, target: detail.target })
    })

    requestComposerSpellcheck(true, 'main')

    // `dispatch` defers to a macrotask so click/keydown handlers settle
    // first -- the request lands on the next tick.
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(seen).toHaveLength(1)
    expect(seen[0].enabled).toBe(true)
    expect(seen[0].target).toBe('main')

    off()
  })

  it('carries the disabled state too', async () => {
    const seen: boolean[] = []

    const off = onComposerSpellcheckRequest(detail => {
      seen.push(detail.enabled)
    })

    requestComposerSpellcheck(false, 'main')

    await new Promise(resolve => setTimeout(resolve, 0))

    expect(seen).toEqual([false])

    off()
  })
})
