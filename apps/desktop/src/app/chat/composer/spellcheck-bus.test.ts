import { describe, expect, it } from 'vitest'

import {
  ackComposerSpellcheck,
  getComposerSpellcheckDesired,
  onComposerSpellcheckRequest,
  requestComposerSpellcheck
} from './focus'

/** Let the bus's deferred dispatch run. */
const tick = () => new Promise(resolve => setTimeout(resolve, 0))

describe('composer spellcheck bus', () => {
  it('delivers the request to the addressed surface', async () => {
    const seen: { enabled: boolean; target: string }[] = []

    const off = onComposerSpellcheckRequest(detail => {
      seen.push({ enabled: detail.enabled, target: detail.target })
    })

    requestComposerSpellcheck(true, 'main')

    await tick()

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

    await tick()

    expect(seen).toEqual([false])

    off()
  })

  it('resolves true when a surface acknowledges the request', async () => {
    const off = onComposerSpellcheckRequest(detail => {
      ackComposerSpellcheck(detail.token, true)
    })

    await expect(requestComposerSpellcheck(true, 'main')).resolves.toBe(true)

    off()
  })

  it('resolves false when nothing claims the request', async () => {
    // No subscriber at all: the request is dropped, and the promise settles
    // false on its timeout rather than reporting a silent success.
    await expect(requestComposerSpellcheck(true, 'main')).resolves.toBe(false)
  })

  it('records the requested state for the surface, so a late mount can read it', async () => {
    const off = onComposerSpellcheckRequest(() => {})

    await requestComposerSpellcheck(true, 'tile:recorded')

    expect(getComposerSpellcheckDesired('tile:recorded')).toBe(true)

    off()
  })

  it('records the disabled state as readily as the enabled one', async () => {
    const off = onComposerSpellcheckRequest(() => {})

    await requestComposerSpellcheck(false, 'tile:disabled')

    expect(getComposerSpellcheckDesired('tile:disabled')).toBe(false)

    off()
  })

  it('keeps each surface\'s recorded state to itself', async () => {
    const off = onComposerSpellcheckRequest(() => {})

    await requestComposerSpellcheck(true, 'tile:one')
    await requestComposerSpellcheck(false, 'tile:two')

    expect(getComposerSpellcheckDesired('tile:one')).toBe(true)
    expect(getComposerSpellcheckDesired('tile:two')).toBe(false)

    off()
  })
})
