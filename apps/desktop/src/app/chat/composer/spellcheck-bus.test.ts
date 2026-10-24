import { describe, expect, it } from 'vitest'

import { ackComposerSpellcheck, onComposerSpellcheckRequest, requestComposerSpellcheck } from './focus'

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

  it('resolves false when the claiming surface reports it did not apply', async () => {
    // A claiming surface with no editable has applied nothing; the ack carries
    // the real outcome rather than a blanket success.
    const off = onComposerSpellcheckRequest(detail => {
      ackComposerSpellcheck(detail.token, false)
    })

    await expect(requestComposerSpellcheck(true, 'main')).resolves.toBe(false)

    off()
  })

  it('resolves false when nothing claims the request', async () => {
    // No subscriber at all: the request is dropped, and the promise settles
    // false on its timeout rather than reporting a silent success.
    await expect(requestComposerSpellcheck(true, 'main')).resolves.toBe(false)
  })

  it('keeps nothing for a surface that was not mounted when asked', async () => {
    // The fail-closed contract: a dropped request leaves no standing state.
    // A surface that mounts afterwards hears nothing and stays at the app
    // default, which is what the caller was told (false).
    await expect(requestComposerSpellcheck(true, 'tile:ghost')).resolves.toBe(false)

    const later: boolean[] = []

    const off = onComposerSpellcheckRequest(detail => {
      later.push(detail.enabled)
    })

    await tick()

    expect(later).toEqual([])

    off()
  })
})
