import { describe, expect, it } from 'vitest'

import { composerHost } from './composer'

describe('host.composer.setSpellcheck', () => {
  it('resolves false when no mounted surface claims the request', async () => {
    // The reviewer's scenario: the address resolves to a surface with no
    // subscriber, so the request is dropped. The verb must report that,
    // rather than claiming the setting was applied.
    await expect(composerHost.setSpellcheck(null, true)).resolves.toBe(false)
  })

  it('resolves false for an address that resolves to no surface at all', async () => {
    // A session id the primary composer is not showing resolves to a tile
    // target; with no tile mounted, nothing claims it.
    await expect(composerHost.setSpellcheck('sess-does-not-exist', true)).resolves.toBe(false)
  })
})
