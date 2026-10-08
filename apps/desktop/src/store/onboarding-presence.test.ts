import { afterEach, describe, expect, it, vi } from 'vitest'

import type { FreeTierStatus } from '@/types/hermes'

const NOTICE_OWED: FreeTierStatus = {
  available: true,
  enabled: true,
  has_guest: true,
  label: 'Nous · free tier',
  model: 'nous/welcome',
  notice_pending: true
}

// The window kind is read once from `?win=` at startup, so each case loads the stores fresh.
async function storesIn(search: string) {
  window.history.replaceState(null, '', `/${search}`)
  vi.resetModules()

  return import('@/store/free-tier')
}

afterEach(() => window.history.replaceState(null, '', '/'))

describe('questionnaire decision per window', () => {
  it.each(['?win=secondary', '?win=hud'])('%s never runs the questionnaire, so its composer shows the owed strip', async search => {
    const { freeTierStripPending } = await storesIn(search)

    expect(freeTierStripPending(NOTICE_OWED, false)).toBe(true)
  })

  it('the main window holds the strip until its due check answers', async () => {
    const { freeTierStripPending } = await storesIn('')

    expect(freeTierStripPending(NOTICE_OWED, false)).toBe(false)
  })
})
