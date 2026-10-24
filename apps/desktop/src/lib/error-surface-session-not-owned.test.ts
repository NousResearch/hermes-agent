import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n'
import { en } from '@/i18n/en'

import { errorCardOffers, type ErrorSurface } from './error-surface'
import { errorCardText } from './error-surface-copy'

// The 4090 refusal: another backend holds this chat's lease. That backend may have no window left:
// an idle Desktop lease is only released by the gateway's session reaper, up to 15 minutes later
// (tui_gateway `_IDLE_LEASE_RELEASE_S` plus one `_REAPER_SCAN_S`). So the card must not claim the
// chat is open somewhere, and must say the chat frees itself.
const surface: ErrorSurface = { code: 'SESSION_NOT_OWNED', layer: 'gateway', retryable: false }

describe('SESSION_NOT_OWNED card', () => {
  it('does not claim the chat is open in another window', () => {
    const { body, title } = errorCardText(en.assistant.thread, surface)

    expect(title).not.toMatch(/\bopen\b/i)
    expect(body).not.toMatch(/\bis open in\b/i)
  })

  it('says the chat frees itself and still offers a new chat, not a retry', () => {
    const { body } = errorCardText(en.assistant.thread, surface)

    expect(body).toMatch(/free itself/)
    expect(body).toMatch(/start a new chat/i)
    expect(errorCardOffers(surface, 'start_new_session')).toBe(true)
    expect(errorCardOffers(surface, 'retry')).toBe(false)
  })

  it('names the same bound in every locale', () => {
    for (const [locale, copy] of Object.entries(TRANSLATIONS)) {
      expect(errorCardText(copy.assistant.thread, surface).body, locale).toMatch(/15/)
    }
  })
})
