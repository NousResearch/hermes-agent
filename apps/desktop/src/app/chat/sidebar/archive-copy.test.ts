import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n/catalog'

import { resolveSessionRowClick } from './session-row-gesture'

describe('archived-session instructions', () => {
  it('advertises modifiers that archive rather than open a tab in every locale', () => {
    for (const [locale, copy] of Object.entries(TRANSLATIONS)) {
      const intro = copy.settings.sessions.archivedIntro

      const modifiers = {
        altKey: /Alt|Option|⌥/.test(intro),
        shiftKey: /Shift|⇧/.test(intro),
        ctrlKey: /Ctrl|⌃/.test(intro),
        metaKey: /⌘/.test(intro)
      }

      expect(resolveSessionRowClick(modifiers, { canOpenWindow: true }), locale).toBe('archive')
      expect(resolveSessionRowClick(modifiers, { canOpenWindow: false }), locale).toBe('archive')
    }
  })
})
