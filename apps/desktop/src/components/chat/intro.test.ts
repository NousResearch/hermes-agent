import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n'
import type { Locale } from '@/i18n'

describe('home screen copy contract', () => {
  const locales = Object.keys(TRANSLATIONS) as Locale[]

  // The Codex-minimal home screen renders a single heading and nothing else, so
  // every locale must ship a non-empty title (no brand mark / tagline / cards).
  it.each(locales)('%s exposes a non-empty home title', locale => {
    expect(TRANSLATIONS[locale].home.title.trim().length).toBeGreaterThan(0)
  })

  it('keeps developer jargon off the Chinese home screen', () => {
    const surface = TRANSLATIONS.zh.home.title.toLowerCase()

    for (const banned of ['traceback', 'repo', 'commit', 'branch', 'file path', 'bug']) {
      expect(surface).not.toContain(banned)
    }
  })
})
