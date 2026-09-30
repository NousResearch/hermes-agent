import { expect, it } from 'vitest'

import { EXTRA_TIP_COPY, EXTRA_TIPS, selectExtraTips } from './playbook-extras'

const none = { browserSignedIn: false, googleConnected: false, vaultExists: false }

it('offers only what is connected: mail tips need Google, file tips need the vault', () => {
  const bare = selectExtraTips(none).map(tip => tip.id)

  expect(bare).toEqual(['voice.daddy'])

  const full = selectExtraTips({ browserSignedIn: true, googleConnected: true, vaultExists: true })

  expect(full).toHaveLength(EXTRA_TIPS.length)
  expect(selectExtraTips({ ...none, googleConnected: true }).every(tip => !tip.needs || tip.needs === 'google')).toBe(true)
})

it('every tip has words in both languages, and any prompt it carries is non-empty text to send', () => {
  for (const tip of EXTRA_TIPS) {
    for (const lang of ['en', 'pl'] as const) {
      const copy = EXTRA_TIP_COPY[lang][tip.id]

      expect(copy.title.length).toBeGreaterThan(0)
      expect(copy.detail.length).toBeGreaterThan(0)
    }

    expect((EXTRA_TIP_COPY.pl[tip.id].prompt === '') === (EXTRA_TIP_COPY.en[tip.id].prompt === '')).toBe(true)
  }
})
