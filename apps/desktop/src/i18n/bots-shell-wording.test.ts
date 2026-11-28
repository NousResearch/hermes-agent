import { afterEach, describe, expect, it } from 'vitest'

import { BOTS_LOCALES } from '@/plugins/hermes-bots/i18n'
import { BOTS_REDESIGN_WORDING } from '@/plugins/hermes-bots/redesign-wording'
import { ID } from '@/plugins/hermes-bots/shared'

import { registerPluginLocales, translatePlugin } from './plugin-i18n'

let dispose = () => {}

afterEach(() => dispose())

const register = () => {
  dispose = registerPluginLocales(ID, BOTS_LOCALES, { redesign: BOTS_REDESIGN_WORDING })
}

describe('Bot Mode wording per shell', () => {
  it('reads main’s classic wording in classic and the redesign’s nouns in the redesign', () => {
    register()

    expect(translatePlugin(ID, 'en', 'roster.title', [], 'classic')).toBe('Bots')
    expect(translatePlugin(ID, 'en', 'roster.title', [], 'redesign')).toBe('Agents')
    expect(translatePlugin(ID, 'en', 'group.memberCount', [3], 'classic')).toBe('3 bots')
    expect(translatePlugin(ID, 'en', 'group.memberCount', [3], 'redesign')).toBe('3 agents')
    expect(translatePlugin(ID, 'en', 'roster.allGateways', [], 'classic')).toBe('All gateways')
    expect(translatePlugin(ID, 'en', 'roster.allGateways', [], 'redesign')).toBe('All machines')
  })

  it('only rewords keys the base bundles have, in every locale', () => {
    for (const [locale, layer] of Object.entries(BOTS_REDESIGN_WORDING)) {
      const base = BOTS_LOCALES[locale] as Record<string, Record<string, unknown>>

      for (const [namespace, leaves] of Object.entries(layer ?? {})) {
        for (const key of Object.keys(leaves as Record<string, unknown>)) {
          expect(base[namespace]?.[key], `${locale}: ${namespace}.${key}`).toBeDefined()
        }
      }
    }
  })

  it('leaves no classic noun in the redesign’s English wording', () => {
    register()
    const english = BOTS_LOCALES.en as Record<string, Record<string, unknown>>

    for (const [namespace, leaves] of Object.entries(english)) {
      for (const [key, value] of Object.entries(leaves)) {
        const sample = typeof value === 'function' ? [2, 3, 4] : []
        const text = translatePlugin(ID, 'en', `${namespace}.${key}`, sample, 'redesign')

        // `[bot:<name>]` is the cron title tag syntax, not copy.
        expect(text.replace(/\[bot:[^\]]*\]/g, ''), `${namespace}.${key}`).not.toMatch(/\bbots?\b/i)
      }
    }
  })
})
