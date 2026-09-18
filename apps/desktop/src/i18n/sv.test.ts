import { describe, expect, it, vi } from 'vitest'

import { BOTS_LOCALES } from '@/plugins/hermes-bots/i18n'
import { KANBAN_LOCALES } from '@/plugins/kanban/i18n'

import { en } from './en'
import { sv } from './sv'

// Inspect the supplied translations: merging English first would hide omissions.
vi.mock('./define-locale', () => ({ defineLocale: (overrides: unknown) => overrides }))

function leaves(node: unknown, prefix = ''): Array<[string, unknown]> {
  if (typeof node !== 'object' || node === null) {
    return [[prefix, node]]
  }

  return Object.entries(node).flatMap(([key, value]) => leaves(value, prefix ? `${prefix}.${key}` : key))
}

describe('Swedish translations', () => {
  it.each([
    ['desktop', en, sv],
    ['kanban', KANBAN_LOCALES.en, KANBAN_LOCALES.sv],
    ['bots', BOTS_LOCALES.en, BOTS_LOCALES.sv]
  ])('covers the English message tree without fallback: %s', (_name, english, swedish) => {
    const source = Object.fromEntries(leaves(english))
    const translated = Object.fromEntries(leaves(swedish))

    expect(Object.keys(translated).sort()).toEqual(Object.keys(source).sort())

    for (const [path, value] of Object.entries(source)) {
      expect(typeof translated[path], path).toBe(typeof value)
    }
  })

  it('keeps the selected profile and actual installation directory in the destination', () => {
    const profile = 'work-profile'
    const directory = '/custom/hermes/profiles/work/plugins'
    const message = sv.settings.plugins.installModal.agentTargetLocal(profile, directory)

    expect(message).toContain(profile)
    expect(message).toContain(directory)
  })

  it.each([0, 1, 2, 21])('preserves counts in pluralized controls: %i', count => {
    expect(sv.connectors.startWith(count)).toContain(String(count))
    expect(sv.statusStack.control.goalDoneTurns(count)).toContain(String(count))
    expect(sv.statusStack.control.loopRuns(count)).toContain(String(count))
    expect(sv.statusStack.control.heartbeatFiredCount(count)).toContain(String(count))
  })

  it('retains literal schema keys and security-sensitive destinations', () => {
    expect(sv.cron.modelImpact.message(2)).toContain('cron.model')
    expect(sv.prompts.vaultSaveDesc('https://example.com')).toContain('https://example.com')
    expect(sv.prompts.vaultUnlockDesc('Example Vault')).toContain('Example Vault')
    expect(sv.assistant.thread.errorOauthExpired('Example Provider')).toContain('Example Provider')
  })
})
