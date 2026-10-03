import { expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', () => ({ usePluginI18n: () => (key: string) => key }))
vi.mock('../plugins/hermes-bots/shared', () => ({ getPluginCtx: () => null }))

import { SUCCESSION_LOCALES, type SuccessionMessages } from '../plugins/hermes-bots/canonical-group-succession-locales'
import { BOTS_LOCALES } from '../plugins/hermes-bots/i18n'

import { TRANSLATIONS } from './catalog'

type Leaf = SuccessionMessages[keyof SuccessionMessages]

const name = ['Mac mini', null] as const
const counts = [1, 2, 3, 5, 21] as const
const names = (...args: unknown[][]) => args

/** The argument shapes each call site passes: known and unknown names, one and many. */
const SAMPLES: Partial<Record<keyof SuccessionMessages, unknown[][]>> = {
  backupBehind: counts.flatMap(n => name.map(value => [value, n])),
  backupOffline: names(['Home VPS', '14:02'], [null, null]),
  continueOnList: names(['Mac mini', 'Home VPS and Laptop'], [null, 'Home VPS']),
  atRisk: counts.flatMap(n => name.map(value => [n, value])),
  notAllowed: names(['Sam', 'Guest box'], [null, null]),
  addBackupFailed: names(['Work box']),
  menuBehind: counts.flatMap(n => name.map(value => [value, n])),
  botsUnavailable: [...counts.map(n => [n, 'Mac mini', 'Atlas Bot']), [2, null, 'Atlas Bot and Mira Bot']],
  workInProgress: names([2, 1, 1], [0, 0, 3]),
  targetBehind: [...counts.map(n => ['Home VPS', n, 'Mac mini']), [null, 2, null]],
  managedBy: names(['Sam', 'Home VPS'], ['Sam', null]),
  notFenced: names(['Laptop', 1, 'Mac mini'], ['Laptop and Pi', 2, null]),
  notFencedCount: counts.map(n => [n, 'Mac mini']),
  continueFailed: names(['Home VPS', 'It couldn’t be reached.'], [null, 'It couldn’t be reached.']),
  waitingTask: ['bot', 'file', 'tool', 'secret', 'other'].flatMap(resource => name.map(value => [value, resource])),
  continuedSummary: names(['Home VPS', 0, 0, 'Mac mini'], ['Home VPS', 1, 1, 'Mac mini'], ['Home VPS', 3, 2, null], [null, 2, 0, null]),
  createdWithoutSuccessors: names(['Harbor launch']),
  continuedOnTwoBody: names(['Mac mini', 'Home VPS']),
  computerNumber: names([1], [2]),
  keep: names(['Home VPS']),
  keepTitle: names(['Home VPS']),
  keepBody: names(['Mac mini']),
  movedWhileOffline: [0, ...counts].flatMap(n => name.map(value => [value, n]))
}

/** Copy that really is the same words in that language. */
const SAME_AS_ENGLISH = new Set(['de.menuOffline', 'de.computerNumber'])

function renders(key: keyof SuccessionMessages, leaf: Leaf): string[] {
  if (typeof leaf === 'string') {return [leaf]}

  return (SAMPLES[key] ?? name.map(value => [value])).map(args => (leaf as (...values: unknown[]) => string)(...args))
}

it('words every continuation message in all nine Desktop locales', () => {
  const english = SUCCESSION_LOCALES.en
  const locales = Object.keys(TRANSLATIONS)

  expect(Object.keys(SUCCESSION_LOCALES).sort()).toEqual([...locales].sort())

  for (const locale of locales) {
    const messages = SUCCESSION_LOCALES[locale as keyof typeof SUCCESSION_LOCALES]
    expect((BOTS_LOCALES[locale as keyof typeof BOTS_LOCALES] as { succession?: unknown })?.succession, locale).toBe(messages)
    expect(Object.keys(messages).sort(), locale).toEqual(Object.keys(english).sort())

    for (const key of Object.keys(english) as (keyof SuccessionMessages)[]) {
      const ours = renders(key, messages[key]), theirs = renders(key, english[key])
      expect(typeof messages[key], `${locale}.${key}`).toBe(typeof english[key])

      for (const [index, text] of ours.entries()) {
        expect(text.trim(), `${locale}.${key}`).not.toBe('')
        expect(text, `${locale}.${key}`).not.toMatch(/undefined|null|NaN|\{\w+\}|#/)

        if (locale !== 'en' && !SAME_AS_ENGLISH.has(`${locale}.${key}`)) {expect(text, `${locale}.${key}`).not.toBe(theirs[index])}
      }
    }
  }
})

it('keeps the names it is given in every locale', () => {
  for (const [locale, messages] of Object.entries(SUCCESSION_LOCALES)) {
    expect(messages.hostOffline('Mac mini'), locale).toContain('Mac mini')
    expect(messages.confirmTitle('Home VPS'), locale).toContain('Home VPS')
    expect(messages.notAllowed('Sam', 'Guest box'), locale).toContain('Sam')
    expect(messages.notAllowed('Sam', 'Guest box'), locale).toContain('Guest box')
    expect(messages.continuedSummary('Home VPS', 2, 1, 'Mac mini'), locale).toContain('Mac mini')
    expect(messages.managedBy('Sam', 'Home VPS'), locale).toContain('Sam')
    expect(messages.waitingTask('Mac mini', 'file'), locale).toContain('Mac mini')
  }
})
