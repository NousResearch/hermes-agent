import { describe, expect, it } from 'vitest'

import {
  adaptStringOverrides,
  applyDocumentLocale,
  flattenMessageKeys,
  formatPositional,
  LOCALE_ENDONYMS,
  mergeTranslations,
  RTL_LOCALES,
  type TranslationOverride,
  unflattenMessages
} from './i18n'

describe('formatPositional', () => {
  it('substitutes positional placeholders and blanks the ones the call left out', () => {
    expect(formatPositional('{0} of {1} on', [2, 5])).toBe('2 of 5 on')
    expect(formatPositional('{1} then {0}', ['a', 'b'])).toBe('b then a')
    expect(formatPositional('{0} left{1}', [3])).toBe('3 left')
  })
})

describe('adaptStringOverrides', () => {
  it('wraps a pack string over a function-valued base entry into a positional formatter', () => {
    const base = { menu: { count: (n: number) => `${n} items`, open: 'Open' }, steps: ['one'] }

    const adapted = adaptStringOverrides(base, {
      menu: { count: '{0} Einträge', open: 'Öffnen' },
      steps: ['eins']
    }) as {
      menu: { count: (n: number) => string; open: string }
      steps: string[]
    }

    expect(adapted.menu.count(2)).toBe('2 Einträge')
    expect(adapted.menu.open).toBe('Öffnen')
    expect(adapted.steps).toEqual(['eins'])
    expect(mergeTranslations(base, adapted as never).menu.count(4)).toBe('4 Einträge')
  })

  it('keeps a function override as-is and a string over a string as-is', () => {
    const fn = (n: number) => `${n}!`

    const adapted = adaptStringOverrides({ a: (n: number) => `${n}`, b: 'x' }, { a: fn, b: 'y' }) as {
      a: typeof fn
      b: string
    }

    expect(adapted.a).toBe(fn)
    expect(adapted.b).toBe('y')
  })
})

describe('unflattenMessages / flattenMessageKeys', () => {
  it('rebuilds a nested tree from dotted keys and lists leaves (functions included) sorted', () => {
    const tree = unflattenMessages({ 'common.save': 'Zapisz', 'common.count': '{0}', title: 'T' })

    expect(tree).toEqual({ common: { save: 'Zapisz', count: '{0}' }, title: 'T' })
    expect(flattenMessageKeys({ z: { count: (n: number) => `${n}`, a: 'x' }, list: ['a'], b: 'y' })).toEqual([
      'b',
      'list',
      'z.a',
      'z.count'
    ])
  })

  it('never lets a flat leaf clobber an existing branch', () => {
    expect(unflattenMessages({ 'a.b': 'x', a: 'y' })).toEqual({ a: { b: 'x' } })
    expect(unflattenMessages({ a: 'y', 'a.b': 'x' })).toEqual({ a: 'y' })
  })

  it('keeps dotted leaf keys intact when the base catalog has them', () => {
    const base = { keybinds: { actions: { 'session.slot.1': 'Slot 1', 'nav.settings': 'Settings' } } }
    const flat = { 'keybinds.actions.session.slot.1': 'Gniazdo 1', 'keybinds.actions.nav.settings': 'Ustawienia' }

    expect(unflattenMessages(flat, base)).toEqual({
      keybinds: { actions: { 'session.slot.1': 'Gniazdo 1', 'nav.settings': 'Ustawienia' } }
    })
    expect(flattenMessageKeys(base)).toEqual(['keybinds.actions.nav.settings', 'keybinds.actions.session.slot.1'])
  })
})

describe('mergeTranslations', () => {
  it('keeps untouched sibling keys on a nested partial override and replaces functions/arrays wholesale', () => {
    interface Catalog {
      menu: { close: string; count: (n: number) => string; open: string }
      steps: string[]
    }

    const base: Catalog = {
      menu: { open: 'Open', close: 'Close', count: n => `${n} items` },
      steps: ['one', 'two', 'three']
    }

    const overrides: TranslationOverride<Catalog> = {
      menu: { close: 'Schließen', count: n => `${n} Einträge` },
      steps: ['eins']
    }

    const merged = mergeTranslations<Catalog>(base, overrides)

    expect(merged.menu.open).toBe('Open')
    expect(merged.menu.close).toBe('Schließen')
    expect(merged.menu.count(2)).toBe('2 Einträge')
    expect(merged.steps).toEqual(['eins'])
    expect(base.menu.close).toBe('Close')
  })
})

describe('RTL_LOCALES', () => {
  it('only names locales that have an endonym', () => {
    for (const locale of RTL_LOCALES) {
      expect(Object.keys(LOCALE_ENDONYMS)).toContain(locale)
    }
  })
})

describe('applyDocumentLocale', () => {
  it('is a no-op without a document', () => {
    expect(typeof document).toBe('undefined')
    expect(() => applyDocumentLocale('ar')).not.toThrow()
  })
})
