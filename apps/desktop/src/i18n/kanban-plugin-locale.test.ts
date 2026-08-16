import { afterEach, describe, expect, it } from 'vitest'

import { registerPluginLocales, translatePlugin } from '@/i18n'

import { KANBAN_LOCALES } from '../plugins/kanban/i18n'

interface CatalogLeaf {
  kind: string
  value: unknown
}

function placeholders(value: string): string[] {
  return [...value.matchAll(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g)].map(match => match[1]).sort()
}

function ownLeaves(value: unknown, prefix = ''): Map<string, CatalogLeaf> {
  if (typeof value === 'function' || typeof value === 'string' || value === null) {
    return new Map([[prefix, { kind: value === null ? 'null' : typeof value, value }]])
  }

  if (value && typeof value === 'object') {
    return new Map(
      Object.keys(value as Record<string, unknown>).flatMap(key => [
        ...ownLeaves((value as Record<string, unknown>)[key], prefix ? `${prefix}.${key}` : key)
      ])
    )
  }

  return new Map([[prefix, { kind: typeof value, value }]])
}

const argumentPairs: ReadonlyArray<readonly [unknown, unknown]> = [
  ['alpha', 'beta'],
  [1, 2],
  [false, true],
  [null, 'value'],
  [undefined, 'value']
]

function observesArgument(fn: (...args: never[]) => unknown, arity: number, index: number): boolean {
  for (const fill of ['value', 1, false, null, undefined]) {
    for (const [left, right] of argumentPairs) {
      const leftArgs: unknown[] = Array.from({ length: arity }, () => fill)
      const rightArgs = [...leftArgs]
      leftArgs[index] = left
      rightArgs[index] = right

      try {
        if (fn(...(leftArgs as never[])) !== fn(...(rightArgs as never[]))) {
          return true
        }
      } catch {
        // This vector does not match the callback's runtime input shape.
      }
    }
  }

  return false
}

let disposeLocale = () => {}

afterEach(() => {
  disposeLocale()

  disposeLocale = () => {}
})

describe('Polish Kanban plugin catalog', () => {
  it('matches English executable paths, value kinds, callback arities, and argument flow', () => {
    const englishLeaves = ownLeaves(KANBAN_LOCALES.en)
    const polishLeaves = ownLeaves(KANBAN_LOCALES.pl)

    expect([...polishLeaves.keys()].sort()).toEqual([...englishLeaves.keys()].sort())

    for (const [path, englishLeaf] of englishLeaves) {
      const polishLeaf = polishLeaves.get(path)
      expect(polishLeaf?.kind, path).toBe(englishLeaf.kind)

      if (englishLeaf.kind === 'string' && polishLeaf?.kind === 'string') {
        expect(placeholders(polishLeaf.value as string), `${path} placeholders`).toEqual(
          placeholders(englishLeaf.value as string)
        )
      }

      if (englishLeaf.kind !== 'function' || polishLeaf?.kind !== 'function') {
        continue
      }

      const englishFn = englishLeaf.value as (...args: never[]) => unknown
      const polishFn = polishLeaf.value as (...args: never[]) => unknown
      expect(polishFn.length, `${path} arity`).toBe(englishFn.length)

      for (let index = 0; index < englishFn.length; index += 1) {
        expect(observesArgument(polishFn, polishFn.length, index), `${path} argument ${index}`).toBe(
          observesArgument(englishFn, englishFn.length, index)
        )
      }
    }
  })

  it('resolves Polish strings and callback arguments through translatePlugin', () => {
    disposeLocale = registerPluginLocales('kanban-polish-seam', KANBAN_LOCALES)

    expect(translatePlugin('kanban-polish-seam', 'pl', 'newTask', [])).toBe('Nowe zadanie')
    expect(translatePlugin('kanban-polish-seam', 'pl', 'col.ready.label', [])).toBe('Gotowe')
    expect(translatePlugin('kanban-polish-seam', 'pl', 'allTenants', [])).toBe('Wszystkie przestrzenie')
    expect(translatePlugin('kanban-polish-seam', 'pl', 'metaTenant', [])).toBe('Przestrzeń')
    expect(translatePlugin('kanban-polish-seam', 'pl', 'countTip', [2, 5])).toBe('Kanban — w toku: 2, gotowe: 5')
    expect(translatePlugin('kanban-polish-seam', 'pl', 'bulkFailed', [2, 7, 'brak połączenia'])).toBe(
      'Nie udało się: 2 z 7 — brak połączenia. Nieudane karty pozostają wybrane.'
    )
    expect(translatePlugin('kanban-polish-seam', 'pl', 'evtCreated', ['Gotowe', 'default'])).toBe(
      'utworzono · kolumna: Gotowe · przypisany profil: default'
    )
  })
})
