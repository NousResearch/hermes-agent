import { PLUGIN_CATEGORIES } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import ru from './ru.json'

const KINDS = ['skill', 'plugin', 'mcp', 'tool', 'toolset'] as const
const CYRILLIC = /[а-яё]/i

/** Control characters (except tab/newline) or U+FFFD mean a broken encoding. */
const BAD_CHARS = {
  test: (text: string) => [...text].some(char => (char.charCodeAt(0) < 32 && !'\t\n\r'.includes(char)) || char === '\ufffd')
}

type Entry = { name?: unknown; description?: unknown; keywords?: unknown }
const entries = (kind: (typeof KINDS)[number]) => Object.entries((ru as Record<string, Record<string, Entry>>)[kind])

describe('ru catalog content dictionary quality', () => {
  it.each(KINDS)('every %s entry has a clean Russian name, optional description and keywords', kind => {
    const rows = entries(kind)
    expect(rows.length).toBeGreaterThan(0)

    for (const [id, row] of rows) {
      expect(id, `${kind} id`).toBe(id.trim())
      expect(id, `${kind} id`).not.toMatch(/\s/)

      expect(typeof row.name, `${kind}:${id} name`).toBe('string')
      expect(String(row.name), `${kind}:${id} name`).toMatch(CYRILLIC)
      expect(String(row.name), `${kind}:${id} name`).toBe(String(row.name).trim())
      expect(String(row.name), `${kind}:${id} name`).toSatisfy((text: string) => !BAD_CHARS.test(text))

      if (row.description !== undefined) {
        expect(String(row.description), `${kind}:${id} description`).toMatch(CYRILLIC)
        expect(String(row.description), `${kind}:${id} description`).toSatisfy((text: string) => !BAD_CHARS.test(text))
      }

      if (row.keywords !== undefined) {
        expect(Array.isArray(row.keywords), `${kind}:${id} keywords`).toBe(true)
        expect((row.keywords as unknown[]).every(word => typeof word === 'string' && word.trim() !== '')).toBe(true)
      }
    }
  })

  it('never carries template placeholders or active markup into display text', () => {
    for (const kind of KINDS) {
      for (const [id, row] of entries(kind)) {
        expect(`${row.name} ${row.description ?? ''}`, `${kind}:${id}`).not.toMatch(/\$\{|\{\{|<\/?(script|iframe|img|a|style)\b/i)
      }
    }
  })

  it('translates every plugin taxonomy key the shared catalog defines', () => {
    const rows = (ru as { pluginCategories?: Record<string, { label: string; blurb: string }> }).pluginCategories ?? {}

    for (const key of Object.keys(PLUGIN_CATEGORIES)) {
      expect(rows[key]?.label, key).toMatch(CYRILLIC)
      expect(rows[key]?.blurb, key).toMatch(CYRILLIC)
    }
  })

  it('has Russian category labels', () => {
    const categories = (ru as { categories: Record<string, Record<string, string>> }).categories

    for (const group of Object.values(categories)) {
      for (const label of Object.values(group)) {
        expect(label).toMatch(CYRILLIC)
        expect(label).toSatisfy((text: string) => !BAD_CHARS.test(text))
      }
    }
  })

  it('covers the bundled catalog at a useful scale', () => {
    expect(entries('skill').length).toBeGreaterThanOrEqual(300)
    expect(entries('plugin').length).toBeGreaterThanOrEqual(900)
    expect(entries('mcp').length).toBeGreaterThanOrEqual(60)
    expect(entries('tool').length).toBeGreaterThanOrEqual(70)
    expect(entries('toolset').length).toBeGreaterThanOrEqual(50)
  })
})
