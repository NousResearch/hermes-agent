// @vitest-environment node
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { compile } from '@tailwindcss/node'
import { describe, expect, it } from 'vitest'

const SRC = dirname(fileURLToPath(import.meta.url))

/**
 * Every inline reference is `class="ref"` plus an optional `data-ref="<kind>"`,
 * and styles.css owns the colour (see the INLINE REFERENCES block). The kind
 * rules (`[data-ref='agent']`, `[data-ref='skill']`, …) and the `.ref` fallback
 * share specificity (0,1,0), so the cascade is decided by source order alone:
 * a fallback declaring `--ref-color` AFTER the kind rules silently repaints
 * every kinded reference — a Bot Mode @mention, a `/skill`, an `@file:` — in
 * the raw primary (#114612). This resolves the cascade for a real element the
 * way the browser does, so the invariant is about the rendered colour, not
 * about where a line happens to sit in the file.
 */
interface Rule {
  body: string
  selectors: string[]
}

/** Unlayered top-level `selector { … }` rules, in source order. */
async function topLevelRules(): Promise<Rule[]> {
  const { build } = await compile('@import "../../../styles.css";\n', { base: SRC, onDependency() {} })
  const css = build([]).replace(/\/\*[\s\S]*?\*\//g, '')
  const rules: Rule[] = []
  let depth = 0
  let start = 0

  for (let i = 0; i < css.length; i += 1) {
    if (css[i] === '{') {
      if (depth === 0) {
        start = i
      }

      depth += 1
    } else if (css[i] === '}') {
      depth -= 1

      if (depth === 0) {
        const prelude = css.slice(css.lastIndexOf('}', start) + 1, start).trim()

        if (!prelude.startsWith('@')) {
          rules.push({ body: css.slice(start + 1, i), selectors: prelude.split(',') })
        }
      }
    }
  }

  return rules
}

/** Specificity of one compound selector against `<x class="ref" data-ref={kind}>`, or null when it does not match. */
function matches(selector: string, kind: string | null): number | null {
  const compound = selector.trim()
  const parts = compound.match(/\.ref|\[data-ref(?:=['"]?([\w-]+)['"]?)?\]/g)

  if (!parts || parts.join('') !== compound) {
    return null // combinators and other classes are outside the reference contract
  }

  for (const part of parts) {
    if (part === '.ref') {
      continue
    }

    const wanted = /=['"]?([\w-]+)/.exec(part)?.[1]

    if (kind === null || (wanted !== undefined && wanted !== kind)) {
      return null
    }
  }

  return parts.length // one class or attribute each
}

/** The `--ref-color` the browser resolves for `<span class="ref" data-ref={kind}>`. */
function resolvedRefColor(rules: Rule[], kind: string | null): string | undefined {
  let winner: { specificity: number; value: string } | undefined

  for (const rule of rules) {
    const specificity = Math.max(-1, ...rule.selectors.map(s => matches(s, kind) ?? -1))
    // Tailwind emits a pre-`color-mix()` fallback first and the real value
    // inside a nested `@supports`; the browser this ships in takes the latter.
    const value = [...rule.body.matchAll(/--ref-color:\s*([^;]+);/g)].at(-1)?.[1].trim()

    // Equal specificity: the later rule wins.
    if (specificity >= 0 && value && (!winner || specificity >= winner.specificity)) {
      winner = { specificity, value }
    }
  }

  return winner?.value
}

describe('inline reference colour cascade', () => {
  it('lets every kind rule outrank the bare .ref fallback', async () => {
    const rules = await topLevelRules()

    // A kinded reference resolves to ITS hue, never to the raw primary.
    expect(resolvedRefColor(rules, 'agent')).toMatch(/^color-mix\(in srgb, var\(--ui-accent\)/)
    expect(resolvedRefColor(rules, 'human')).toMatch(/^color-mix\(in srgb, var\(--ui-warm\)/)
    expect(resolvedRefColor(rules, 'broadcast')).toMatch(/^color-mix\(in srgb, var\(--foreground\)/)
    expect(resolvedRefColor(rules, 'skill')).toMatch(/^color-mix\(in srgb, var\(--ui-warm\)/)
    expect(resolvedRefColor(rules, 'file')).toBe('var(--ui-text-secondary)')
  })

  it('keeps the primary link colour for an unkinded reference', async () => {
    const rules = await topLevelRules()

    expect(resolvedRefColor(rules, null)).toBe('var(--dt-primary)')
  })
})
