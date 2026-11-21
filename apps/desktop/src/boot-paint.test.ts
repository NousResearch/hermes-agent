// The first frame of every opaque window. `ready-to-show` waits for a paint, so
// whatever <body> resolves to before the module graph executes is what the user
// sees: with the stylesheet's light `:root` seeds that is rgb(248,249,254) across
// the whole window for as long as the bundle takes (measured at 100-200ms on a
// warm machine, in BOTH the Electron app and the browser-hosted webapp, which is
// built from this same index.html).
//
// The contract under test is the handshake between the boot <script> and the boot
// <style> in ../index.html: the script publishes the themed colour and a marker
// attribute, the rule paints <body> with it while the marker stands, and the rule
// retires the instant applyTheme() publishes the real theme.
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const APP = join(dirname(fileURLToPath(import.meta.url)), '..')
const INDEX = readFileSync(join(APP, 'index.html'), 'utf8')
const STYLES = readFileSync(join(APP, 'src/styles.css'), 'utf8')

const withoutComments = (css: string) => css.replace(/\/\*[\s\S]*?\*\//g, '')

/** The head block between `<tag>` and `</tag>`, with its comments stripped. */
function headBlock(tag: 'script' | 'style'): string {
  const match = INDEX.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`))

  if (!match) {
    throw new Error(`index.html has no boot <${tag}>`)
  }

  return match[1]
}

interface Rule {
  selector: string
  declarations: string
}

/** Every rule at the top nesting level; an at-rule's own body is left unparsed. */
function topLevelRules(css: string): Rule[] {
  const rules: Rule[] = []
  let depth = 0
  let prelude = ''
  let open = -1

  for (let index = 0; index < css.length; index += 1) {
    const char = css[index]

    if (char === '{') {
      if (depth === 0) {
        open = index
      }

      depth += 1
    } else if (char === '}') {
      depth -= 1

      if (depth === 0) {
        rules.push({ selector: prelude.trim(), declarations: css.slice(open + 1, index).trim() })
        prelude = ''
      }
    } else if (depth === 0) {
      prelude += char
    }
  }

  return rules
}

/** The body of every `@layer name { ... }` block, so its own rules can be read. */
function layerBodies(css: string): string[] {
  return [...css.matchAll(/@layer\s+[\w\s,]+\{/g)].map(match => {
    // Start on the brace itself: the layer's body is then the one top-level block.
    const [layer] = topLevelRules(css.slice(match.index! + match[0].length - 1))

    return layer?.declarations ?? ''
  })
}

/** The boot rule that paints <body>, as the browser would apply it. */
function bootBodyRule(): Rule {
  const rules = topLevelRules(withoutComments(headBlock('style')))
  const painters = rules.filter(rule => /(^|\s)body$/.test(rule.selector) && /background\s*:/.test(rule.declarations))
  expect(painters).toHaveLength(1)

  return painters[0]
}

/** The custom property the rule paints with. */
function paintedVariable(rule: Rule): string {
  const match = rule.declarations.match(/var\(\s*(--[\w-]+)\s*\)/)

  if (!match) {
    throw new Error(`the boot rule paints no custom property: ${rule.declarations}`)
  }

  return match[1]
}

/** Run the head script exactly as the browser does, before any bundle code exists. */
function runBootScript() {
  new Function(headBlock('script'))()
}

const isDark = (hex: string) => [1, 3, 5].reduce((sum, at) => sum + parseInt(hex.slice(at, at + 2), 16), 0) / 3 < 128

describe('boot pre-paint', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  beforeEach(() => {
    window.localStorage.clear()
    const root = document.documentElement
    root.removeAttribute('style')

    for (const name of root.getAttributeNames()) {
      if (name.startsWith('data-hermes')) {
        root.removeAttribute(name)
      }
    }
  })

  it('paints <body> with the stored theme background before the bundle runs', () => {
    window.localStorage.setItem('hermes-boot-background', '#080808')
    window.localStorage.setItem('hermes-boot-color-scheme', 'dark')

    runBootScript()

    const rule = bootBodyRule()
    // The selector the script's own DOM writes have to satisfy — not the
    // attribute names spelled out again here.
    expect(document.querySelector(rule.selector)).toBe(document.body)
    expect(document.documentElement.style.getPropertyValue(paintedVariable(rule))).toBe('#080808')
    expect(document.documentElement.style.colorScheme).toBe('dark')
  })

  it.each([
    { dark: true, scheme: 'dark' },
    { dark: false, scheme: 'light' }
  ])('follows the host scheme when no theme is stored yet (dark: $dark)', ({ dark, scheme }) => {
    // jsdom answers no media queries at all, and the script's fallback branch is
    // only reachable through one — a first launch, before any theme is stored.
    vi.stubGlobal('matchMedia', (query: string) => ({ matches: dark && query.includes('dark') }))

    runBootScript()

    const rule = bootBodyRule()
    expect(document.querySelector(rule.selector)).toBe(document.body)
    const colour = document.documentElement.style.getPropertyValue(paintedVariable(rule))
    expect(colour).toMatch(/^#[0-9a-f]{6}$/i)
    expect(isDark(colour)).toBe(dark)
    expect(document.documentElement.style.colorScheme).toBe(scheme)
  })

  it('retires as soon as applyTheme publishes the real mode', () => {
    runBootScript()
    const rule = bootBodyRule()

    // applyTheme() sets this in the same synchronous block as the brand seeds
    // that make the stylesheet's own body background correct.
    document.documentElement.dataset.hermesMode = 'dark'

    expect(document.querySelector(rule.selector)).toBeNull()
  })

  it('never paints over the window glass material', () => {
    runBootScript()
    const rule = bootBodyRule()

    // Under glass <body> is the single translucent painter; an opaque boot fill
    // there would hide the native material instead of hiding a flash.
    document.documentElement.setAttribute('data-hermes-glass', '')

    expect(document.querySelector(rule.selector)).toBeNull()
  })

  it('outranks the stylesheet: the boot rule is unlayered, styles.css paints <body> in a layer', () => {
    // This is the whole reason the fix needs no !important and no specificity
    // games — and the reason moving either side would bring the flash back.
    expect(withoutComments(headBlock('style'))).not.toMatch(/@layer/)

    const layered = layerBodies(withoutComments(STYLES))
      .flatMap(topLevelRules)
      .filter(rule => rule.selector === 'body' && /background\s*:/.test(rule.declarations))

    expect(layered).toHaveLength(1)
  })
})
