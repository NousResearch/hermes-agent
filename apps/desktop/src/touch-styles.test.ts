// @vitest-environment node
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const STYLES = readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'styles.css'), 'utf8').replace(
  /\/\*[\s\S]*?\*\//g,
  ''
)

const COARSE_POINTER_MEDIA = /@media\s*\(\s*hover\s*:\s*none\s*\)\s*,\s*\(\s*pointer\s*:\s*coarse\s*\)\s*\{/g
const BROWSER_HOST = "html[data-hermes-desktop-host='browser']"

/** The text between the brace at `open` and its matching close. */
function blockBody(css: string, open: number): string {
  let depth = 0

  for (let index = open; index < css.length; index += 1) {
    if (css[index] === '{') {
      depth += 1
    } else if (css[index] === '}' && (depth -= 1) === 0) {
      return css.slice(open + 1, index)
    }
  }

  throw new Error('Unbalanced block in styles.css')
}

/** Every selector of the block's own rules, split on commas outside parentheses. */
function ruleSelectors(body: string): string[] {
  const selectors: string[] = []
  let depth = 0
  let prelude = ''

  for (const char of body) {
    if (char === '{') {
      if (depth === 0) {
        let parens = 0
        let current = ''

        for (const part of prelude) {
          parens += part === '(' ? 1 : part === ')' ? -1 : 0

          if (part === ',' && parens === 0) {
            selectors.push(current.trim())
            current = ''
          } else {
            current += part
          }
        }

        selectors.push(current.trim())
        prelude = ''
      }

      depth += 1
    } else if (char === '}') {
      depth -= 1
    } else if (depth === 0) {
      prelude += char
    }
  }

  return selectors
}

describe('touch styles', () => {
  // Finger-sized controls and always-visible message actions are the browser-hosted Desktop's
  // touch layout. Native Desktop on a coarse-pointer device keeps its compact hover layout, so
  // every rule under the coarse-pointer query is scoped to the browser host.
  it('applies coarse-pointer rules only in the browser-hosted Desktop', () => {
    const blocks = [...STYLES.matchAll(COARSE_POINTER_MEDIA)].map(match =>
      blockBody(STYLES, match.index + match[0].length - 1)
    )

    expect(blocks.length).toBeGreaterThan(0)

    for (const selector of blocks.flatMap(ruleSelectors)) {
      expect(selector.startsWith(BROWSER_HOST), selector).toBe(true)
    }
  })

  // Text inside a clickable row of an assistant message (a button around a code
  // span) is part of the control in native Desktop, as on main: the rule that
  // makes paragraphs and code selectable even there is the browser host's.
  it('makes paragraphs and code inside clickable assistant rows selectable only in the browser-hosted Desktop', () => {
    const messageText = /\[data-slot='aui_assistant-message-content'\]\s+(?::is\([^)]*\b(?:p|pre|code)\b[^)]*\)|(?:p|pre|code)\b)/
    const selectors = ruleSelectors(STYLES).filter(selector => messageText.test(selector))

    expect(selectors.length).toBeGreaterThan(0)

    for (const selector of selectors) {
      expect(selector.startsWith(BROWSER_HOST), selector).toBe(true)
    }
  })
})
