import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

// Regression guard for #95089: on iPadOS Safari the keyboard bar classifies
// focused text surfaces heuristically and offers the native contact AutoFill
// card (name/address) over the composer — which already has its own in-app
// flow and no business receiving contact suggestions. Safari keys that offer
// off the field's autocomplete state, its name attribute, and (for fields in
// a form) the surrounding <form>'s autocomplete state.
//
// This is a static-analysis guard in the same category as an ESLint rule:
// it asserts every composer text surface carries the suppression attributes,
// so a refactor that drops one fails here rather than resurfacing the iOS
// AutoFill bar for users.
//
// Extraction strategy: composer tags span many lines and contain `>` inside
// JSX expression arrow functions, so tag-boundary regexing is fragile. We
// anchor on stable markers (data-slot={RICH_INPUT_SLOT}, ComposerPrimitive.*)
// and scan a window of surrounding source instead.

const COMPOSER_INDEX = resolve(__dirname, 'index.tsx')
const USER_EDIT_COMPOSER = resolve(
  __dirname,
  '../../../components/assistant-ui/thread/user-edit-composer.tsx'
)

const WINDOW = 2600

function sourceWindow(content: string, marker: string, where: string): string {
  const idx = content.indexOf(marker)
  expect(idx, `${where}: marker "${marker}" not found`).toBeGreaterThanOrEqual(0)
  return content.slice(Math.max(0, idx - WINDOW), idx + WINDOW)
}

// The visible rich editor is a contentEditable div. The WHATWG autofill
// algorithm only applies to input/select/textarea, so autoComplete= is not
// even defined there; suppression means no name= semantic handle plus the
// autocapitalize/autocorrect/spellcheck-off trio and the password-manager
// opt-out data attributes.
const EDITOR_REQUIRED = [
  'autoCapitalize="off"',
  'autoCorrect="off"',
  'spellCheck={false}',
  'data-1p-ignore=""',
  'data-composer-rich-input=""',
  'data-lpignore="true"'
]

describe('composer fields suppress native contact AutoFill (#95089)', () => {
  for (const [path, label] of [
    [COMPOSER_INDEX, 'chat/composer/index.tsx'],
    [USER_EDIT_COMPOSER, 'user-edit-composer.tsx']
  ] as const) {
    it(`${label}: rich editor carries suppression attributes`, () => {
      const win = sourceWindow(
        readFileSync(path, 'utf-8'),
        'data-slot={RICH_INPUT_SLOT}',
        `${label} rich editor`
      )

      for (const attr of EDITOR_REQUIRED) {
        expect(win.includes(attr), `${label} rich editor: missing ${attr}`).toBe(true)
      }

      // A name attribute would hand Safari's contact AutoFill heuristics a
      // semantic handle ("name", "email", "search", …); the editor must not
      // give it one.
      expect(/\sname=/.test(win), `${label} rich editor: must not carry name=`).toBe(false)
    })

    it(`${label}: hidden textarea + form root carry autoComplete="off"`, () => {
      const content = readFileSync(path, 'utf-8')

      // The sr-only <textarea> under ComposerPrimitive.Input participates in
      // the form submit binding — the WHATWG autofill path applies to it. It
      // must not declare its own name= (assistant-ui's ComposerInput already
      // injects name="input" as a prop default; our markup adds none).
      const taWin = sourceWindow(content, '<ComposerPrimitive.Input', `${label} hidden textarea`)
      expect(taWin.includes('autoComplete="off"'), `${label} hidden textarea: missing autoComplete="off"`).toBe(
        true
      )
      // Slice to the actual <textarea …/> tag (the window may contain sibling
      // markup like icon components that legitimately use name=).
      const taTag = taWin.slice(
        taWin.indexOf('<textarea'),
        taWin.indexOf('/>', taWin.indexOf('<textarea')) + 2
      )
      expect(/\sname="/.test(taTag), `${label} hidden textarea: must not carry name="..."`).toBe(false)

      // The form wrapper: Safari consults the <form>'s autocomplete state for
      // fields inside it, so ComposerPrimitive.Root (which renders the form,
      // see @assistant-ui/react primitives/composer/ComposerRoot) must be
      // autocomplete="off" as well.
      const rootWin = sourceWindow(content, '<ComposerPrimitive.Root', `${label} form root`)
      expect(
        rootWin.includes('autoComplete="off"'),
        `${label}: ComposerPrimitive.Root missing autoComplete="off"`
      ).toBe(true)
    })
  }
})
