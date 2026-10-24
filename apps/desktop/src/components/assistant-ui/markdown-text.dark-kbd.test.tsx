import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

// #144072: Tailwind Typography's `.prose` palette styles markdown key caps
// with `var(--tw-prose-kbd)` (near-black) and a ring of
// `var(--tw-prose-kbd-shadows)`, and the dark palette's invert variants are
// never applied — `prose-invert` is not used and nothing re-points the tokens
// under `.dark`. Body text survives via the container's `text-foreground` and
// inline code has its own themed rule in this sheet, but the element-specific
// kbd rules still resolve against the light palette, so key caps are
// near-black on every dark theme. This mounts the real stylesheet and the real
// markdown component, so the cascade — not a regex over the source — decides
// what a key cap resolves to in dark mode.
const STYLES = resolve(dirname(fileURLToPath(import.meta.url)), '../../styles.css')

beforeAll(() => {
  const sheet = globalThis.document.createElement('style')
  sheet.textContent = readFileSync(STYLES, 'utf8')
  globalThis.document.head.appendChild(sheet)
})

afterEach(() => {
  cleanup()
  globalThis.document.documentElement.classList.remove('dark')
})

describe('markdown prose key caps in dark mode', () => {
  it('renders <kbd> and inline code inside the prose container', () => {
    const { container } = render(
      <MarkdownTextContent
        isRunning={false}
        text={'Press <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>N</kbd> then run `npm test` to check.'}
      />
    )

    // The markdown surface the fix scopes to: the .aui-md container with the
    // typography .prose class.
    const prose = container.querySelector('.aui-md.prose')
    expect(prose).not.toBeNull()

    // Raw inline <kbd> passes the sanitizer (default schema allows kbd) and
    // lands as a real element Typography can style.
    const keys = prose!.querySelectorAll('kbd')
    expect(keys.length).toBe(3)
    expect(keys[0]!.textContent).toBe('Ctrl')

    // Inline code span from backticks.
    expect(prose!.querySelector('code')?.textContent).toContain('npm test')
  })

  it('re-points the typography kbd ink to the app foreground in dark mode', () => {
    const { container } = render(
      <MarkdownTextContent isRunning={false} text={'Press <kbd>Ctrl</kbd> then run `npm test`.'} />
    )

    const kbd = container.querySelector('.aui-md.prose kbd')
    expect(kbd).not.toBeNull()

    globalThis.document.documentElement.classList.add('dark')

    const kbdStyle = getComputedStyle(kbd!)
    // The dark override in styles.css owns the paint: the near-black
    // typography light palette must be replaced by the app foreground mix.
    // jsdom keeps var() unresolved inside color-mix, so compare the exact
    // declared value. On main (no override) this is jsdom's default
    // 'rgb(0, 0, 0)' with no box-shadow at all.
    expect(kbdStyle.getPropertyValue('color').trim()).toBe(
      'color-mix(in srgb, var(--dt-foreground) 88%, transparent)'
    )
    const shadow = kbdStyle.getPropertyValue('box-shadow').trim()
    expect(shadow).toContain('color-mix(in srgb, var(--dt-foreground) 22%, transparent)')
    expect(shadow).toContain('0 3px 0')
  })

  it('keeps light mode untouched when dark is off', () => {
    const { container } = render(
      <MarkdownTextContent isRunning={false} text={'Press <kbd>Ctrl</kbd> then run `npm test`.'} />
    )

    globalThis.document.documentElement.classList.remove('dark')

    const kbdStyle = getComputedStyle(container.querySelector('.aui-md.prose kbd')!)
    // The override is scoped to `.dark`, so with the class off no app rule may
    // paint the key cap: the raw stylesheet the test mounts carries no `.prose`
    // rules (the typography CSS is emitted at build time), so an unscoped
    // override would be the only way this resolves to something.
    const kbdColor = kbdStyle.getPropertyValue('color').trim()
    expect(kbdColor).not.toContain('color-mix')
    expect(kbdStyle.getPropertyValue('box-shadow').trim()).toBe('')
  })
})
