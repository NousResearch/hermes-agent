import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

// #95089 — iPadOS Safari hardware-keyboard contact AutoFill bar.
//
// The visible rich composer is a contentEditable div (the sr-only
// ComposerPrimitive.Input textarea is aria-hidden and never focused, so its
// autoComplete="off" does nothing for Safari). WebKit classifies focused
// editors heuristically; these assertions lock the suppression metadata onto
// the VISIBLE, focusable editor so a refactor can't drop it.

function readComposerSource(): string {
  const here = dirname(fileURLToPath(import.meta.url))

  return readFileSync(resolve(here, './index.tsx'), 'utf8')
}

function extractEditorBlock(src: string): string {
  // The visible editor's opening tag runs from its aria-disabled anchor to
  // the contentEditable prop — capture everything WebKit sees on that div.
  const start = src.indexOf("aria-disabled={inputDisabled ? true : undefined}")

  if (start === -1) {
    new Error('composer editor anchor missing')
  }

  const end = src.indexOf('contentEditable={', start)

  if (end === -1 || end <= start) {
    new Error('editor block end missing')
  }

  return src.slice(start, end)
}

describe('rich composer AutoFill suppression (#95089)', () => {
  const src = readComposerSource()
  const editorBlock = extractEditorBlock(src)

  it('marks the visible editor with password-manager opt-outs', () => {
    expect(editorBlock).toContain('data-1p-ignore=""')
    expect(editorBlock).toContain('data-lpignore="true"')
    expect(editorBlock).toContain('data-composer-rich-input=""')
  })

  it('opts the composer form out of autocomplete (form default overrides field hints in Safari)', () => {
    const rootIdx = src.indexOf('<ComposerPrimitive.Root')

    if (rootIdx === -1) {
      new Error('composer form primitive missing')
    }

    expect(src.slice(rootIdx, rootIdx + 600)).toMatch(/autoComplete="off"/)
  })

  it('documents why the attributes exist (refactor guard)', () => {
    expect(src).toContain('#95089')
    expect(src).toMatch(/WebKit AutoFill suppression/)
  })
})
