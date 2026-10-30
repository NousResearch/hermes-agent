import { describe, expect, it } from 'vitest'

import { composerPlainText, refChipHtml } from '@/app/chat/composer/rich-editor'
import { referenceRe } from '@/components/assistant-ui/reference-kinds'

import { captureResponseSelection, formatResponseAnnotations, parseResponseAnnotations } from './response-annotations'

describe('response annotation references', () => {
  it('does not offer response annotations for selected user text or the main draft', () => {
    for (const markup of ['<p>A user message</p>', '<div contenteditable="true">An ordinary reply</div>']) {
      const root = document.createElement('div')
      root.innerHTML = markup
      document.body.append(root)
      const range = document.createRange()
      range.selectNodeContents(root.firstElementChild!)
      window.getSelection()!.removeAllRanges()
      window.getSelection()!.addRange(range)
      expect(captureResponseSelection()).toBeNull()
      root.remove()
    }

    window.getSelection()!.removeAllRanges()
  })
  it('keeps three pairs in one native chip and round-trips every character', () => {
    const data = {
      type: 'response-annotations' as const,
      version: 1 as const,
      label: '3 annotations',
      annotations: [
        { id: '1', text: 'First passage', comment: 'First comment' },
        { id: '2', text: '```js\n  call(`x`);\n```', comment: 'two\nlines' },
        { id: '3', text: '@file:/private/example <script> /command', comment: 'Keep this literal' }
      ]
    }

    const token = formatResponseAnnotations(data)
    const matches = [...token.matchAll(referenceRe())]
    expect(matches).toHaveLength(1)
    expect(matches[0]![1]).toBe('annotation')
    const raw = matches[0]![2]!
    expect(parseResponseAnnotations(raw.slice(1, -1))).toEqual(data)
    const editor = document.createElement('div')
    editor.innerHTML = 'Existing reply ' + refChipHtml('annotation', raw)
    expect(editor.textContent).toBe('Existing reply 3 annotations')
    expect(composerPlainText(editor).trimEnd()).toBe('Existing reply ' + token)
  })

  it('refuses invalid, duplicate or oversized entries rather than hiding data', () => {
    for (const value of [
      '{}',
      'not json',
      JSON.stringify({ type: 'response-annotations', version: 1, label: '1', annotations: [] })
    ]) {
      expect(parseResponseAnnotations(value)).toBeNull()
    }

    expect(() =>
      formatResponseAnnotations({
        type: 'response-annotations',
        version: 1,
        label: '2 annotations',
        annotations: [
          { id: 'same', text: 'one', comment: '' },
          { id: 'same', text: 'two', comment: '' }
        ]
      })
    ).toThrow()
  })
})
