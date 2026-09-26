import { describe, expect, it } from 'vitest'

import { type ExtraListStyle, listEdit, type ListKey } from './markdown-lists'

const ALL_STYLES = new Set<ExtraListStyle>(['letters', 'roman', 'outline', 'parentheses', 'symbols'])

// `|` marks the caret in both the input and the expected result.
function press(draft: string, key: ListKey, styles: ReadonlySet<ExtraListStyle> = new Set()) {
  const caret = draft.indexOf('|')
  const edit = listEdit(draft.replace('|', ''), caret, key, styles)

  return edit && `${edit.text.slice(0, edit.caret)}|${edit.text.slice(edit.caret)}`
}

describe('Shift+Enter on a list line', () => {
  it.each([
    ['1. first|', '1. first\n2. |'],
    ['9) first|', '9) first\n10) |'],
    ['09. first|', '09. first\n10. |'],
    ['- milk|', '- milk\n- |'],
    ['* milk|', '* milk\n* |'],
    ['- [x] done|', '- [x] done\n- [ ] |'],
    ['3. [ ] todo|', '3. [ ] todo\n4. [ ] |'],
    ['   - child|', '   - child\n   - |'],
    ['1. ab|cd', '1. ab\n2. |cd']
  ])('%j continues as %j', (draft, expected) => {
    expect(press(draft, 'newline')).toBe(expected)
  })

  it('ends the list on an empty item, leaving a blank line so Markdown ends it too', () => {
    expect(press('1. first\n2. |', 'newline')).toBe('1. first\n\n|')
    expect(press('1. first\n2. |\nnext', 'newline')).toBe('1. first\n|\nnext')
  })

  it('moves an empty child up a level instead of ending the whole list', () => {
    expect(press('1. parent\n   - child\n   - |', 'newline')).toBe('1. parent\n   - child\n2. |')
  })

  it('leaves plain text, prose that looks like a marker, the marker itself, and code alone', () => {
    for (const draft of ['hello|', 'Mr. Smith|', '1.5 hours|', '-dash|', '1|. first', '```\n1. code|']) {
      expect(press(draft, 'newline')).toBeNull()
    }
  })
})

describe('Tab and Shift+Tab nest list items', () => {
  it('nests under the item above at its content column, restarting in the same style', () => {
    expect(press('1. parent\n2. |', 'indent')).toBe('1. parent\n   1. |')
    expect(press('10. parent\n11. child|', 'indent')).toBe('10. parent\n    1. child|')
    expect(press('- parent\n- child|', 'indent')).toBe('- parent\n  - child|')
    expect(press('- [ ] parent\n- [ ] |', 'indent')).toBe('- [ ] parent\n  - [ ] |')
  })

  it('joins a child list that already exists instead of restarting it', () => {
    expect(press('1. parent\n   1. one\n2. |', 'indent')).toBe('1. parent\n   1. one\n   2. |')
  })

  it('outdents back into the parent list and continues its numbering', () => {
    expect(press('1. parent\n   1. child|', 'outdent')).toBe('1. parent\n2. child|')
    expect(press('1. parent\n   - a\n     - deep|', 'outdent')).toBe('1. parent\n   - a\n   - deep|')
  })

  it('keeps Tab inside the composer on a list line with nowhere to go', () => {
    expect(press('1. only|', 'indent')).toBe('1. only|')
    expect(press('1. top|', 'outdent')).toBe('1. top|')
  })

  it('keeps the caret on the same text when the marker changes width', () => {
    expect(press('9. parent\n10. ab|cd', 'indent')).toBe('9. parent\n   1. ab|cd')
  })
})

describe('Backspace', () => {
  it('clears an empty marker and keeps its indent', () => {
    expect(press('- a\n- |', 'backspace')).toBe('- a\n|')
    expect(press('1. a\n   - |', 'backspace')).toBe('1. a\n   |')
  })

  it('stays a normal Backspace once the item has text', () => {
    expect(press('- a|', 'backspace')).toBeNull()
  })
})

describe('extra list styles', () => {
  it('are plain text until turned on', () => {
    for (const draft of ['a. x|', 'iv. x|', '1.2. x|', '(1) x|', '• x|']) {
      expect(press(draft, 'newline')).toBeNull()
    }
  })

  it.each([
    ['a. x|', 'a. x\nb. |'],
    ['z) x|', 'z) x\naa) |'],
    ['(A) x|', '(A) x\n(B) |'],
    ['i. x|', 'i. x\nii. |'],
    ['IV. x|', 'IV. x\nV. |'],
    ['1.9. x|', '1.9. x\n1.10. |'],
    ['• x|', '• x\n• |'],
    ['h. x\ni. y|', 'h. x\ni. y\nj. |'],
    ['iv. x\nv. y|', 'iv. x\nv. y\nvi. |']
  ])('%j continues as %j', (draft, expected) => {
    expect(press(draft, 'newline', ALL_STYLES)).toBe(expected)
  })

  it('nest in their own style', () => {
    expect(press('b. parent\nc. |', 'indent', ALL_STYLES)).toBe('b. parent\n   a. |')
    expect(press('1.2. parent\n1.3. |', 'indent', ALL_STYLES)).toBe('1.2. parent\n     1.2.1. |')
  })

  it('does not read words at a line start as numerals', () => {
    expect(press('mix. it|', 'newline', ALL_STYLES)).toBeNull()
  })
})
