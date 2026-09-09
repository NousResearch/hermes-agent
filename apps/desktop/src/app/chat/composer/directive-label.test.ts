import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { hermesDirectiveFormatter } from '@/components/assistant-ui/directive-text'

import { classify } from './hooks/use-at-completions'
import { useComposerTrigger } from './hooks/use-composer-trigger'
import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'

/** A row exactly as tui_gateway's complete.path emits it, run through the
 *  real classify() the popover uses. */
function backendRow(text: string, display: string, meta: string): Unstable_TriggerItem {
  const c = classify({ text, display, meta })

  return {
    id: `${text}|0`,
    type: c.type,
    label: c.display,
    metadata: { icon: c.type, display: c.display, meta: c.meta, rawText: text, insertId: c.insertId }
  }
}

function typed(text: string) {
  const editor = document.createElement('div')

  editor.contentEditable = 'true'
  editor.dataset.slot = RICH_INPUT_SLOT
  document.body.append(editor)
  editor.append(document.createTextNode(text))

  const range = document.createRange()

  range.selectNodeContents(editor)
  range.collapse(false)

  const sel = window.getSelection()

  sel?.removeAllRanges()
  sel?.addRange(range)

  const editorRef = { current: editor as HTMLDivElement | null }

  const { result } = renderHook(() =>
    useComposerTrigger({
      at: { adapter: null, loading: false },
      draftRef: { current: text },
      editorRef,
      requestMainFocus: vi.fn(),
      setComposerText: vi.fn(),
      slash: { adapter: null, loading: false }
    })
  )

  act(() => result.current.refreshTrigger())

  return { editor, result }
}

/** The label the sent message renders for a committed draft. */
function sentLabel(draft: string) {
  return hermesDirectiveFormatter
    .parse(draft)
    .filter((s): s is Extract<typeof s, { kind: 'mention' }> => s.kind === 'mention')
    .map(s => s.label)
    .join(',')
}

describe('one label per reference, on every surface', () => {
  it('the popover row, the committed chip, and the sent chip all read the same', () => {
    const cases = [
      { text: '@folder:apps/desktop/', display: 'desktop/', meta: 'dir' },
      { text: '@file:apps/desktop/src/main.tsx', display: 'main.tsx', meta: 'apps/desktop/src' },
      { text: '@folder:apps/desktop/src/', display: 'src/', meta: 'dir' }
    ]

    for (const entry of cases) {
      const item = backendRow(entry.text, entry.display, entry.meta)
      const { editor, result } = typed('@desk')

      act(() => result.current.replaceTriggerWithChip(item))

      const row = String((item.metadata as { display: string }).display)
      const chip = editor.querySelector('[data-ref-text]')?.textContent ?? ''

      expect(chip).toBe(row)
      expect(sentLabel(composerPlainText(editor))).toBe(row)
    }
  })

  it('an agent mention inserts as plain @name text, never an @simple: chip', () => {
    // Rows exactly as complete.path emits profile mentions (colon-less text).
    const item = backendRow('@mr-tester', '@mr-tester', 'agent profile')

    expect(hermesDirectiveFormatter.serialize(item)).toBe('@mr-tester')

    const { editor, result } = typed('@mr-')

    act(() => result.current.replaceTriggerWithChip(item))

    const plain = composerPlainText(editor)

    expect(plain).toContain('@mr-tester')
    expect(plain).not.toContain('@simple:')
  })

  it('a folder pick reads as its path, not a bare basename', () => {
    // `src` and `desktop` repeat all over a repo — the row you picked said
    // where it was, and the chip has to keep saying it.
    const item = backendRow('@folder:apps/desktop/', 'desktop/', 'dir')

    expect(item.label).toBe('apps/desktop/')

    const { editor, result } = typed('@desk')

    act(() => result.current.replaceTriggerWithChip(item))

    expect(editor.querySelector('[data-ref-text]')?.textContent).toBe('apps/desktop/')
  })

  it('Tab-descend leaves the live query, and the scope when there is one', () => {
    const { editor, result } = typed('@folder:desk')

    act(() =>
      result.current.replaceTriggerWithChip(backendRow('@folder:apps/desktop/', 'desktop/', 'dir'), {
        descend: true
      })
    )

    // Mid-browse the editor holds the live query, scope included — that's the
    // path being typed, not a label, and it's what the next completion reads.
    expect(composerPlainText(editor)).toBe('@folder:apps/desktop/')
  })

  it('a url still reads host + path on every surface', () => {
    const item = backendRow('@url:https://github.com/NousResearch/hermes-agent/pull/74533', '', '')
    const { editor, result } = typed('@gith')

    act(() => result.current.replaceTriggerWithChip(item))

    const expected = 'github.com/NousResearch/hermes-agent/pull/74533'

    expect(item.label).toBe(expected)
    expect(editor.querySelector('[data-ref-text]')?.textContent).toBe(expected)
    expect(sentLabel(composerPlainText(editor))).toBe(expected)
  })
})
