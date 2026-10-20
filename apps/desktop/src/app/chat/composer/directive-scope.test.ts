import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useComposerTrigger } from './hooks/use-composer-trigger'
import { pathifyRefs } from './path-refs'
import { composerPlainText, insertComposerContentsAtCaret, RICH_INPUT_SLOT } from './rich-editor'
import { detectTrigger, openDirectiveScope, textBeforeCaret } from './text-utils'
import { linkifyUrls } from './url-refs'

function folderItem(rel: string): Unstable_TriggerItem {
  const rawText = `@folder:${rel}/`

  return {
    id: `${rawText}|0`,
    type: 'folder',
    label: rel.split('/').filter(Boolean).pop() ?? rel,
    metadata: { icon: 'folder', display: `${rel}/`, meta: 'dir', rawText, insertId: `${rel}/` }
  }
}

/** Literally-typed text, caret `fromEnd` characters before the end. */
function typed(text: string, fromEnd = 0) {
  const editor = document.createElement('div')

  editor.contentEditable = 'true'
  editor.dataset.slot = RICH_INPUT_SLOT
  document.body.append(editor)

  const node = document.createTextNode(text)

  editor.append(node)

  const range = document.createRange()

  range.setStart(node, text.length - fromEnd)
  range.collapse(true)

  const sel = window.getSelection()

  sel?.removeAllRanges()
  sel?.addRange(range)

  return editor
}

function withTrigger(editor: HTMLDivElement, draft: string) {
  const editorRef = { current: editor as HTMLDivElement | null }

  const { result } = renderHook(() =>
    useComposerTrigger({
      at: { adapter: null, loading: false },
      draftRef: { current: draft },
      editorRef,
      requestMainFocus: vi.fn(),
      setComposerText: vi.fn(),
      slash: { adapter: null, loading: false }
    })
  )

  act(() => result.current.refreshTrigger())

  return result
}

/** The composer's paste handler, minus the clipboard plumbing. */
function paste(editor: HTMLDivElement, text: string) {
  insertComposerContentsAtCaret(editor, pathifyRefs(linkifyUrls(text)), openDirectiveScope(editor))
}

describe('directive scope is a browse mode, not text to maintain', () => {
  it('Tab-descend carries the scope down instead of dropping to a bare path', () => {
    const editor = typed('@folder:apps/deskt')
    const result = withTrigger(editor, '@folder:apps/deskt')

    expect(result.current.trigger).toMatchObject({ kind: '@', scope: 'folder', value: 'apps/deskt' })

    act(() => result.current.replaceTriggerWithChip(folderItem('apps/desktop'), { descend: true }))

    expect(composerPlainText(editor)).toBe('@folder:apps/desktop/')
  })

  it('Backspace climbs the path, then drops the whole scope', () => {
    const editor = typed('@folder:apps/desktop/')
    const result = withTrigger(editor, '@folder:apps/desktop/')

    act(() => result.current.ascendTriggerPath())
    expect(composerPlainText(editor)).toBe('@folder:apps/')

    act(() => result.current.refreshTrigger())
    act(() => result.current.ascendTriggerPath())
    expect(composerPlainText(editor)).toBe('@folder:')

    // The scope is one unit: Backspace drops it whole rather than nibbling
    // back through `:`, `r`, `e`, `d`, `l`, `o`, `f`.
    act(() => result.current.refreshTrigger())
    act(() => result.current.ascendTriggerPath())
    expect(composerPlainText(editor)).toBe('@')
  })

  it('leaves Backspace alone when there is no scope and no path', () => {
    const editor = typed('@apps')
    const result = withTrigger(editor, '@apps')

    let handled = true

    act(() => {
      handled = result.current.ascendTriggerPath()
    })

    expect(handled).toBe(false)
  })

  it('a pick mid-message keeps the trailing prose and consumes the whole token', () => {
    const editor = typed('@folder:apps/deskt and some trailing words', 24)
    const result = withTrigger(editor, '@folder:apps/deskt and some trailing words')

    act(() => result.current.replaceTriggerWithChip(folderItem('apps/desktop')))

    expect(composerPlainText(editor)).toBe('@folder:`apps/desktop/` and some trailing words')
    expect(editor.querySelector('[data-ref-kind="folder"]')).not.toBeNull()
  })

  it('pasting into an open @url: scope consumes it instead of stacking', () => {
    const editor = typed('refer to @url:')

    paste(editor, 'https://github.com/NousResearch/hermes-agent/pull/74533')

    expect(composerPlainText(editor)).toBe('refer to @url:`https://github.com/NousResearch/hermes-agent/pull/74533`')
    expect(editor.textContent).not.toContain('@url:@url:')
  })

  it('a normal paste with no open scope is untouched', () => {
    const editor = typed('look at ')

    paste(editor, 'https://example.com/x')

    expect(composerPlainText(editor)).toBe('look at @url:`https://example.com/x`')
  })

  it('scope parsing leaves an unscoped @ query alone', () => {
    expect(detectTrigger('@apps/desk')).toMatchObject({ kind: '@', value: 'apps/desk' })
    expect(detectTrigger('@apps/desk')?.scope).toBeUndefined()
  })

  it('openDirectiveScope only fires on an EMPTY scope', () => {
    // The count is what a paste consumes: `@url:` is 5 characters of syntax
    // the user never typed and shouldn't be left holding.
    expect(openDirectiveScope(typed('@url:'))).toBe(5)
    expect(openDirectiveScope(typed('@url:https://x.com'))).toBe(0)
    expect(openDirectiveScope(typed('plain text'))).toBe(0)
  })

  it('chips stay atomic to scope detection', () => {
    const editor = typed('@folder:apps/desktop/')
    const result = withTrigger(editor, '@folder:apps/desktop/')

    act(() => result.current.replaceTriggerWithChip(folderItem('apps/desktop')))

    // A committed chip is one object-replacement char, so a fresh `@` typed
    // after it opens an unscoped browse rather than inheriting the old scope.
    expect(detectTrigger(`${textBeforeCaret(editor)}@`)?.scope).toBeUndefined()
  })
})
