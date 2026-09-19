import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { act, renderHook } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { rememberDesktopCommandsCatalog } from '@/lib/desktop-slash-commands'

import { composerPlainText, renderComposerContents, RICH_INPUT_SLOT } from '../rich-editor'

import { collapseSelectionOntoTrigger, rebuildAroundCaret, useComposerTrigger } from './use-composer-trigger'

beforeEach(() => {
  rememberDesktopCommandsCatalog({
    commands: {
      '/goal': { argument_mode: 'mixed', desktop: null },
      '/personality': { argument_mode: 'options', desktop: null }
    }
  })
})

afterEach(() => {
  rememberDesktopCommandsCatalog(undefined)
})

/** A live contentEditable seeded with `text`, caret parked at the end. */
function mountEditor(text: string) {
  const editor = document.createElement('div')
  editor.dataset.slot = RICH_INPUT_SLOT
  editor.contentEditable = 'true'
  document.body.append(editor)
  renderComposerContents(editor, text)

  const range = document.createRange()
  range.selectNodeContents(editor)
  range.collapse(false)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)

  return editor
}

const item = (command: string, group = 'Skills'): Unstable_TriggerItem => ({
  id: command,
  type: 'slash',
  label: command.slice(1),
  metadata: { command, display: command, meta: '', group, action: '', rawText: command }
})

function mountTrigger(editor: HTMLDivElement, items: Unstable_TriggerItem[]) {
  const editorRef = createRef<HTMLDivElement>() as { current: HTMLDivElement | null }
  editorRef.current = editor

  const draftRef = { current: composerPlainText(editor) }

  const adapter: Unstable_TriggerAdapter = {
    categories: () => [],
    categoryItems: () => [],
    search: () => items
  }

  const setComposerText = vi.fn()

  const hook = renderHook(() =>
    useComposerTrigger({
      at: { adapter: null, loading: false },
      draftRef,
      editorRef,
      requestMainFocus: vi.fn(),
      setComposerText,
      slash: { adapter, loading: false }
    })
  )

  return { draftRef, hook, setComposerText }
}

describe('useComposerTrigger — slash anywhere in the prompt', () => {
  it('opens the completion list for a slash typed mid-message', () => {
    const editor = mountEditor('please run /cle')
    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.trigger).toMatchObject({ kind: '/', inline: true, query: 'cle' })
    expect(hook.result.current.triggerItems).toHaveLength(1)
  })

  it('inserts the picked skill inline and keeps the surrounding prose intact', () => {
    const editor = mountEditor('please run /cle')
    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())
    act(() => hook.result.current.replaceTriggerWithChip(item('/clean')))

    // The `/cle` the user typed is replaced by the full command; "please run"
    // in front of it survives untouched.
    expect(composerPlainText(editor)).toBe('please run /clean ')
  })

  it('offers only skills mid-message, not app commands', () => {
    // `/model` and `/new` act on the app — meaningless as a reference in prose.
    const editor = mountEditor('please run /')
    const { hook } = mountTrigger(editor, [item('/clean'), item('/model', 'Commands'), item('/new', 'Commands')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.triggerItems.map(i => i.label)).toEqual(['clean'])
  })

  it('still offers the full command set at the start of the prompt', () => {
    const editor = mountEditor('/')
    const { hook } = mountTrigger(editor, [item('/clean'), item('/model', 'Commands')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.triggerItems.map(i => i.label)).toEqual(['clean', 'model'])
  })

  it('still opens the list for a slash at the start of the prompt', () => {
    const editor = mountEditor('/cle')
    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.trigger).toMatchObject({ kind: '/', query: 'cle' })
    expect(hook.result.current.trigger?.inline).toBeUndefined()
  })

  it('leaves a mid-message file path alone', () => {
    const editor = mountEditor('open src/foo/bar')
    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.trigger).toBeNull()
  })

  it('opens the list for a second slash after a leading command', () => {
    // `/work /cle`: the command regex's argument tail would otherwise swallow
    // `/cle` as an argument to `/work`, and a no-arg command suppresses the
    // popover — so every slash after the first went dead.
    const editor = mountEditor('/work /cle')
    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.trigger).toMatchObject({ kind: '/', inline: true, query: 'cle' })
    expect(hook.result.current.triggerItems).toHaveLength(1)
  })

  it('inserts the second command without disturbing the first', () => {
    const editor = mountEditor('/work rewrite the composer /cle')
    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())
    act(() => hook.result.current.replaceTriggerWithChip(item('/clean')))

    expect(composerPlainText(editor)).toBe('/work rewrite the composer /clean ')
  })
})

describe('useComposerTrigger — free-text slash arguments', () => {
  it('keeps a picked /goal command as editable text while retaining subcommand completion', () => {
    const editor = mountEditor('/go')
    const goal = item('/goal', 'Commands')
    const { hook } = mountTrigger(editor, [goal])

    act(() => hook.result.current.refreshTrigger())
    act(() => hook.result.current.replaceTriggerWithChip(goal))

    expect(composerPlainText(editor)).toBe('/goal ')
    expect(editor.querySelector('[data-slash-kind]')).toBeNull()
    expect(hook.result.current.trigger).not.toBeNull()
  })

  it('does not seal a multi-word /goal into a chip when the option list runs empty', () => {
    const editor = mountEditor('/goal finish the full prompt')
    const { hook } = mountTrigger(editor, [])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.slashFreeTextArgStage).toBe(true)
    expect(hook.result.current.commitTypedSlashDirective()).toBe(false)
    expect(composerPlainText(editor)).toBe('/goal finish the full prompt')
    expect(editor.querySelector('[data-slash-kind]')).toBeNull()
  })

  it('treats the default highlight as a suggestion until the user arrows to a row', () => {
    const editor = mountEditor('/goal stat')
    const { hook } = mountTrigger(editor, [item('/goal status', 'Options')])

    act(() => hook.result.current.refreshTrigger())
    expect(hook.result.current.triggerActiveExplicit).toBe(false)

    act(() => hook.result.current.moveTriggerActive(1))
    expect(hook.result.current.triggerActiveExplicit).toBe(true)
  })

  it('drops a deliberate selection once the query moves on', () => {
    const editor = mountEditor('/goal stat')
    const { hook } = mountTrigger(editor, [item('/goal status', 'Options')])

    act(() => hook.result.current.refreshTrigger())
    act(() => hook.result.current.moveTriggerActive(1))

    renderComposerContents(editor, '/goal start the migration')
    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.triggerActiveExplicit).toBe(false)
  })

  it('keeps a multi-word /resume search typeable instead of firing the picker action', () => {
    // The session list always ends in a "Browse all sessions…" action row, so
    // an accept here doesn't insert a chip — it empties the composer and opens
    // the overlay, taking the half-typed query with it.
    const editor = mountEditor('/resume my new')
    const { hook } = mountTrigger(editor, [item('/resume', 'Sessions')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.slashFreeTextArgStage).toBe(true)
    expect(hook.result.current.commitTypedSlashDirective()).toBe(false)
    expect(composerPlainText(editor)).toBe('/resume my new')
  })

  it('still commits a fully typed finite option as one directive chip', () => {
    const editor = mountEditor('/personality creative')
    const { hook } = mountTrigger(editor, [])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.slashFreeTextArgStage).toBe(false)
    act(() => {
      expect(hook.result.current.commitTypedSlashDirective()).toBe(true)
    })
    expect(composerPlainText(editor)).toBe('/personality creative ')
    expect(editor.querySelector('[data-slash-kind]')?.getAttribute('data-ref-text')).toBe('/personality creative')
  })
})

describe('collapseSelectionOntoTrigger + rebuildAroundCaret (non-collapsed selection)', () => {
  /** Ctrl+A shape: whole editor selected. Fires no input event, so the
   *  composer never normalizes it — the accept path has to. */
  function selectAllIn(editor: HTMLElement) {
    const range = document.createRange()
    range.selectNodeContents(editor)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(range)
  }

  it('collapses a whole-editor selection onto the end of the trigger token', () => {
    const editor = mountEditor('/ne')
    selectAllIn(editor)

    expect(collapseSelectionOntoTrigger(editor, '/', 'ne')).toBe(true)

    const selection = window.getSelection()!
    expect(selection.getRangeAt(0).collapsed).toBe(true)
  })

  it('re-marks emptiness after a fragment rebuild so the placeholder stops painting', () => {
    // The rebuild renders an empty prefix (which sets data-empty), then appends
    // the chip + suffix — appends that fire no input event. Without the re-mark
    // the placeholder hint painted over the committed content.
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.contentEditable = 'true'
    document.body.append(editor)
    renderComposerContents(editor, '/new')
    selectAllIn(editor)

    expect(collapseSelectionOntoTrigger(editor, '/', 'new')).toBe(true)

    const chip = document.createElement('span')
    chip.contentEditable = 'false'
    chip.dataset.refText = '/new '
    chip.append(document.createTextNode('/new '))
    const chipFragment = document.createDocumentFragment()
    chipFragment.append(chip)

    rebuildAroundCaret(editor, 4, chipFragment)

    expect(composerPlainText(editor)).toBe('/new ')
    expect(editor.dataset.empty).toBeUndefined()
  })
})

describe('useComposerTrigger — chip survival (the plaintext-demotion bug class)', () => {
  it('keeps a leading command pill through a Backspace path-ascend', () => {
    // The reported repro: `/work @folder…` then Backspace — both chips went
    // plaintext because ascend re-rendered the whole editor from text.
    const editor = mountEditor('/work @Desktop/')
    const { hook } = mountTrigger(editor, [])

    expect(editor.querySelector('[data-slash-kind]')).not.toBeNull()

    act(() => hook.result.current.refreshTrigger())
    expect(hook.result.current.trigger).toMatchObject({ kind: '@', query: 'Desktop/' })

    let ran = false
    act(() => {
      ran = hook.result.current.ascendTriggerPath()
    })

    expect(ran).toBe(true)
    expect(composerPlainText(editor)).toBe('/work @')
    expect(editor.querySelector('[data-slash-kind]')).not.toBeNull()
  })

  it('keeps a leading command pill when a folder pick commits its ref chip', () => {
    const editor = mountEditor('/work @Desk')

    const folder: Unstable_TriggerItem = {
      id: 'folder:Desktop',
      type: 'folder',
      label: 'Desktop',
      metadata: { rawText: '@folder:Desktop', insertId: 'Desktop' }
    }

    const { hook } = mountTrigger(editor, [folder])

    act(() => hook.result.current.refreshTrigger())
    act(() => hook.result.current.replaceTriggerWithChip(folder))

    expect(composerPlainText(editor)).toBe('/work @folder:`Desktop` ')
    expect(editor.querySelector('[data-slash-kind]')).not.toBeNull()
    expect(editor.querySelector('[data-ref-kind="folder"]')).not.toBeNull()
  })

  it('commits in place when Chromium has split the token across text nodes', () => {
    // Chromium fragments text nodes around contenteditable=false chips; the
    // commit path must span the fragments instead of bailing to a full
    // re-render.
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.contentEditable = 'true'
    document.body.append(editor)
    editor.append(document.createTextNode('please run /c'), document.createTextNode('le'))

    const caret = document.createRange()
    caret.setStart(editor.lastChild!, 2)
    caret.collapse(true)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(caret)

    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())
    act(() => hook.result.current.replaceTriggerWithChip(item('/clean')))

    expect(composerPlainText(editor)).toBe('please run /clean ')
    expect(editor.querySelector('[data-slash-kind]')).not.toBeNull()
  })

  it('replaces the token when the editor is selected-all (Ctrl+A) at accept time', () => {
    // Ctrl+A (native select-all) leaves a NON-COLLAPSED selection behind; it
    // fires no input event, so the composer never normalizes it. An accept
    // must treat the selected token as the thing being replaced — the rebuild
    // used to read the caret at the selection's document-order START (0),
    // re-appending the typed token after the chip.
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.contentEditable = 'true'
    document.body.append(editor)
    renderComposerContents(editor, '/ne')
    const all = document.createRange()
    all.selectNodeContents(editor)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(all)

    const { hook } = mountTrigger(editor, [item('/new', 'Commands')])

    act(() => hook.result.current.refreshTrigger())
    act(() => hook.result.current.moveTriggerActive(1))
    act(() => hook.result.current.moveTriggerActive(-1))
    act(() => hook.result.current.replaceTriggerWithChip(item('/new', 'Commands')))

    expect(composerPlainText(editor)).toBe('/new ')
    expect(editor.querySelector('[data-slash-kind]')).not.toBeNull()
    // The stale-empty regression: the rebuild set data-empty while wiping the
    // editor, and nothing re-marked it after the append — the placeholder hint
    // painted over the chip.
    expect(editor.dataset.empty).toBeUndefined()
  })

  it('replaces the token under select-all when prose precedes it', () => {
    // The selection's DOM offsets are (node, child-index) pairs, NOT character
    // offsets: with prose in front, the whole-editor selection's endOffset is
    // the number of child nodes, which used to be read as an absolute text
    // offset and made the token search conclude the selection didn't reach it.
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.contentEditable = 'true'
    document.body.append(editor)
    renderComposerContents(editor, 'please run /cle')
    const all = document.createRange()
    all.selectNodeContents(editor)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(all)

    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())
    act(() => hook.result.current.replaceTriggerWithChip(item('/clean')))

    expect(composerPlainText(editor)).toBe('please run /clean ')
  })
})
