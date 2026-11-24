import { act, cleanup, render, waitFor } from '@testing-library/react'
import { type ReactNode, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $freshDraftKey,
  adoptNewSessionDraft,
  announceNewSessionDraftKey,
  clearSessionDraft,
  mainComposerScope,
  NEW_SESSION_DRAFT_KEY,
  rotateFreshDraftKey,
  stashSessionDraft
} from '@/store/composer'

import { markActiveComposer, requestComposerFocus } from '../focus'
import { caretOffsetInEditor, composerPlainText } from '../rich-editor'

import { useComposerDraft } from './use-composer-draft'

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => ({ setText: vi.fn() }) }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) => selector({ composer: { text: '' } }),
  useComposerRuntime: () => ({
    getState: () => ({ text: '' }),
    subscribe: () => () => undefined
  })
}))

/** The live main composer's editor, set by whichever EditableComposer last mounted. */
let editor: HTMLDivElement | null = null

/** A main composer with a real contenteditable, scoped to `scope` (null = fresh chat). */
function EditableComposer({ scope }: { scope: string | null }) {
  const draft = useComposerDraft({
    activeQueueSessionKey: scope,
    focusKey: null,
    inputDisabled: false,
    queueEditRef: { current: null },
    sessionId: scope ?? ''
  })

  return (
    <div
      contentEditable
      data-slot="composer-rich-input"
      ref={element => {
        draft.editorRef.current = element
        editor = element
      }}
      tabIndex={0}
    />
  )
}

/** The cold-start resume: the fresh chat typed `typed`, then the restore of
 *  `restored` was announced and the fresh chat's composer unmounted behind the
 *  restored chat's loading view (which mounts no composer). */
function startColdStartResume(typed: string, restored: string) {
  markActiveComposer('main')
  const fresh = render(<EditableComposer scope={null} />)

  stashSessionDraft(null, typed, [])
  announceNewSessionDraftKey(restored)
  fresh.unmount()
}

/** Keys type-to-focus (use-keybinds) routes while no composer is mounted; it has already preventDefault-ed them. */
const typeWhileUnmounted = (chars: string) => {
  for (const char of chars) {
    requestComposerFocus('active', { typeChar: char })
  }
}

describe('useComposerDraft — typing across the cold-start resume remount', () => {
  afterEach(() => {
    cleanup()
    editor = null
    mainComposerScope.clear()

    for (const scope of ['session-A', 'session-B', 'session-created', NEW_SESSION_DRAFT_KEY, $freshDraftKey.get()]) {
      clearSessionDraft(scope)
    }

    // Drop any announcement a test left pending.
    adoptNewSessionDraft(null)
    rotateFreshDraftKey()
  })

  it('puts the caret after the carried draft when the remounted composer adopts it, so later keystrokes append', async () => {
    // The real editor renders through FloatingComposerSurface's portal host,
    // which the PARENT attaches in its own layout effect — after the child
    // hook has already painted the adopted draft into a still-detached editor.
    function PortalSurface({ children }: { children: ReactNode }) {
      const anchorRef = useRef<HTMLDivElement>(null)
      const [host] = useState(() => document.createElement('div'))

      useLayoutEffect(() => {
        anchorRef.current?.appendChild(host)

        return () => host.remove()
      }, [host])

      return (
        <>
          <div ref={anchorRef} />
          {createPortal(children, host)}
        </>
      )
    }

    // Typed on the fresh chat while booting; the cold-start resume then
    // announces the restored id and remounts the composer on it.
    stashSessionDraft(null, 'Hello, can y', [])
    announceNewSessionDraftKey('session-created')
    markActiveComposer('main')

    render(
      <PortalSurface>
        <EditableComposer scope="session-created" />
      </PortalSurface>
    )

    await waitFor(() => expect(document.activeElement).toBe(editor))

    const selection = window.getSelection()!

    expect(composerPlainText(editor!)).toBe('Hello, can y')
    // Chromium collapses a selection-less contenteditable's caret to offset 0
    // on focus, so the next keystrokes landed BEFORE the draft
    // ("r me? ...Hello, can y"). The caret must sit inside the editor, at the end.
    expect(selection.rangeCount).toBe(1)
    expect(editor!.contains(selection.anchorNode)).toBe(true)
    expect(selection.isCollapsed).toBe(true)
    expect(caretOffsetInEditor(editor!)).toBe('Hello, can y'.length)
  })

  it('keeps keys typed while the cold-start resume has no composer mounted and appends them after the carried draft', async () => {
    startColdStartResume('Hello, c', 'session-created')
    typeWhileUnmounted('an you')

    render(<EditableComposer scope="session-created" />)

    await waitFor(() => expect(document.activeElement).toBe(editor))
    expect(composerPlainText(editor!)).toBe('Hello, can you')
    expect(caretOffsetInEditor(editor!)).toBe('Hello, can you'.length)
  })

  it('never hands the held keys to another chat that mounts before the restored one', async () => {
    startColdStartResume('Hello, c', 'session-created')
    typeWhileUnmounted('an you')

    // The user opens chat B from the sidebar before the restore finishes:
    // B's composer mounts on the same 'main' target first.
    const other = render(<EditableComposer scope="session-B" />)

    expect(composerPlainText(editor!)).toBe('')

    // The restored chat still gets its draft and the keys typed for it.
    other.unmount()
    render(<EditableComposer scope="session-created" />)

    await waitFor(() => expect(document.activeElement).toBe(editor))
    expect(composerPlainText(editor!)).toBe('Hello, can you')
  })

  it('drops the held keys once the handoff is abandoned instead of replaying them later', () => {
    startColdStartResume('Hello, c', 'session-created')
    typeWhileUnmounted('an you')

    // The restore never lands: another assignment replaces the announcement.
    act(() => announceNewSessionDraftKey(null))
    render(<EditableComposer scope="session-B" />)

    expect(composerPlainText(editor!)).toBe('')
  })

  it('does not replay keys that no composer owned when no draft handoff is pending', () => {
    markActiveComposer('main')
    requestComposerFocus('active', { typeChar: 'x' })
    render(<EditableComposer scope="session-A" />)

    expect(composerPlainText(editor!)).toBe('')
  })
})
