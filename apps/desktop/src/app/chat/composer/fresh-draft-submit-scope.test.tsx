// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { sessionContextDrift } from '@/app/session/hooks/session-context-drift'
import { I18nProvider } from '@/i18n'
import { $freshDraftKey, mainComposerScope, rotateFreshDraftKey } from '@/store/composer'

import { RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
})

// THE INVARIANT: the first send of a new chat survives the drift gate.
//
// The composer and the drift gate meet here. ChatBar scopes a SESSIONLESS
// draft to the per-lifecycle fresh key (`__new__:<uuid>`) — not to null — and
// threads that scope into `onSubmit`. submit.ts then measures it against the
// chat it is about to create. Nothing in either module alone is wrong, and
// every unit harness for submit.ts passes `null` for a new chat, so the pair
// could drift apart unnoticed: the real composer's `__new__:<uuid>` read as
// "the composer has a DIFFERENT chat loaded" and aborted every first send
// AFTER session.create had already minted the runtime and re-homed the route.
// The symptom was a new chat that created a session, never submitted the
// prompt, never persisted a `sessions` row, and answered every transcript read
// for that id with 404 "Session not found".
//
// So this test asserts the handoff end to end: the scope the REAL ChatBar
// emits for a sessionless chat is a scope the REAL drift gate tolerates, both
// before a stored id exists and after the create publishes one.
const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ freshDraftKey, onSubmit }: { freshDraftKey: string; onSubmit: ChatBarProps['onSubmit'] }) {
  // Only here to satisfy ChatBar's provider; nothing in this test reads it.
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: false,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          {/* No sessionId / queueSessionKey: a brand-new chat with no backend session yet. */}
          <ChatBar
            busy={false}
            disabled={false}
            freshDraftKey={freshDraftKey}
            gateway={null}
            onCancel={vi.fn()}
            onSubmit={onSubmit}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

type ChatBarProps = Parameters<typeof ChatBar>[0]

describe('the first send of a new chat', () => {
  it('emits the fresh-draft scope that the drift gate must tolerate', async () => {
    const freshDraftKey = rotateFreshDraftKey()

    expect($freshDraftKey.get()).toBe(freshDraftKey)

    const onSubmit = vi.fn(async () => true) as unknown as ChatBarProps['onSubmit']

    const { container } = render(<Harness freshDraftKey={freshDraftKey} onSubmit={onSubmit} />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })

    await act(async () => {
      editor.focus()
      editor.textContent = 'first question of a brand new chat'
      fireEvent.input(editor)
      fireEvent.keyDown(editor, { key: 'Enter', isComposing: false })
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)

    const composerScope = (vi.mocked(onSubmit).mock.calls[0][1] as { composerScope?: null | string }).composerScope

    // What ChatBar actually sends for a sessionless chat.
    expect(composerScope).toBe(freshDraftKey)

    // Pre-create: no stored target resolved yet (resolveComposerSessionKey(null)).
    expect(
      sessionContextDrift({
        startRouteToken: '/new::',
        nowRouteToken: '/new::',
        startSelectedStoredId: null,
        nowSelectedStoredId: null,
        submitTargetStoredId: null,
        composerScope,
        submitTargetComposerScope: null
      })
    ).toBeNull()

    // Post-create: submit.ts has re-pinned its baseline to the chat it minted.
    expect(
      sessionContextDrift({
        startRouteToken: '/stored-brand-new::',
        nowRouteToken: '/stored-brand-new::',
        startSelectedStoredId: 'stored-brand-new',
        nowSelectedStoredId: 'stored-brand-new',
        submitTargetStoredId: 'stored-brand-new',
        composerScope,
        submitTargetComposerScope: 'stored-brand-new'
      })
    ).toBeNull()
  })
})
