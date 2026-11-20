import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { host } from '@/sdk'
import { composerHost } from '@/sdk/composer'
import { mainComposerScope, stashSessionDraft } from '@/store/composer'
import { $activeGatewayProfile, $activeProfile } from '@/store/profile'
import { $activeSessionId, $connection, $gatewayState, $selectedStoredSessionId } from '@/store/session'

import { $composerPredictions } from './predictions'
import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import { placeCaretAtEnd } from './test-utils'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

const sessionId = 'prediction-integration'
const suggestions = ['First complete suggestion', '第二个完整建议\n可编辑，不会自动发送', 'Third suggestion']

// Same passive-effect publication as a runtime contribution after a reply.
function Publisher({ enabled }: { enabled: boolean }) {
  useEffect(() => {
    const handle = enabled
      ? composerHost.setPredictions({ sessionId, suggestions, profile: 'default', connectionId: 'local' })
      : null

    return () => {
      if (handle) {
        composerHost.clearPredictions(handle)
      }
    }
  }, [enabled])

  return null
}

function Harness({
  busy = false,
  disabled = false,
  publish = false,
  profile = 'default',
  onSubmit = vi.fn(async () => true)
}: {
  busy?: boolean
  disabled?: boolean
  publish?: boolean
  profile?: string
  onSubmit?: (text: string) => Promise<boolean>
}) {
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
          <ChatBar
            busy={busy}
            disabled={disabled}
            gateway={null}
            onCancel={vi.fn()}
            onSubmit={onSubmit}
            profile={profile}
            queueSessionKey={sessionId}
            sessionId={sessionId}
            state={state}
          />
          <Publisher enabled={publish} />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

const publishPredictions = () =>
  act(() => composerHost.setPredictions({ sessionId, suggestions, profile: 'default', connectionId: 'local' }))

const getEditor = (container: HTMLElement) => {
  const editor = container.querySelector<HTMLDivElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
  editor.tabIndex = 0
  act(() => editor.focus())
  placeCaretAtEnd(editor)

  return editor
}

const preview = (container: HTMLElement) => container.querySelector('[data-slot="composer-prediction-text"]')

beforeEach(() => {
  $activeProfile.set('default')
  $activeGatewayProfile.set('default')
  $connection.set({
    baseUrl: 'http://localhost',
    wsUrl: 'ws://localhost',
    token: '',
    logs: [],
    nativeOverlayWidth: 0,
    windowButtonPosition: null,
    isFullscreen: false,
    connectionId: 'local'
  })
  $selectedStoredSessionId.set(sessionId)
  $activeSessionId.set(sessionId)
  $gatewayState.set('open')
  stashSessionDraft(sessionId, '', [])
  mainComposerScope.clear()
})
afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  stashSessionDraft(sessionId, '', [])
  $activeSessionId.set(null)
  $gatewayState.set('idle')
})

describe('ChatBar and public composer prediction SDK', () => {
  it('accepts the SDK routed profile with a stale HTTP profile and rejects old-scope results', () => {
    const view = render(<Harness profile="default" />)
    publishPredictions()
    const old = $composerPredictions.get()!.handle
    act(() => $activeGatewayProfile.set('work'))
    expect(preview(view.container)).toBeNull()
    expect($activeProfile.get()).toBe('default')
    view.rerender(<Harness profile="work" />)
    act(() => {
      expect(
        host.composer.setPredictions({ sessionId, suggestions, profile: 'default', connectionId: 'local' })
      ).toBeNull()
      expect(
        host.composer.setPredictions({
          sessionId,
          suggestions,
          profile: host.state.profile.get(),
          connectionId: host.state.connectionId.get()!
        })
      ).not.toBeNull()
      expect(host.composer.clearPredictions(old)).toBe(false)
    })
    expect(preview(view.container)?.textContent).toBe(suggestions[0])
    act(() => $activeProfile.set('work'))
    expect(preview(view.container)?.textContent).toBe(suggestions[0])
  })

  it('exposes the agreed contract on host.composer', () => {
    expect(host.composer.setPredictions).toBe(composerHost.setPredictions)
    expect(host.composer.clearPredictions).toBe(composerHost.clearPredictions)
  })

  it('receives effect-published suggestions on mount and after a busy reply settles', () => {
    const view = render(<Harness publish />)
    expect(preview(view.container)?.textContent).toBe(suggestions[0])
    view.rerender(<Harness busy />)
    expect(preview(view.container)).toBeNull()
    view.rerender(<Harness publish />)
    expect(preview(view.container)?.textContent).toBe(suggestions[0])
  })

  it('cycles/wraps through real key events and accepts into the editable draft without sending', async () => {
    const onSubmit = vi.fn(async (_text: string) => true)
    const view = render(<Harness onSubmit={onSubmit} />)
    const editor = getEditor(view.container)
    publishPredictions()
    expect(composerPlainText(editor)).toBe('')
    expect(await composerHost.getDraft(sessionId)).toBe('')
    expect(editor.getAttribute('data-placeholder')).toBe('')

    for (const expected of [suggestions[1], suggestions[2], suggestions[0], suggestions[1]]) {
      expect(fireEvent.keyDown(editor, { key: 'Tab' })).toBe(false)
      expect(preview(view.container)?.textContent).toBe(expected)
    }

    expect(fireEvent.keyDown(editor, { key: 'ArrowRight' })).toBe(false)
    expect(composerPlainText(editor)).toBe(suggestions[1])
    expect(await composerHost.getDraft(sessionId)).toBe(suggestions[1])
    expect(preview(view.container)).toBeNull()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(editor.getAttribute('contenteditable')).toBe('true')
    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Enter' })
    })
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[0]).toBe(suggestions[1])
  })

  it('records acceptance in native draft undo history', () => {
    const view = render(<Harness />)
    const editor = getEditor(view.container)
    publishPredictions()
    fireEvent.keyDown(editor, { key: 'ArrowRight' })
    expect(composerPlainText(editor)).toBe(suggestions[0])
    fireEvent.keyDown(editor, { key: 'z', ctrlKey: true })
    expect(composerPlainText(editor)).toBe('')
    expect(preview(view.container)).toBeNull()
  })

  it('preserves leading and trailing whitespace on acceptance without submitting', () => {
    const onSubmit = vi.fn(async () => true)
    const view = render(<Harness onSubmit={onSubmit} />)
    const editor = getEditor(view.container)
    const text = '  indented\n完整消息  '
    act(() => {
      expect(
        composerHost.setPredictions({ sessionId, profile: 'default', connectionId: 'local', suggestions: [text] })
      ).not.toBeNull()
    })
    expect(preview(view.container)?.textContent).toBe(text)
    expect(composerPlainText(editor)).toBe('')
    fireEvent.keyDown(editor, { key: 'ArrowRight' })
    expect(composerPlainText(editor)).toBe(text)
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('rejected publication has no effect on the editable draft or submit port', async () => {
    const onSubmit = vi.fn(async () => true)
    const view = render(<Harness onSubmit={onSubmit} />)
    const editor = getEditor(view.container)
    await act(async () => {
      expect(await composerHost.setDraft(sessionId, 'My draft')).toBe(true)
    })
    act(() => {
      expect(
        composerHost.setPredictions({ sessionId, profile: 'default', connectionId: 'local', suggestions })
      ).toBeNull()
    })
    expect(composerPlainText(editor)).toBe('My draft')
    expect(await composerHost.getDraft(sessionId)).toBe('My draft')
    expect(preview(view.container)).toBeNull()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('keeps Shift+Tab and modified ArrowRight native and Escape dismisses once', () => {
    const view = render(<Harness />)
    const editor = getEditor(view.container)
    publishPredictions()
    expect(fireEvent.keyDown(editor, { key: 'Tab', shiftKey: true })).toBe(true)
    expect(fireEvent.keyDown(editor, { key: 'ArrowRight', metaKey: true })).toBe(true)
    expect(preview(view.container)?.textContent).toBe(suggestions[0])
    expect(fireEvent.keyDown(editor, { key: 'Escape' })).toBe(false)
    expect(preview(view.container)).toBeNull()
    publishPredictions()
    expect(preview(view.container)).toBeNull()
    expect(fireEvent.keyDown(editor, { key: 'Tab' })).toBe(true)
  })

  it('invalidates on typing before the coalesced state flush and rejects late publication', () => {
    const view = render(<Harness />)
    const editor = getEditor(view.container)
    publishPredictions()
    act(() => {
      editor.textContent = 'My actual draft'
      placeCaretAtEnd(editor)
      fireEvent.input(editor)
      composerHost.setPredictions({ sessionId, suggestions, profile: 'default', connectionId: 'local' })
      fireEvent.keyDown(editor, { key: 'ArrowRight' })
    })
    expect(composerPlainText(editor)).toBe('My actual draft')
    expect(preview(view.container)).toBeNull()
  })

  it('removes the preview before IME preedit and does not accept/submit a candidate key', () => {
    const onSubmit = vi.fn(async () => true)
    const view = render(<Harness onSubmit={onSubmit} />)
    const editor = getEditor(view.container)
    publishPredictions()
    fireEvent.compositionStart(editor)
    expect(preview(view.container)).toBeNull()
    expect(fireEvent.keyDown(editor, { key: 'Tab', isComposing: true })).toBe(true)
    expect(fireEvent.keyDown(editor, { key: 'ArrowRight', isComposing: true })).toBe(true)
    act(() => {
      editor.textContent = '你好'
      fireEvent.input(editor)
      fireEvent.compositionEnd(editor)
    })
    fireEvent.keyDown(editor, { key: 'Enter', keyCode: 229 })
    expect(composerPlainText(editor)).toBe('你好')
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('preserves a restored draft instead of displaying or accepting predictions', () => {
    stashSessionDraft(sessionId, 'Saved draft', [])
    const view = render(<Harness publish />)
    const editor = getEditor(view.container)
    publishPredictions()
    expect(preview(view.container)).toBeNull()
    fireEvent.keyDown(editor, { key: 'ArrowRight' })
    expect(composerPlainText(editor)).toBe('Saved draft')
  })

  it.each(['/', '@'])('leaves %s completion keys with the native trigger engine', async token => {
    const view = render(<Harness />)
    const editor = getEditor(view.container)
    publishPredictions()
    await act(async () => {
      editor.textContent = token
      placeCaretAtEnd(editor)
      fireEvent.input(editor)
      fireEvent.keyUp(editor, { key: token })
    })
    publishPredictions()
    expect(preview(view.container)).toBeNull()
    fireEvent.keyDown(editor, { key: 'Tab' })
    expect(composerPlainText(editor)).not.toContain(suggestions[0])
    expect($composerPredictions.get()).toBeNull()
  })

  it.each(['busy', 'disabled'] as const)('does not publish or intercept Right/Tab while %s', flag => {
    const view = render(<Harness {...{ [flag]: true }} publish />)
    const editor = getEditor(view.container)
    publishPredictions()
    expect(preview(view.container)).toBeNull()
    expect(fireEvent.keyDown(editor, { key: 'ArrowRight' })).toBe(true)
    expect(fireEvent.keyDown(editor, { key: 'Tab' })).toBe(true)
  })
})
