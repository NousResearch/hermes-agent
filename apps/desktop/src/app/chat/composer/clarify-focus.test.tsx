import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { ClarifyTool } from '@/components/assistant-ui/clarify-tool'
import { PaneGroupContext } from '@/components/pane-shell/pane-visibility'
import { I18nProvider } from '@/i18n'
import { clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $composerPopout, setComposerPoppedOut } from '@/store/composer-popout'
import { $gateway } from '@/store/gateway'
import { rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId } from '@/store/session'

import { FloatingComposerSurface } from './floating-surface'
import { focusComposerInput, getActiveComposer } from './focus'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from './scope'

// Only the assistant runtime's running state is stubbed; the form, request
// registry, response handling, surface registration and focus tracker are real.
vi.mock('@assistant-ui/react', () => ({ useAuiState: () => true }))

afterEach(async () => {
  cleanup()
  await act(async () => {})
  clearClarifyRequest()
  resetServerRequestsForTests()
  $activeSessionId.set(null)
  $gateway.set(null)
  $composerPopout.set({ poppedOut: false, position: { bottom: 24, right: 24 } })
  window.getSelection()?.removeAllRanges()
  vi.useRealTimers()
})

function mount(kind: 'other' | 'open' | 'batch', floating: boolean) {
  const choices = kind === 'open' ? null : ['Alpha', 'Beta']

  const questions = kind === 'batch' ? [
    { qid: 'q0', question: 'First?', choices, multiSelect: false },
    { qid: 'q1', question: 'Second?', choices: null, multiSelect: false }
  ] : undefined

  const respond = vi.fn()
  const request = vi.fn(async () => ({ ok: true }))
  $activeSessionId.set('focus-session')
  $gateway.set({ request } as never)
  rememberServerRequest({ id: 'focus-request', method: 'clarify', params: {}, respond, fail: vi.fn() })
  setClarifyRequest({ requestId: 'focus-request', sessionId: 'focus-session', question: 'First?', choices, multiSelect: false, questions })
  const args: ToolCallMessagePartProps['args'] = questions ? { questions } : { question: 'First?', choices }

  const props: ToolCallMessagePartProps = {
    args, argsText: JSON.stringify(args), toolCallId: 'focus-tool', toolName: 'clarify', type: 'tool-call',
    status: { type: 'running' }, result: undefined, isError: false,
    addResult: vi.fn(), resume: vi.fn(), respondToApproval: vi.fn()
  }

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <PaneGroupContext value="focus-group">
        <ComposerScopeProvider value={MAIN_COMPOSER_SCOPE}>
          <ComposerSurfaceProvider value="focus-surface">
            <div data-chat-surface="" data-composer-surface-id="focus-surface" data-tree-group="focus-group">
              <ClarifyTool {...props} />
              <FloatingComposerSurface>
                <div aria-label="Draft" contentEditable data-slot="composer-rich-input" role="textbox" tabIndex={0} />
              </FloatingComposerSurface>
            </div>
          </ComposerSurfaceProvider>
        </ComposerScopeProvider>
      </PaneGroupContext>
    </I18nProvider>
  )
  act(() => setComposerPoppedOut(floating))

  return { draft: screen.getByLabelText('Draft'), fields: [...globalThis.document.querySelectorAll('textarea')], respond, request }
}

it.each([false, true].flatMap(floating => ['other', 'open', 'batch'].map(kind => ({ floating, kind: kind as 'other' | 'open' | 'batch' }))))(
  'keeps real clarify input through pointerup, focusin and movement ($kind, floating=$floating)',
  async ({ kind, floating }) => {
    const { draft, fields, respond, request } = mount(kind, floating)

    for (const [index, field] of fields.entries()) {
      const row = field.closest('label') ?? field
      fireEvent.pointerDown(row)
      fireEvent.pointerUp(row)
      // Native label activation / the card's keyboard handler focus after
      // pointerup. jsdom does not implement label default activation.
      act(() => field.focus())
      expect(globalThis.document.activeElement).toBe(field)
      fireEvent.change(field, { target: { value: `中文回答${index}` } })
      field.setSelectionRange(1, 3)
      fireEvent.pointerMove(field, { buttons: 0, clientX: 40 + index, clientY: 60 })
      expect(globalThis.document.activeElement).toBe(field)
      expect([field.selectionStart, field.selectionEnd]).toEqual([1, 3])
      fireEvent.compositionStart(globalThis.document.activeElement!, { data: '' })
      fireEvent.keyDown(globalThis.document.activeElement!, { key: 'Enter', isComposing: true, keyCode: 229 })
      fireEvent.compositionEnd(globalThis.document.activeElement!, { data: '中文' })
      expect(respond).not.toHaveBeenCalled()
      expect(request).not.toHaveBeenCalled()
      expect(draft.textContent).toBe('')
    }

    await act(async () => fireEvent.click(screen.getByRole('button', { name: /continue/i })))

    if (kind === 'batch') {
      expect(request.mock.calls).toEqual([
        ['clarify.lock', { answer: '中文回答0', question_id: 'q0', request_id: 'focus-request' }],
        ['clarify.lock', { answer: '中文回答1', question_id: 'q1', request_id: 'focus-request' }]
      ])
    } else {
      expect(respond).toHaveBeenCalledExactlyOnceWith({ answer: '中文回答0' })
    }

    fireEvent.pointerDown(draft)
    act(() => draft.focus())
    fireEvent.pointerUp(draft)
    expect(globalThis.document.activeElement).toBe(draft)
  }
)

it('allows the real Other keyboard shortcut to focus its textarea', () => {
  const { fields } = mount('other', false)
  fireEvent.keyDown(window, { key: '3' })
  expect(globalThis.document.activeElement).toBe(fields[0])
})

it('late composer retries respect a newly focused clarify input and its selection', () => {
  vi.useFakeTimers()
  const { draft, fields } = mount('open', false)
  act(() => focusComposerInput(draft))
  fireEvent.pointerDown(fields[0])
  act(() => fields[0].focus())
  fireEvent.pointerUp(fields[0])
  fireEvent.change(fields[0], { target: { value: '保留选区' } })
  fields[0].setSelectionRange(1, 3)
  act(() => vi.runAllTimers())
  expect(globalThis.document.activeElement).toBe(fields[0])
  expect([fields[0].selectionStart, fields[0].selectionEnd]).toEqual([1, 3])
})

it.each([false, true])('keeps focus routing usable with a clarify in another pane (hidden=%s)', hidden => {
  const { fields } = mount('open', false)
  render(
    <PaneGroupContext value="other-group">
      <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: 'other' }}>
        <ComposerSurfaceProvider value="other-surface">
          <div data-chat-surface="" data-composer-surface-id="other-surface" data-tree-group="other-group">
            <FloatingComposerSurface>
              <div aria-label="Other draft" contentEditable data-slot="composer-rich-input" role="textbox" tabIndex={0} />
            </FloatingComposerSurface>
          </div>
        </ComposerSurfaceProvider>
      </ComposerScopeProvider>
    </PaneGroupContext>
  )
  const other = screen.getByLabelText('Other draft')
  act(() => fields[0].focus())
  fields[0].closest('form')!.toggleAttribute('data-pane-hidden', hidden)
  fireEvent.pointerMove(other, { buttons: 0, clientX: 200, clientY: 100 })
  expect(getActiveComposer()).toBe('other')
  expect(globalThis.document.activeElement).toBe(fields[0])
  fireEvent.pointerDown(other)
  act(() => other.focus())
  fireEvent.pointerUp(other)
  expect(globalThis.document.activeElement).toBe(other)
})
