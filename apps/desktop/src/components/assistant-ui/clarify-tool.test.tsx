import { type ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ClarifyTool } from '@/components/assistant-ui/clarify-tool'
import { I18nProvider } from '@/i18n'
import { $clarifyRequests, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $activeSessionId } from '@/store/session'

vi.mock('@assistant-ui/react', () => ({
  useAuiState: () => true
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: vi.fn()
}))

type TestClarifyArgs = {
  allowOther?: boolean
  choices: string[]
  maxSelections?: number | null
  minSelections?: number | null
  multiSelect?: boolean
  question: string
}

function resetClarifyTestState() {
  clearNotifications()
  clearClarifyRequest()
  $clarifyRequests.set({})
  $activeSessionId.set(null)
  $gateway.set(null)
}

function renderClarifyTool(requestMock = vi.fn().mockResolvedValue({ ok: true }), argsOverride: Partial<TestClarifyArgs> = {}) {
  const args: TestClarifyArgs = {
    question: 'Which context should Hermes use?',
    choices: ['Past sessions', 'Local reports', 'Web sources'],
    ...argsOverride
  }

  $clarifyRequests.set({})
  $gateway.set({ request: requestMock } as never)
  setClarifyRequest({
    requestId: 'req-1',
    question: 'Which context should Hermes use?',
    choices: ['Past sessions', 'Local reports', 'Web sources'],
    multiSelect: Boolean(args.multiSelect),
    sessionId: null
  })

  const props: ToolCallMessagePartProps = {
    type: 'tool-call',
    toolCallId: 'tool-1',
    toolName: 'clarify',
    args,
    argsText: '',
    result: undefined,
    status: { type: 'running' },
    addResult: vi.fn(),
    resume: vi.fn()
  }

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ClarifyTool {...props} />
    </I18nProvider>
  )

  return requestMock
}

const REQUEST_ID = 'clarify-req-1'
const SESSION_ID = 'session-1'

function renderPendingClarify(request: { request?: ReturnType<typeof vi.fn> } = {}) {
  const requestMock = request.request ?? vi.fn(async () => ({ ok: true }))

  $activeSessionId.set(SESSION_ID)
  setClarifyRequest({
    requestId: REQUEST_ID,
    sessionId: SESSION_ID,
    question: 'Pick one?',
    choices: ['Continue waiting', 'Stop now']
  })
  $gateway.set({ request: requestMock } as never)

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ClarifyTool
        {...({
          args: { question: 'Pick one?', choices: ['Continue waiting', 'Stop now'] },
          result: undefined,
          status: { type: 'running' }
        } as unknown as Parameters<typeof ClarifyTool>[0])}
      />
    </I18nProvider>
  )

  return requestMock
}

beforeEach(() => {
  resetClarifyTestState()
})

afterEach(() => {
  cleanup()
  resetClarifyTestState()
  vi.restoreAllMocks()
})

describe('ClarifyTool selection status UX', () => {
  it('keeps regular choice clicks as immediate single-choice submit', async () => {
    const request = renderClarifyTool()

    fireEvent.click(screen.getByRole('button', { name: 'Past sessions' }))

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        'clarify.respond',
        {
          request_id: 'req-1',
          answer: 'Past sessions'
        },
        120_000
      )
    )
  })

  it('shows the selected single choice while the response is pending', () => {
    const request = renderClarifyTool(vi.fn(() => new Promise(() => {})))

    fireEvent.click(screen.getByRole('button', { name: 'Past sessions' }))

    expect(request).toHaveBeenCalledWith(
      'clarify.respond',
      {
        request_id: 'req-1',
        answer: 'Past sessions'
      },
      120_000
    )
    const status = screen.getByRole('status')
    expect(status.textContent).toContain('Selected')
    expect(status.textContent).toContain('Past sessions')
  })

  it('submits a single-choice answer from its letter shortcut', async () => {
    const request = renderClarifyTool()

    fireEvent.keyDown(window, { key: 'b' })

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        'clarify.respond',
        {
          request_id: 'req-1',
          answer: 'Local reports'
        },
        120_000
      )
    )
  })

  it('focuses Other from the trailing letter shortcut and submits it with Enter', async () => {
    const request = renderClarifyTool()

    fireEvent.keyDown(window, { key: 'd' })

    const other = screen.getByRole('textbox', { name: 'Other (type your answer)' })
    expect(document.activeElement).toBe(other)

    fireEvent.change(other, { target: { value: 'Use a custom path' } })
    fireEvent.keyDown(other, { key: 'Enter' })

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        'clarify.respond',
        {
          request_id: 'req-1',
          answer: 'Use a custom path'
        },
        120_000
      )
    )
  })

  it('does not expose staged multi-select controls for single-choice prompts', () => {
    renderClarifyTool()

    expect(screen.queryByRole('button', { name: 'Toggle Past sessions for multi-select' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Select selected' })).toBeNull()
  })

  it('keeps the multi-select submit button visible before any choice is staged', () => {
    renderClarifyTool(vi.fn().mockResolvedValue({ ok: true }), { multiSelect: true })

    expect(screen.getByRole('note').textContent).toContain('Multi-select')
    const submit = screen.getByRole('button', { name: 'Select selected' })
    expect(submit.hasAttribute('disabled')).toBe(true)
  })

  it('hides Skip when a multi-select prompt requires at least one selection', () => {
    renderClarifyTool(vi.fn().mockResolvedValue({ ok: true }), {
      allowOther: false,
      minSelections: 2,
      multiSelect: true
    })

    expect(screen.queryByRole('button', { name: 'Skip' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Select selected' })).toBeTruthy()
  })

  it('hides Skip for constrained single-select prompts that disallow Other', () => {
    renderClarifyTool(vi.fn().mockResolvedValue({ ok: true }), { allowOther: false })

    expect(screen.queryByRole('button', { name: 'Skip' })).toBeNull()
  })

  it('uses multi-select rows to stage multiple choices without submitting until Select selected', async () => {
    const request = renderClarifyTool(vi.fn().mockResolvedValue({ ok: true }), { multiSelect: true })

    fireEvent.click(screen.getByRole('button', { name: 'Toggle Past sessions for multi-select' }))
    fireEvent.click(screen.getByRole('button', { name: 'Toggle Web sources for multi-select' }))

    expect(request).not.toHaveBeenCalled()
    expect(screen.getByText('2 selected')).toBeTruthy()
    expect(screen.getByText('Selected')).toBeTruthy()
    expect(screen.getByText('Past sessions, Web sources')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Select selected' }))

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        'clarify.respond',
        {
          request_id: 'req-1',
          answer: 'Past sessions, Web sources'
        },
        120_000
      )
    )
  })

  it('toggles multi-select choices by letter shortcut and submits them with Enter', async () => {
    const request = renderClarifyTool(vi.fn().mockResolvedValue({ ok: true }), { multiSelect: true })

    fireEvent.keyDown(window, { key: 'a' })
    fireEvent.keyDown(window, { key: 'c' })

    expect(request).not.toHaveBeenCalled()
    expect(screen.getByText('2 selected')).toBeTruthy()
    expect(screen.getByText('Past sessions, Web sources')).toBeTruthy()

    fireEvent.keyDown(window, { key: 'Enter' })

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        'clarify.respond',
        {
          request_id: 'req-1',
          answer: 'Past sessions, Web sources'
        },
        120_000
      )
    )
  })

  it('keeps free-form Other submission available for multi-select prompts', async () => {
    const request = renderClarifyTool(vi.fn().mockResolvedValue({ ok: true }), { multiSelect: true })

    const other = screen.getByRole('textbox', { name: 'Other (type your answer)' })
    fireEvent.focus(other)
    fireEvent.change(other, { target: { value: 'Use a custom path' } })

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        'clarify.respond',
        {
          request_id: 'req-1',
          answer: 'Use a custom path'
        },
        120_000
      )
    )
  })

  it('shows the current free-form draft as a custom selection', () => {
    renderClarifyTool()

    const other = screen.getByRole('textbox', { name: 'Other (type your answer)' })

    fireEvent.focus(other)
    fireEvent.change(other, {
      target: { value: 'Use only local reports' }
    })

    expect(screen.getByText('Selected')).toBeTruthy()
    expect(screen.getByText('Other (type your answer): Use only local reports')).toBeTruthy()
  })
})

describe('ClarifyTool response timeout handling', () => {
  it('uses a long dedicated timeout for clarify.respond acknowledgements', async () => {
    const requestMock = renderPendingClarify()

    fireEvent.click(screen.getByRole('button', { name: /Continue waiting/i }))

    await waitFor(() => {
      expect(requestMock).toHaveBeenCalledWith(
        'clarify.respond',
        { request_id: REQUEST_ID, answer: 'Continue waiting' },
        120_000
      )
    })
  })

  it('warns that a timed-out clarify response may still be processing', async () => {
    const requestMock = renderPendingClarify({
      request: vi.fn(async () => {
        throw new Error('request timed out: clarify.respond')
      })
    })

    fireEvent.click(screen.getByRole('button', { name: /Stop now/i }))

    await waitFor(() => expect(requestMock).toHaveBeenCalled())

    await waitFor(() => {
      expect($notifications.get()[0]).toMatchObject({
        kind: 'warning',
        title: 'Clarify response may still be processing',
        message: expect.stringContaining('wait a moment before trying again')
      })
    })
  })

  it('clears expired clarify requests when the backend has no pending request', async () => {
    const requestMock = renderPendingClarify({
      request: vi.fn(async () => {
        throw new Error('RPC 4009: no pending answer request')
      })
    })

    fireEvent.click(screen.getByRole('button', { name: /Continue waiting/i }))

    await waitFor(() => expect(requestMock).toHaveBeenCalled())

    await waitFor(() => {
      expect($clarifyRequests.get()[SESSION_ID]).toBeUndefined()
      expect(screen.getByText('Clarify request expired')).toBeTruthy()
      expect($notifications.get()[0]).toMatchObject({
        kind: 'warning',
        title: 'Clarify request expired',
        message: expect.stringContaining('no longer pending')
      })
    })
  })
})
