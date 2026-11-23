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
  approvalKindFields,
  type ApprovalRequest,
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

describe('folder and site approval cards', () => {
  function setAccess(extra: Partial<ApprovalRequest> = {}) {
    $activeSessionId.set('sess-1')
    setApprovalRequest({
      choices: ['once', 'always', 'deny'],
      command: '',
      description: 'Allow read and write access to C:\\Users\\me\\docs',
      kind: 'access',
      requestId: 'sandbox-card-1',
      risk: 'The agent can change and delete files in this folder.',
      serverRequestId: 'srq-approval',
      sessionId: 'sess-1',
      target: { access: 'read_write', path: 'C:\\Users\\me\\docs' },
      ...extra
    })
  }

  function setSite(extra: Partial<ApprovalRequest> = {}) {
    $activeSessionId.set('sess-1')
    setApprovalRequest({
      choices: ['once', 'always', 'deny'],
      command: '',
      description: 'Allow the sandbox to reach example.com',
      kind: 'network',
      requestId: 'sandbox-card-2',
      risk: 'Any program in the sandbox, http and https.',
      serverRequestId: 'srq-approval',
      sessionId: 'sess-1',
      target: { host: 'example.com' },
      ...extra
    })
  }

  const reasonField = () => screen.getByRole('textbox', { name: 'Reason for denying' }) as HTMLInputElement

  it('renders a folder card: the folder, its access level, the risk line; Deny / Always allow… / Allow', async () => {
    setAccess()
    const { container } = render(<PendingApprovalStack />)

    expect(screen.getByText('Folder access')).toBeTruthy()
    expect(screen.getByText('C:\\Users\\me\\docs')).toBeTruthy()
    expect(screen.getByText(/Read and write/)).toBeTruthy()
    expect(screen.getByText('The agent can change and delete files in this folder.')).toBeTruthy()
    expect(screen.getAllByRole('button').map(button => button.textContent?.replace('↵', '').trim())).toEqual([
      'Deny',
      'Always allow…',
      'Allow'
    ])
    expect(screen.queryByText(/Command/)).toBeNull()
    expect(container.querySelector('pre')).toBeNull()
    expect(container.querySelector('[data-approval-kind="access"] .codicon-folder')).not.toBeNull()
    expect(screen.queryByRole('textbox')).toBeNull()

    fireEvent.pointerMove(screen.getByRole('button', { name: /^Allow/ }), { pointerType: 'mouse' })
    expect((await screen.findByRole('tooltip')).textContent).toContain('Until the sandbox is switched off')
  })

  it('never puts the bidirectional control characters of a site name on screen as they came', () => {
    // U+202E makes the text after it read right to left, so the name on the card would differ
    // from the host the answer is for. A card's fields come through approvalKindFields.
    setSite(approvalKindFields({ host: 'evil.example\u202Emoc.buhtig', kind: 'network' }))
    const { container } = render(<PendingApprovalStack />)
    const shown = container.querySelector('[data-slot="tool-approval-target"]')?.textContent

    expect(shown).not.toMatch(/[\u202A-\u202E\u2066-\u2069]/)
    expect(shown).toContain('evil.example\\u202emoc.buhtig')
  })

  it('names a read-only folder as read only', () => {
    setAccess({ target: { access: 'read', path: 'C:\\Users\\me\\docs' } })
    render(<PendingApprovalStack />)

    expect(screen.getByText(/Read only/)).toBeTruthy()
    expect(screen.queryByText(/Read and write/)).toBeNull()
  })

  it('hides the turn activity row for a folder card and shows it for a command', () => {
    setAccess()
    const { container, unmount } = render(<PendingApprovalStack />)

    expect(container.querySelector('[data-approval-activity]')).toBeNull()
    unmount()

    setRequest('rm -rf /tmp/x', undefined, { requestId: 'apr-cmd' })
    const second = render(<PendingApprovalStack />)
    expect(second.container.querySelector('[data-approval-activity]')).not.toBeNull()
  })

  it('renders a site card: the host and the risk line', () => {
    setSite()
    const { container } = render(<PendingApprovalStack />)

    expect(screen.getByText('Website access')).toBeTruthy()
    expect(screen.getByText('example.com')).toBeTruthy()
    expect(screen.getByText('Any program in the sandbox, http and https.')).toBeTruthy()
    expect(container.querySelector('.codicon-globe')).not.toBeNull()
    expect(container.querySelector('.codicon-terminal')).toBeNull()
  })

  it('names a site by its label when the request carries one', () => {
    setSite({ target: { host: 'api.keenable.ai', label: 'Web search (api.keenable.ai)' } })
    render(<PendingApprovalStack />)

    expect(screen.getByText('Web search (api.keenable.ai)')).toBeTruthy()
  })

  it('answers Allow with {choice: "once"}', async () => {
    mockGateway()
    const respond = liveApproval()
    setSite()
    render(<PendingApprovalStack />)

    fireEvent.click(screen.getByRole('button', { name: /^Allow/ }))
    await waitFor(() => expect(respond).toHaveBeenCalledWith({ choice: 'once' }))
  })

  it('Deny opens a reason field; Enter sends the reason with {choice: "deny"}', async () => {
    mockGateway()
    const respond = liveApproval()
    setSite()
    render(<PendingApprovalStack />)

    fireEvent.click(screen.getByRole('button', { name: /^Deny$/ }))
    expect(respond).not.toHaveBeenCalled()
    const field = reasonField()
    expect(document.activeElement).toBe(field)
    expect(field.placeholder).toBe('Tell Hermes why (optional)')
    expect(field.maxLength).toBe(500)
    // While the field is open the card offers Cancel and Deny only.
    expect(screen.getAllByRole('button').map(button => button.textContent?.replace('↵', '').trim())).toEqual([
      'Cancel',
      'Deny'
    ])

    fireEvent.change(field, { target: { value: '  Not this site  ' } })
    fireEvent.submit(field.closest('form')!)

    await waitFor(() => expect(respond).toHaveBeenCalledWith({ choice: 'deny', reason: 'Not this site' }))
    await waitFor(() => expect(sessionApprovalRequests('sess-1').get()).toEqual([]))
  })

  it('sends no reason when the field is left empty', async () => {
    mockGateway()
    const respond = liveApproval()
    setAccess()
    render(<PendingApprovalStack />)

    fireEvent.click(screen.getByRole('button', { name: /^Deny$/ }))
    fireEvent.click(screen.getByRole('button', { name: /^Deny/ }))

    await waitFor(() => expect(respond).toHaveBeenCalledWith({ choice: 'deny' }))
  })

  it('Escape in the reason field cancels: nothing is sent and the card is back', async () => {
    mockGateway()
    const respond = liveApproval()
    setAccess()
    render(<PendingApprovalStack />)

    fireEvent.click(screen.getByRole('button', { name: /^Deny$/ }))
    fireEvent.change(reasonField(), { target: { value: 'draft' } })
    fireEvent.keyDown(reasonField(), { key: 'Escape' })

    expect(screen.queryByRole('textbox')).toBeNull()
    expect(screen.getByRole('button', { name: /^Allow/ })).toBeTruthy()
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole('button', { name: /^Deny$/ })))
    expect(respond).not.toHaveBeenCalled()
    expect(sessionApprovalRequests('sess-1').get()).toHaveLength(1)
  })

  it('the stack’s Escape opens the reason field, Escape closes it, Enter then denies', async () => {
    mockGateway()
    const respond = liveApproval()
    setSite()
    render(<PendingApprovalStack />)

    act(() => {
      handleApprovalKey(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }))
    })
    releaseApprovalKey()
    expect(reasonField()).toBeTruthy()
    expect(respond).not.toHaveBeenCalled()

    // Focus left the field: the stack's keys act on Cancel and Deny.
    act(() => reasonField().blur())
    act(() => {
      handleApprovalKey(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }))
    })
    releaseApprovalKey()
    expect(screen.queryByRole('textbox')).toBeNull()

    act(() => {
      handleApprovalKey(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }))
    })
    releaseApprovalKey()
    act(() => reasonField().blur())
    act(() => {
      handleApprovalKey(new KeyboardEvent('keydown', { key: 'Enter', cancelable: true }))
    })

    await waitFor(() => expect(respond).toHaveBeenCalledWith({ choice: 'deny' }))
  })

  it('carries the reason on the approval.respond fallback of a restored card', async () => {
    const rpc = mockGateway()
    setAccess({ serverRequestId: undefined })
    render(<PendingApprovalStack />)

    fireEvent.click(screen.getByRole('button', { name: /^Deny$/ }))
    fireEvent.change(reasonField(), { target: { value: 'Private' } })
    fireEvent.submit(reasonField().closest('form')!)

    await waitFor(() =>
      expect(rpc).toHaveBeenCalledWith(
        'approval.respond',
        { all: false, choice: 'deny', reason: 'Private', request_id: 'sandbox-card-1', session_id: 'sess-1' },
        APPROVAL_RESPOND_REQUEST_TIMEOUT_MS,
        undefined
      )
    )
  })

  it('never offers a per-session allow, and drops Always allow… when the request does not list it', () => {
    setSite({ choices: ['once', 'session', 'always', 'deny'] })
    const { unmount } = render(<PendingApprovalStack />)

    expect(screen.queryByText(/session/i)).toBeNull()
    expect(screen.queryByRole('button', { name: /More approval options/ })).toBeNull()
    unmount()
    clearAllPrompts()

    setSite({ choices: ['once', 'deny'] })
    render(<PendingApprovalStack />)

    expect(screen.queryByRole('button', { name: 'Always allow…' })).toBeNull()
    expect(screen.getByRole('button', { name: /^Allow/ })).toBeTruthy()
  })

  it('confirms Always allow with folder copy, no command block and no config.yaml claim', async () => {
    mockGateway()
    const respond = liveApproval()
    setAccess()
    render(<PendingApprovalStack />)

    fireEvent.click(screen.getByRole('button', { name: 'Always allow…' }))

    const dialog = await screen.findByRole('dialog')
    expect(dialog.textContent).toContain('Always allow this folder?')
    expect(dialog.textContent).toContain('profile’s sandbox')
    expect(dialog.textContent).not.toMatch(/config\.yaml|command/i)
    expect(dialog.querySelector('pre')).toBeNull()
    expect(respond).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Always allow' }))
    await waitFor(() => expect(respond).toHaveBeenCalledWith({ choice: 'always' }))
  })

  it('titles the site confirm with site copy', async () => {
    mockGateway()
    liveApproval()
    setSite()
    render(<PendingApprovalStack />)

    fireEvent.click(screen.getByRole('button', { name: 'Always allow…' }))

    const dialog = await screen.findByRole('dialog')
    expect(dialog.textContent).toContain('Always allow this site?')
    expect(dialog.textContent).toContain('allowed sites')
    expect(dialog.textContent).toContain('example.com')
  })
})
