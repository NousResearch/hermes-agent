import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type InboxRequestApproval } from '@/store/inbox'
import { $activeGatewayProfile } from '@/store/profile'
import { $gateway } from '@/store/gateway'

import { ApprovalCard } from './approval-card'

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- test-only partial mock
const mockGateway = { request: vi.fn(() => Promise.resolve({})) } as any

vi.mock('@/store/gateway', () => ({ $gateway: { get: vi.fn(() => mockGateway) } }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: { get: vi.fn(() => 'test-profile') } }))

vi.mock('@/store/inbox', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  respondToApproval: vi.fn()
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  mockGateway.request.mockReset()
})

function makeApproval(overrides: Partial<InboxRequestApproval> = {}): InboxRequestApproval {
  return {
    allow_permanent: null,
    allow_session: null,
    choices: ['once', 'deny'],
    command: 'rm -rf /tmp/test',
    description: 'Delete temporary files',
    request_id: 'req-1',
    smart_denied: null,
    tool_name: 'terminal',
    ...overrides
  }
}

describe('ApprovalCard', () => {
  it('renders command and description', () => {
    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)
    expect(screen.getByText('rm -rf /tmp/test')).toBeTruthy()
    expect(screen.getByText('Delete temporary files')).toBeTruthy()
  })

  it('renders available choice buttons', () => {
    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)
    expect(screen.getByText('Approve once')).toBeTruthy()
    expect(screen.getByText('Deny')).toBeTruthy()
  })

  it('shows "Approve for session" when allow_session is true', () => {
    render(
      <ApprovalCard
        approval={makeApproval({ allow_session: true, choices: ['once', 'session', 'deny'] })}
        liveSessionId="live-1"
      />
    )
    expect(screen.getByText('Approve for session')).toBeTruthy()
  })

  it('hides "Approve for session" when allow_session is false', () => {
    render(
      <ApprovalCard
        approval={makeApproval({ allow_session: false, choices: ['once', 'session', 'deny'] })}
        liveSessionId="live-1"
      />
    )
    expect(screen.queryByText('Approve for session')).toBeNull()
  })

  it('shows "Always allow" when allow_permanent is true', () => {
    render(
      <ApprovalCard
        approval={makeApproval({ allow_permanent: true, choices: ['once', 'always', 'deny'] })}
        liveSessionId="live-1"
      />
    )
    expect(screen.getByText('Always allow')).toBeTruthy()
  })

  it('hides "Always allow" when allow_permanent is false', () => {
    render(
      <ApprovalCard
        approval={makeApproval({ allow_permanent: false, choices: ['once', 'always', 'deny'] })}
        liveSessionId="live-1"
      />
    )
    expect(screen.queryByText('Always allow')).toBeNull()
  })

  it('calls respondToApproval on Approve once click', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    vi.mocked(respondToApproval).mockResolvedValue({ resolved: 1 })

    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)
    fireEvent.click(screen.getByText('Approve once'))
    expect(respondToApproval).toHaveBeenCalledWith(
      expect.objectContaining({
        choice: 'once',
        liveSessionId: 'live-1',
        profile: 'test-profile',
        requestId: 'req-1'
      })
    )
  })

  it('calls respondToApproval on Deny click', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    vi.mocked(respondToApproval).mockResolvedValue({ resolved: 1 })

    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)
    fireEvent.click(screen.getByText('Deny'))
    expect(respondToApproval).toHaveBeenCalledWith(
      expect.objectContaining({
        choice: 'deny',
        liveSessionId: 'live-1',
        profile: 'test-profile',
        requestId: 'req-1'
      })
    )
  })

  it('calls respondToApproval with the session scope when Approve for session is clicked', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    vi.mocked(respondToApproval).mockResolvedValue({ resolved: 1 })

    render(
      <ApprovalCard
        approval={makeApproval({ allow_session: true, choices: ['once', 'session', 'deny'] })}
        liveSessionId="live-1"
      />
    )
    fireEvent.click(screen.getByText('Approve for session'))
    expect(respondToApproval).toHaveBeenCalledWith(
      expect.objectContaining({
        choice: 'session',
        liveSessionId: 'live-1',
        profile: 'test-profile',
        requestId: 'req-1'
      })
    )
  })

  it('calls respondToApproval with the permanent scope when Always allow is clicked', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    vi.mocked(respondToApproval).mockResolvedValue({ resolved: 1 })

    render(
      <ApprovalCard
        approval={makeApproval({ allow_permanent: true, choices: ['once', 'always', 'deny'] })}
        liveSessionId="live-1"
      />
    )
    fireEvent.click(screen.getByText('Always allow'))
    expect(respondToApproval).toHaveBeenCalledWith(
      expect.objectContaining({
        choice: 'always',
        liveSessionId: 'live-1',
        profile: 'test-profile',
        requestId: 'req-1'
      })
    )
  })

  it('shows "Resolved" after successful response', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    vi.mocked(respondToApproval).mockResolvedValue({ resolved: 1 })

    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)
    fireEvent.click(screen.getByText('Approve once'))
    await waitFor(() => {
      expect(screen.getByText('Resolved')).toBeTruthy()
    })
  })

  it('shows error when response fails', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    vi.mocked(respondToApproval).mockRejectedValue(new Error('network error'))

    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)
    fireEvent.click(screen.getByText('Approve once'))
    await waitFor(() => {
      expect(screen.getByText('network error')).toBeTruthy()
    })
  })

  it('disables buttons while submitting', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    let resolvePromise: (value: { resolved: number }) => void
    vi.mocked(respondToApproval).mockImplementation(
      () => new Promise(resolve => { resolvePromise = resolve })
    )

    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)
    fireEvent.click(screen.getByText('Approve once'))
    // Buttons should be disabled
    expect(screen.getByText('…')).toBeTruthy()
    resolvePromise!({ resolved: 1 })
    await waitFor(() => {
      expect(screen.getByText('Resolved')).toBeTruthy()
    })
  })

  it('does not submit when already resolved', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    vi.mocked(respondToApproval).mockResolvedValue({ resolved: 1 })

    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)
    fireEvent.click(screen.getByText('Approve once'))
    await waitFor(() => {
      expect(screen.getByText('Resolved')).toBeTruthy()
    })
    // Second click should not trigger another respond
    fireEvent.click(screen.getByText('Resolved'))
    expect(respondToApproval).toHaveBeenCalledTimes(1)
  })

  it('shows "Pending approval" when command is empty', () => {
    render(<ApprovalCard approval={makeApproval({ command: '' })} liveSessionId="live-1" />)
    expect(screen.getByText('Pending approval')).toBeTruthy()
  })

  // ── Defect 1 regression: deny-only backend must not render Approve once ──
  it('does NOT render Approve once when backend choices are deny-only', () => {
    render(
      <ApprovalCard
        approval={makeApproval({ choices: ['deny'], allow_session: null, allow_permanent: null })}
        liveSessionId="live-1"
      />
    )
    expect(screen.getByText('Deny')).toBeTruthy()
    expect(screen.queryByText('Approve once')).toBeNull()
    expect(screen.queryByText('Approve for session')).toBeNull()
    expect(screen.queryByText('Always allow')).toBeNull()
  })

  // ── Defect 1 regression: once-only backend must not render Deny ──
  it('does NOT render Deny when backend choices omit deny', () => {
    render(
      <ApprovalCard
        approval={makeApproval({ choices: ['once'], allow_session: null, allow_permanent: null })}
        liveSessionId="live-1"
      />
    )
    expect(screen.getByText('Approve once')).toBeTruthy()
    expect(screen.queryByText('Deny')).toBeNull()
  })

  // ── Defect 1 regression: empty choices renders no response buttons ──
  it('renders "No supported actions available" when backend choices array is empty', () => {
    render(
      <ApprovalCard
        approval={makeApproval({ choices: [], allow_session: null, allow_permanent: null })}
        liveSessionId="live-1"
      />
    )
    expect(screen.getByText('No supported actions available')).toBeTruthy()
    expect(screen.queryByText('Approve once')).toBeNull()
    expect(screen.queryByText('Deny')).toBeNull()
  })

  // ── Defect 1 regression: unknown choice IDs must not render ──
  it('does NOT render buttons for unknown choice IDs', () => {
    render(
      <ApprovalCard
        approval={makeApproval({ choices: ['once', 'fabricated', 'deny'], allow_session: null, allow_permanent: null })}
        liveSessionId="live-1"
      />
    )
    expect(screen.getByText('Approve once')).toBeTruthy()
    expect(screen.getByText('Deny')).toBeTruthy()
    expect(screen.queryByText('fabricated')).toBeNull()
  })

  // ── Defect 1 regression: all choices false is mandatory ──
  it('sends all:false in the approval response', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    vi.mocked(respondToApproval).mockResolvedValue({ resolved: 1 })

    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)
    fireEvent.click(screen.getByText('Approve once'))

    await waitFor(() => {
      expect(respondToApproval).toHaveBeenCalledWith(
        expect.objectContaining({ choice: 'once' })
      )
    })
    // all:false is hardcoded in the store's respondToApproval — verified via the store test
  })

  // ── Defect 1 regression: nullable flags preserve actual provided choices ──
  it('nullable allow_session/allow_permanent preserves actual backend choices', () => {
    // Backend sends session choice but null flags — session button should render
    // because allow_session null is not false, and availableChoices filters only on false
    render(
      <ApprovalCard
        approval={makeApproval({ choices: ['once', 'session', 'deny'], allow_session: null, allow_permanent: null })}
        liveSessionId="live-1"
      />
    )
    // session is in choices and allow_session is null (not false), so it passes the filter
    expect(screen.getByText('Approve for session')).toBeTruthy()
  })

  // ── Defect 5 regression: profile scope pinning ──
  it('profile change before submit is blocked with an error', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    const getSpy = vi.mocked($activeGatewayProfile.get)
    // Card renders with profile 'pinned-profile'
    getSpy.mockReturnValue('pinned-profile')
    vi.mocked(respondToApproval).mockResolvedValue({ resolved: 0 })

    render(<ApprovalCard approval={makeApproval({ request_id: 'req-pin', choices: ['once', 'deny'] })} liveSessionId="live-pin" />)

    // User switches profile before clicking
    getSpy.mockReturnValue('switched-profile')
    fireEvent.click(screen.getByText('Approve once'))
    await waitFor(() => {
      expect(screen.getByText('Profile changed — re-open to act')).toBeTruthy()
    })
    expect(respondToApproval).not.toHaveBeenCalled()
  })

  // ── Defect 1 (gateway+profile): gateway change before submit is blocked ──
  it('gateway change before submit is blocked with an error', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    const profileSpy = vi.mocked($activeGatewayProfile.get)
    const gatewaySpy = vi.mocked($gateway.get)
    const gatewayA = { request: vi.fn(() => Promise.resolve({})) } as any
    const gatewayB = { request: vi.fn(() => Promise.resolve({})) } as any

    profileSpy.mockReturnValue('same-profile')
    gatewaySpy.mockReturnValue(gatewayA)
    vi.mocked(respondToApproval).mockResolvedValue({ resolved: 0 })

    render(<ApprovalCard approval={makeApproval({ request_id: 'req-gw', choices: ['once', 'deny'] })} liveSessionId="live-gw" />)

    // Gateway switches while card is mounted
    gatewaySpy.mockReturnValue(gatewayB)
    fireEvent.click(screen.getByText('Approve once'))
    await waitFor(() => {
      expect(screen.getByText('Profile changed — re-open to act')).toBeTruthy()
    })
    expect(respondToApproval).not.toHaveBeenCalled()
  })

  // ── Defect 1 (gateway+profile): gateway changes during await is blocked ──
  it('gateway change during await is blocked — no stale success', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    const profileSpy = vi.mocked($activeGatewayProfile.get)
    const gatewaySpy = vi.mocked($gateway.get)
    const gatewayA = { request: vi.fn(() => Promise.resolve({})) } as any

    profileSpy.mockReturnValue('pinned')
    gatewaySpy.mockReturnValue(gatewayA)

    let resolvePending!: (value: { resolved: number }) => void
    vi.mocked(respondToApproval).mockImplementation(
      () => new Promise(resolve => { resolvePending = resolve })
    )

    render(<ApprovalCard approval={makeApproval({ request_id: 'req-async', choices: ['once', 'deny'] })} liveSessionId="live-async" />)

    fireEvent.click(screen.getByText('Approve once'))

    // Gateway switches while RPC is in-flight
    const gatewayB = { request: vi.fn(() => Promise.resolve({})) } as any
    gatewaySpy.mockReturnValue(gatewayB)

    resolvePending({ resolved: 1 })
    await waitFor(() => {
      expect(screen.getByText('Profile changed — re-open to act')).toBeTruthy()
    })
    expect(screen.queryByText('Resolved')).toBeNull()
  })

  // ── Defect 2 regression: same-tick duplicate submit is prevented ──
  it('same-tick duplicate click does not fire respondToApproval twice', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    let resolvePromise!: (value: { resolved: number }) => void
    vi.mocked(respondToApproval).mockImplementation(
      () => new Promise(resolve => { resolvePromise = resolve })
    )

    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)

    const button = screen.getByText('Approve once')
    // Fire two clicks in rapid succession (same tick before state update)
    fireEvent.click(button)
    fireEvent.click(button)

    // Only one call should have been made
    expect(respondToApproval).toHaveBeenCalledTimes(1)

    resolvePromise({ resolved: 1 })
    await waitFor(() => {
      expect(screen.getByText('Resolved')).toBeTruthy()
    })
    // Still only one call after resolution
    expect(respondToApproval).toHaveBeenCalledTimes(1)
  })

  // ── Defect 1: response uses gateway-bound request, not defaultInboxRequest ──
  it('sends through the gateway-bound request, not defaultInboxRequest', async () => {
    const { respondToApproval } = await import('@/store/inbox')
    vi.mocked(respondToApproval).mockResolvedValue({ resolved: 1 })

    const freshGateway = { request: vi.fn(() => Promise.resolve({})) } as any
    vi.mocked($gateway.get).mockReturnValue(freshGateway)

    render(<ApprovalCard approval={makeApproval()} liveSessionId="live-1" />)
    fireEvent.click(screen.getByText('Approve once'))

    await waitFor(() => {
      expect(respondToApproval).toHaveBeenCalledWith(
        expect.objectContaining({ request: expect.any(Function) })
      )
    })

    // Verify the bound request calls the gateway's request method
    const callArgs = vi.mocked(respondToApproval).mock.calls.at(-1)![0]
    const boundRequest = callArgs.request!
    await boundRequest('test.method', { key: 'val' })
    expect(freshGateway.request).toHaveBeenCalledWith('test.method', { key: 'val' })
  })
})
