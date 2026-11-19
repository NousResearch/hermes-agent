// @vitest-environment jsdom
// @vitest-environment-options {"url":"https://atouch-agent-web.unitalk.ai/"}
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { type ReactNode, StrictMode, useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clerkSessionHost } from '@/store/clerk-session'

import { AuthGate } from './auth-gate'

const mocks = vi.hoisted(() => ({
  auth: { getToken: vi.fn(async () => 'cookie-token'), isLoaded: true, isSignedIn: true, orgId: 'org_1', sessionId: 'sess_1', userId: 'user_1' },
  memberships: vi.fn(async () => ({ data: [], total_count: 0 })),
  close: vi.fn(),
  closeSecondary: vi.fn(),
  setActive: vi.fn(async (_input: { organization: null | string }) => undefined),
  signOut: vi.fn(async () => undefined),
  submitClerkToken: vi.fn(),
  mount: vi.fn(),
  unmount: vi.fn()
}))

vi.mock('@clerk/electron/react', () => ({
  ClerkProvider: ({ children }: { children: ReactNode }) => children,
  SignIn: () => <div>Native sign-in</div>,
  SignUp: () => null,
  useAuth: () => ({ isLoaded: true, isSignedIn: false })
}))

const clerk = {
  session: { getToken: mocks.auth.getToken },
  setActive: mocks.setActive,
  signOut: mocks.signOut
}
const user = { id: 'user_1', getOrganizationMemberships: mocks.memberships }

vi.mock('@clerk/react', () => ({
  ClerkProvider: ({ children }: { children: ReactNode }) => children,
  OrganizationSwitcher: () => <div>Choose organization</div>,
  SignIn: () => <div>Clerk sign-in</div>,
  SignUp: () => <div>Clerk sign-up</div>,
  useAuth: () => mocks.auth,
  useClerk: () => clerk,
  useOrganization: () => ({ organization: null }),
  useUser: () => ({ user: mocks.auth.isSignedIn ? user : null })
}))

vi.mock('@/api/config', () => ({ submitClerkToken: mocks.submitClerkToken }))
vi.mock('@/store/gateway', () => ({
  activeGateway: () => ({ close: mocks.close }),
  closeSecondaryGateways: mocks.closeSecondary
}))
vi.mock('@/i18n', () => ({ useI18n: () => ({ locale: 'en', t: { common: { retry: 'Retry' } } }) }))
vi.mock('@/lib/clerk-config', () => ({
  clerkBrowserEnabled: true,
  clerkEnabled: false,
  clerkPublishableKey: 'pk_test_placeholder'
}))

const ownInstance = (ownerId = 'org_1') => ({
  _id: 'instance_1', app: 'hermes', ownerId, status: 'ready', userId: 'user_1', webDomain: 'atouch-agent-web.unitalk.ai'
})
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  headers: { 'Content-Type': 'application/json' }, status
})

let currentInstance: ReturnType<typeof ownInstance> | null
const fetcher = vi.fn<typeof fetch>(async input => {
  switch (new URL(String(input)).pathname) {
    case '/api/instance': return json({ instance: currentInstance })
    case '/api/auth/me': return json({ user_id: 'basic-user' })
    case '/auth/logout': return json({ ok: true })
    case '/api/auth-init': return json({ url: 'https://atouch-agent-web.unitalk.ai/' })
    default: throw new Error(`Unexpected request: ${String(input)}`)
  }
})

function AppContent() {
  useEffect(() => {
    mocks.mount()
    return mocks.unmount
  }, [])

  return <div>app content</div>
}

const app = () => <AuthGate><AppContent /></AuthGate>

beforeEach(() => {
  vi.clearAllMocks()
  mocks.auth.isLoaded = true
  mocks.auth.isSignedIn = true
  mocks.auth.orgId = 'org_1'
  mocks.setActive.mockImplementation(async ({ organization }) => { mocks.auth.orgId = organization ?? '' })
  mocks.signOut.mockImplementation(async () => { mocks.auth.isSignedIn = false })
  currentInstance = ownInstance()
  vi.stubGlobal('fetch', fetcher)
  vi.stubGlobal('__HERMES_AUTH_REQUIRED__', true)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('AuthGate in the browser host', () => {
  it('ends the Hermes session and shows Clerk sign-in without mounting the viewer while signed out', async () => {
    mocks.auth.isSignedIn = false
    render(app())
    await screen.findByText('Clerk sign-in')
    expect(mocks.mount).not.toHaveBeenCalled()
    expect(fetcher.mock.calls.map(([url]) => new URL(String(url)).pathname)).toEqual(['/auth/logout'])
    expect(window.location.hostname).toBe('atouch-agent-web.unitalk.ai')
  })

  it('waits for instance resolution before mounting the viewer, including Strict Mode remounts', async () => {
    let resolve!: (response: Response) => void
    fetcher.mockImplementationOnce(() => new Promise<Response>(done => { resolve = done }))
    render(<StrictMode>{app()}</StrictMode>)
    await waitFor(() => expect(fetcher).toHaveBeenCalledOnce())
    expect(screen.queryByText('app content')).toBeNull()
    expect(mocks.mount).not.toHaveBeenCalled()
    expect(window.dispatchEvent(new CustomEvent('hermes:browser-reauth-required', { cancelable: true }))).toBe(false)
    await act(async () => { resolve(json({ instance: ownInstance() })) })
    await screen.findByText('app content')
    expect(mocks.auth.getToken).toHaveBeenCalledWith({ skipCache: true })
    expect(mocks.submitClerkToken).not.toHaveBeenCalled()
  })

  it('blocks the old instance on an account-panel org switch, and recovers after choosing an org with an instance', async () => {
    const view = render(app())
    await screen.findByText('app content')
    currentInstance = null
    await act(async () => { await clerkSessionHost.setActiveOrganization('org_2') })
    await screen.findByText('Create an instance for this organization to continue.')
    expect(screen.queryByText('app content')).toBeNull()
    expect(mocks.unmount).toHaveBeenCalled()
    expect(mocks.close).toHaveBeenCalled()
    expect(mocks.closeSecondary).toHaveBeenCalled()
    expect(mocks.submitClerkToken).not.toHaveBeenCalled()
    expect(screen.getByText('Choose organization')).toBeTruthy()

    currentInstance = ownInstance('org_3')
    mocks.auth.orgId = 'org_3'
    view.rerender(app())
    await screen.findByText('app content')
    const initCall = fetcher.mock.calls.find(([url]) => new URL(String(url)).pathname === '/api/auth-init')
    expect(initCall?.[1]?.credentials).toBe('include')
    expect(new Headers(initCall?.[1]?.headers).has('Authorization')).toBe(false)
  })

  it('disconnects and signs out of Hermes when the user-panel signs out of Clerk', async () => {
    render(app())
    await screen.findByText('app content')
    await act(async () => { await clerkSessionHost.signOut() })
    await screen.findByText('Clerk sign-in')
    expect(mocks.signOut).toHaveBeenCalledOnce()
    expect(mocks.unmount).toHaveBeenCalled()
    expect(fetcher.mock.calls.filter(([url]) => new URL(String(url)).pathname === '/auth/logout')).toHaveLength(1)
    expect(mocks.submitClerkToken).not.toHaveBeenCalled()
  })

  it.each([
    { status: 404, error: 'Instance not found', message: 'Create an instance for this organization to continue.' },
    { status: 403, error: 'Active organization required', message: 'Select an organization to open your instance.' },
    { status: 503, error: 'Instance could not be retrieved', message: 'Instance could not be retrieved' }
  ])('shows the appropriate recovery screen for instance HTTP $status', async ({ status, error, message }) => {
    fetcher.mockResolvedValueOnce(json({ error }, status))
    render(app())
    await screen.findByText(message)
    expect(mocks.mount).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(screen.getByText('Choose organization')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await screen.findByText('app content')
  })

  it('offers sign-in recovery for instance HTTP 401 while keeping the old viewer closed', async () => {
    fetcher.mockResolvedValueOnce(json({ error: 'Unauthorized' }, 401))
    render(app())
    await screen.findByText('Your Unitalk session could not be authenticated. Sign in again to continue.')
    expect(mocks.mount).not.toHaveBeenCalled()
    expect(screen.queryByText('Create an instance for this organization to continue.')).toBeNull()
    expect(screen.queryByText('Choose organization')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Sign in again' }))
    await screen.findByText('Clerk sign-in')
    expect(mocks.signOut).toHaveBeenCalledOnce()
  })
})
