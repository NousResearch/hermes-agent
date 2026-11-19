// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AuthGate } from './auth-gate'

vi.mock('@clerk/electron/react', () => ({
  ClerkProvider: ({ children }: { children: ReactNode }) => children,
  SignIn: () => null,
  SignUp: () => null,
  useAuth: () => ({ isLoaded: true, isSignedIn: false })
}))

vi.mock('@clerk/react', () => ({
  ClerkProvider: ({ children }: { children: ReactNode }) => children
}))

vi.mock('./clerk-session-bridge', () => ({ ClerkSessionBridge: () => <div>clerk session bridge</div> }))

vi.mock('@/lib/clerk-config', () => ({
  clerkBrowserEnabled: true,
  clerkEnabled: false,
  clerkPublishableKey: 'pk_test_placeholder'
}))

afterEach(cleanup)

describe('AuthGate in the browser host', () => {
  it('renders the app instead of gating it when there is no Electron Clerk bridge', () => {
    render(
      <AuthGate>
        <div>app content</div>
      </AuthGate>
    )

    expect(screen.getByText('app content')).toBeTruthy()
  })

  it('mounts the session bridge so the account panel can open Clerk sign-in', () => {
    render(
      <AuthGate>
        <div>app content</div>
      </AuthGate>
    )

    expect(screen.getByText('clerk session bridge')).toBeTruthy()
  })
})
