import { ClerkProvider as ElectronClerkProvider, SignIn, SignUp, useAuth } from '@clerk/electron/react'
import { ClerkProvider as BrowserClerkProvider } from '@clerk/react'
import { type ReactNode, useEffect, useState } from 'react'

import { clerkBrowserEnabled, clerkEnabled, clerkPublishableKey } from '@/lib/clerk-config'

import { BrowserClerkAuthGate } from './browser-clerk-auth-gate'
import { ClerkSessionBridge } from './clerk-session-bridge'

const AUTH_CHANNEL = 'unitalk-auth'
type AuthMode = 'sign-in' | 'sign-up'

const authMode = (): AuthMode => (window.location.hash.includes('/sign-up') ? 'sign-up' : 'sign-in')

export function AuthGate({ children }: { children: ReactNode }) {
  if (clerkEnabled) {
    return (
      <ElectronClerkProvider publishableKey={clerkPublishableKey}>
        <ClerkAuthGate>{children}</ClerkAuthGate>
      </ElectronClerkProvider>
    )
  }

  if (clerkBrowserEnabled) {
    return (
      <BrowserClerkProvider publishableKey={clerkPublishableKey}>
        <BrowserClerkAuthGate>{children}</BrowserClerkAuthGate>
      </BrowserClerkProvider>
    )
  }

  return <>{children}</>
}

function ClerkAuthGate({ children }: { children: ReactNode }) {
  const { isLoaded, isSignedIn } = useAuth()
  const [mode, setMode] = useState<AuthMode>(authMode)

  useEffect(() => {
    const onHashChange = () => setMode(authMode())

    window.addEventListener('hashchange', onHashChange)

    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  useEffect(() => {
    if (!isLoaded) {
      return
    }

    const channel = new BroadcastChannel(AUTH_CHANNEL)
    channel.postMessage({ type: isSignedIn ? 'signed-in' : 'signed-out' })
    channel.close()
  }, [isLoaded, isSignedIn])

  useEffect(() => {
    if (!isLoaded) {
      return
    }

    const channel = new BroadcastChannel(AUTH_CHANNEL)

    channel.onmessage = event => {
      const type = event.data?.type

      if ((type === 'signed-out' && isSignedIn) || (type === 'signed-in' && !isSignedIn)) {
        window.location.reload()
      }
    }

    return () => channel.close()
  }, [isLoaded, isSignedIn])

  useEffect(() => {
    if (isSignedIn && (window.location.hash.includes('/sign-in') || window.location.hash.includes('/sign-up'))) {
      window.location.hash = '#/'
    }
  }, [isSignedIn])

  if (!isSignedIn) {
    return (
      <div className="fixed inset-0 overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-6">
          {mode === 'sign-up' ? (
            <SignUp routing="hash" signInUrl="#/sign-in" />
          ) : (
            <SignIn routing="hash" signUpUrl="#/sign-up" />
          )}
        </div>
      </div>
    )
  }

  return (
    <>
      {children}
      <ClerkSessionBridge />
    </>
  )
}
