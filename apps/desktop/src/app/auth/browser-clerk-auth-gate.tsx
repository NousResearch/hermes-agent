import { useAuth, useClerk } from '@clerk/react'
import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useState } from 'react'

import { browserBootstrap } from '@/lib/browser-transport'
import { clerkErrorMessage } from '@/lib/clerk-session-operations'
import { clerkSessionHost } from '@/store/clerk-session'
import { activeGateway, closeSecondaryGateways } from '@/store/gateway'
import { createWebappInstanceSession, sameWebappIdentity, type WebappIdentity } from '@/store/webapp-instance'

import { ClerkSessionBridge } from './clerk-session-bridge'
import { WebappInstanceStatus } from './webapp-instance-status'

interface BrowserClerkAuthGateProps {
  children: ReactNode
}

export function BrowserClerkAuthGate({ children }: BrowserClerkAuthGateProps) {
  const clerk = useClerk()
  const { isLoaded, isSignedIn, orgId, sessionId, userId } = useAuth()
  const [actionError, setActionError] = useState<null | string>(null)
  const [session] = useState(() =>
    createWebappInstanceSession({
      basePath: browserBootstrap()?.basePath ?? '',
      fetch: (...args) => fetch(...args),
      navigate: url => window.location.replace(url),
      origin: window.location.origin,
      refreshSession: async () => {
        const token = await clerk.session?.getToken({ skipCache: true })

        if (!token) {
          throw new Error('Clerk session is not ready')
        }
      },
      pause: () => {
        closeSecondaryGateways()
        activeGateway()?.close()
      }
    })
  )

  const state = useStore(session.state)

  const identity: WebappIdentity | null =
    isSignedIn && sessionId && userId ? { organizationId: orgId ?? null, sessionId, userId } : null

  const ready = isLoaded && isSignedIn && state.status === 'ready' && sameWebappIdentity(state.identity, identity)

  useEffect(() => {
    if (ready && /\/sign-(in|up)/.test(window.location.hash)) {
      window.location.hash = '#/'
    }
  }, [ready])

  useEffect(() => {
    // In-flight requests from the previous viewer must not navigate away during a handoff.
    const onReauth = (event: Event) => {
      if (session.state.get().status !== 'ready') {
        event.preventDefault()
      }
    }

    window.addEventListener('hermes:browser-reauth-required', onReauth)

    return () => window.removeEventListener('hermes:browser-reauth-required', onReauth)
  }, [session])

  const pending = !isLoaded || state.status === 'checking' || state.status === 'redirecting' ||
    (state.status === 'ready' && !ready)

  const signOut = () => {
    setActionError(null)
    void clerkSessionHost.signOut().catch(error => setActionError(clerkErrorMessage(error)))
  }

  const retry = () => {
    setActionError(null)
    void session.retry(identity)
  }

  return (
    <>
      <ClerkSessionBridge instanceSession={session} />
      {ready ? (
        children
      ) : (
        <div className="fixed inset-0 flex items-center justify-center overflow-y-auto p-6">
          <WebappInstanceStatus
            actionError={actionError}
            onRetry={retry}
            onSignOut={signOut}
            pending={pending}
            signedIn={Boolean(isSignedIn)}
            state={state}
          />
        </div>
      )}
    </>
  )
}
