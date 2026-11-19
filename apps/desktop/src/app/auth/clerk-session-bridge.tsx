import { useAuth, useClerk, useOrganization, useUser } from '@clerk/react'
import { randomUuid } from '@hermes/shared'
import { useCallback, useEffect, useRef, useState } from 'react'

import { submitClerkToken } from '@/api/config'
import { setClerkTokenGetter } from '@/lib/clerk-session'
import {
  applyActiveOrganization,
  clerkErrorMessage,
  createClerkOrganization,
  refreshClerkSession
} from '@/lib/clerk-session-operations'
import {
  loadClerkOrganizations,
  summarizeClerkOrganization,
  summarizeClerkUser,
  withActiveOrganization
} from '@/lib/clerk-session-org'
import {
  CLERK_SESSION_CHANGED_KIND,
  createClerkSessionBus,
  isClerkSessionChangedMessage,
  sendClerkSessionChanged,
  setActiveClerkSessionBus
} from '@/lib/clerk-session-sync'
import {
  type ClerkOrganizationSummary,
  type ClerkSessionActions,
  DISABLED_CLERK_SNAPSHOT,
  publishClerkSession,
  registerClerkSessionActions,
  registerClerkSignInAction
} from '@/store/clerk-session'
import { sameWebappIdentity, type WebappInstanceSession } from '@/store/webapp-instance'

interface ClerkSessionBridgeProps {
  instanceSession?: WebappInstanceSession
}

export function ClerkSessionBridge({ instanceSession }: ClerkSessionBridgeProps) {
  const clerk = useClerk()
  const { isLoaded, isSignedIn, getToken, orgId, sessionId, userId } = useAuth()
  const { user } = useUser()
  const { organization } = useOrganization()
  const [organizations, setOrganizations] = useState<ClerkOrganizationSummary[]>([])
  const [tokenSyncError, setTokenSyncError] = useState<null | string>(null)
  const [sessionError, setSessionError] = useState<null | string>(null)
  const [signedOutLocally, setSignedOutLocally] = useState(false)
  const [sourceId] = useState(() => randomUuid())
  const organizationGenerationRef = useRef(0)

  useEffect(() => () => instanceSession?.dispose(), [instanceSession])

  useEffect(() => {
    if (!instanceSession || !isLoaded) {
      return
    }

    void instanceSession.synchronize(
      isSignedIn && sessionId && userId ? { organizationId: orgId ?? null, sessionId, userId } : null
    )
  }, [instanceSession, isLoaded, isSignedIn, orgId, sessionId, userId])

  useEffect(() => {
    setClerkTokenGetter(() => getToken())

    return () => setClerkTokenGetter(null)
  }, [getToken])

  // The browser's sign-in screen remains mounted while organization actions are off.
  useEffect(() => {
    if (typeof clerk.openSignIn !== 'function') {
      return
    }

    return registerClerkSignInAction(() => clerk.openSignIn({}))
  }, [clerk])

  // Unlike the Electron gate (which unmounts this bridge on sign-out), the
  // browser keeps it mounted, so a later sign-in must clear the local flag.
  useEffect(() => {
    if (isSignedIn) {
      setSignedOutLocally(false)
    }
  }, [isSignedIn])

  const refreshOrganizations = useCallback(async () => {
    const generation = ++organizationGenerationRef.current

    try {
      const rows = await loadClerkOrganizations(user)

      if (generation === organizationGenerationRef.current) {
        setOrganizations(rows)
        setSessionError(null)
      }
    } catch (error) {
      if (generation === organizationGenerationRef.current) {
        setSessionError(clerkErrorMessage(error))
      }
    }
  }, [user])

  useEffect(() => {
    if (!isSignedIn || !user) {
      setOrganizations([])

      return
    }

    void refreshOrganizations()
  }, [isSignedIn, refreshOrganizations, user])

  useEffect(() => {
    if (signedOutLocally) {
      publishClerkSession({ ...DISABLED_CLERK_SNAPSHOT, enabled: true, status: 'signed-out' })

      return
    }

    if (!isLoaded && !user) {
      publishClerkSession({ ...DISABLED_CLERK_SNAPSHOT, enabled: true, status: 'loading' })

      return
    }

    if (!isSignedIn || !user) {
      publishClerkSession({ ...DISABLED_CLERK_SNAPSHOT, enabled: true, status: 'signed-out' })

      return
    }

    const activeRole = organizations.find(entry => entry.id === organization?.id)?.role ?? null
    const activeOrganization = summarizeClerkOrganization(organization, activeRole)

    publishClerkSession({
      activeOrganization,
      enabled: true,
      error: sessionError,
      organizations: withActiveOrganization(organizations, activeOrganization),
      status: 'signed-in',
      tokenSyncError,
      user: summarizeClerkUser(user)
    })
  }, [isLoaded, isSignedIn, organization, organizations, sessionError, signedOutLocally, tokenSyncError, user])

  const activateOrganization = useCallback(
    async (organizationId: null | string) => {
      if (instanceSession) {
        if (!sessionId || !userId) {
          throw new Error('Clerk session is not ready')
        }

        await instanceSession.synchronize({ organizationId, sessionId, userId }, () =>
          clerk.setActive({ organization: organizationId })
        )
      } else {
        const result = await applyActiveOrganization(clerk, organizationId, submitClerkToken)

        setTokenSyncError(result.tokenSyncError)
      }
    },
    [clerk, instanceSession, sessionId, userId]
  )

  const syncFromPeer = useCallback(
    async (organizationId: null | string) => {
      setSessionError(null)

      try {
        await activateOrganization(organizationId)
        await refreshOrganizations()
      } catch (error) {
        setSessionError(clerkErrorMessage(error))

        if (!instanceSession) {
          window.location.reload()
        }
      }
    },
    [activateOrganization, instanceSession, refreshOrganizations]
  )

  useEffect(() => {
    if (!isSignedIn || !user) {
      return
    }

    const bus = createClerkSessionBus()

    setActiveClerkSessionBus(bus)

    if (!bus) {
      return
    }

    const unsubscribe = bus.subscribe(message => {
      if (!isClerkSessionChangedMessage(message) || message.source === sourceId) {
        return
      }

      void syncFromPeer(message.organizationId)
    })

    return () => {
      unsubscribe()
      bus.close()
      setActiveClerkSessionBus(null)
    }
  }, [isSignedIn, sourceId, syncFromPeer, user])

  useEffect(() => {
    if (!isSignedIn || !user) {
      return registerClerkSessionActions(null)
    }

    const switchOrganization = async (organizationId: null | string) => {
      setSessionError(null)
      setTokenSyncError(null)

      try {
        await activateOrganization(organizationId)

        if (
          instanceSession &&
          !sameWebappIdentity(
            instanceSession.state.get().identity,
            sessionId && userId ? { organizationId, sessionId, userId } : null
          )
        ) {
          return
        }

        sendClerkSessionChanged({
          kind: CLERK_SESSION_CHANGED_KIND,
          organizationId,
          source: sourceId
        })
        await refreshOrganizations()
      } catch (error) {
        setSessionError(clerkErrorMessage(error))
        throw error
      }
    }

    const actions: ClerkSessionActions = {
      createOrganization: async input => {
        const name = input.name.trim()
        const slug = input.slug?.trim()

        if (!name) {
          throw new Error('Organization name is required')
        }

        const created = await createClerkOrganization(clerk, slug ? { name, slug } : { name })

        await switchOrganization(created.id)

        const summary = summarizeClerkOrganization(created)

        if (!summary) {
          throw new Error('Clerk returned an organization without an id')
        }

        return summary
      },
      refresh: async () => {
        setSessionError(null)

        try {
          const result = await refreshClerkSession(clerk, instanceSession ? undefined : submitClerkToken)

          setTokenSyncError(result.tokenSyncError)

          if (instanceSession) {
            await instanceSession.retry()
          }

          await refreshOrganizations()
        } catch (error) {
          setSessionError(clerkErrorMessage(error))
          throw error
        }
      },
      setActiveOrganization: switchOrganization,
      signOut: async () => {
        setSignedOutLocally(true)

        try {
          if (instanceSession) {
            await instanceSession.synchronize(null, () => clerk.signOut())
          } else {
            await clerk.signOut()
          }

          publishClerkSession({ ...DISABLED_CLERK_SNAPSHOT, enabled: true, status: 'signed-out' })
        } catch (error) {
          setSignedOutLocally(false)
          throw error
        }
      }
    }

    return registerClerkSessionActions(actions)
  }, [activateOrganization, clerk, instanceSession, isSignedIn, refreshOrganizations, sessionId, sourceId, user, userId])

  return null
}
