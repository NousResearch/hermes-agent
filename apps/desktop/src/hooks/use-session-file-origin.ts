import { useStore } from '@nanostores/react'
import { useCallback } from 'react'

import { capabilityScoped } from '@/api/client'
import { useComposerScope } from '@/app/chat/composer/scope'
import { useSessionView } from '@/app/chat/session-view'
import type { GatewayFileOrigin } from '@/lib/media'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { profileScopeForSessionOwner } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

/** Capture the transcript's backend before any asynchronous file operation. */
export function useSessionFileOrigin() {
  const view = useSessionView()
  const scope = useComposerScope()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const sessionId = storedId || runtimeId

  const resolveOrigin = useCallback((): GatewayFileOrigin => {
    const owner =
      knownOwnerForSession(storedId) ??
      knownOwnerForSession(runtimeId) ??
      (scope.connectionId ? { connectionId: scope.connectionId, profile: scope.profile || 'default' } : undefined)

    assertSessionOwnerResolved(owner, { method: 'file.read', sessionId })

    const scoped = owner ? capabilityScoped(profileScopeForSessionOwner(owner)) : undefined

    return {
      owner: scoped ? { connectionId: scoped.connectionId, profile: scoped.profile } : undefined,
      sessionId: sessionId || undefined
    }
  }, [storedId, runtimeId, sessionId, scope.connectionId, scope.profile])

  return { resolveOrigin, runtimeId, storedId }
}
