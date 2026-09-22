import { getApiRequestConnection, getApiRequestProfile, type ProfileScope } from '@/hermes'
import { RECONNECT_ATTEMPT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'
import { requestGatewayForAgent } from '@/store/gateway'

export interface OnboardingScope {
  connectionId?: null | string
  profile?: null | string
}

/** Capture both halves once. An absent connection keeps legacy per-profile
 * routing; it must not acquire a different ambient registry owner later. */
export function captureOnboardingScope(scope?: ProfileScope): OnboardingScope {
  if (scope && typeof scope === 'object') {
    return { ...scope }
  }

  return {
    connectionId: getApiRequestConnection(),
    profile: scope === undefined ? getApiRequestProfile() : scope
  }
}

/** Desktop profile keys can be SSH aliases. Only shared descriptors interpret
 * them as backend request scopes; dedicated backends already own their home. */
export async function requestOnboardingGateway<T>(
  scope: OnboardingScope,
  method: string,
  params: Record<string, unknown> = {}
): Promise<T> {
  const profile = scope.profile || 'default'
  const desktop = window.hermesDesktop

  if (scope.connectionId && !desktop.getConnectionFor) {
    throw new Error('This Desktop build cannot dial registry connections. Update Hermes Desktop.')
  }

  const connection = await withTimeout(
    scope.connectionId
      ? desktop.getConnectionFor!({ connectionId: scope.connectionId, profile })
      : desktop.getConnection(profile),
    RECONNECT_ATTEMPT_TIMEOUT_MS,
    `Timed out resolving provider setup for "${profile}"`
  )

  const routedParams = { ...params }

  delete routedParams.profile

  if (connection.sharedPrimary || connection.sharedRemote) {
    routedParams.profile = profile
  }

  return requestGatewayForAgent<T>(scope.connectionId ?? null, profile, method, routedParams)
}
