import { getApiRequestConnection, getApiRequestProfile, type ProfileScope } from '@/hermes'
import { RECONNECT_ATTEMPT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'
import { requestGatewayForAgent } from '@/store/gateway'

export interface OnboardingScope {
  connectionId?: null | string
  profile?: null | string
}

/** Capture both halves once. An absent connection keeps legacy per-profile
 * routing; it must not acquire a different ambient registry owner later.
 * A blank profile is normalized to null: REST writes then carry no profile
 * (the backend's launch home), and readiness must omit it the same way. */
export function captureOnboardingScope(scope?: ProfileScope): OnboardingScope {
  const captured =
    scope && typeof scope === 'object'
      ? scope
      : { connectionId: getApiRequestConnection(), profile: scope === undefined ? getApiRequestProfile() : scope }

  return { connectionId: captured.connectionId ?? null, profile: captured.profile?.trim() || null }
}

/** Desktop profile keys can be SSH aliases. Only shared descriptors interpret
 * them as backend request scopes; dedicated backends already own their home. */
export async function requestOnboardingGateway<T>(
  scope: OnboardingScope,
  method: string,
  params: Record<string, unknown> = {}
): Promise<T> {
  // `default` is only the Desktop routing key for the launch home here.
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

  if (scope.profile && (connection.sharedPrimary || connection.sharedRemote)) {
    routedParams.profile = scope.profile
  }

  return requestGatewayForAgent<T>(scope.connectionId ?? null, profile, method, routedParams)
}
