import { useQuery } from '@tanstack/react-query'
import { useRef } from 'react'

import { requestForSessionProfile } from '@/store/session-request-router'
import type { AccountUsageResponse, AccountUsageSnapshot } from '@/types/hermes'

export const ACCOUNT_USAGE_REFRESH_MS = 3 * 60_000
export const ACCOUNT_USAGE_BACKOFF_MS = 15 * 60_000
export const ACCOUNT_USAGE_REQUEST_TIMEOUT_MS = 45_000

export type GatewayRequester = <T = unknown>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<T>

export interface AccountUsageOptions {
  connectionScope: string
  gatewayState: string
  owner: AccountUsageOwnerScope
  profile: string
  provider: string
  requestGateway: GatewayRequester
  sessionId: null | string
}

export interface AccountUsageCacheAmbient {
  connectionScope: string
  profile: string
}

export interface AccountUsageOwnerRoute {
  connectionId: string
  profile: string
  targetProfile?: string
}

export type AccountUsageOwnerScope = AccountUsageOwnerRoute | null | string | undefined

/**
 * True when the session owner is known enough to route an RPC. A real
 * session with an unresolved owner must not fetch (ambient is not a fallback).
 */
export function accountUsageOwnerIsResolved(
  owner: AccountUsageOwnerScope
): owner is AccountUsageOwnerRoute | string {
  if (owner == null) {
    return false
  }

  if (typeof owner === 'string') {
    return Boolean(owner.trim())
  }

  return Boolean(owner.connectionId.trim())
}

/**
 * Align the React Query cache identity with the session-owner RPC route.
 *
 * An owner route keys by `connectionId` (same as the model picker). A bare
 * profile name is a pool profile of the ambient connection. Unresolved owner
 * (null/undefined) falls back to ambient identity — that path is only for
 * no-session chrome; a real session with no owner disables the query.
 */
export function accountUsageCacheIdentity(
  owner: AccountUsageOwnerScope,
  ambient: AccountUsageCacheAmbient
): AccountUsageCacheAmbient {
  if (owner && typeof owner === 'object' && owner.connectionId.trim()) {
    return {
      connectionScope: owner.connectionId.trim(),
      profile: (owner.targetProfile || owner.profile).trim() || ambient.profile
    }
  }

  if (typeof owner === 'string' && owner.trim()) {
    return {
      connectionScope: ambient.connectionScope,
      profile: owner.trim()
    }
  }

  return { connectionScope: ambient.connectionScope, profile: ambient.profile }
}

export function createAccountUsageRequester(
  owner: AccountUsageOwnerScope,
  ambientRequest: GatewayRequester
): GatewayRequester {
  return (method, params, timeoutMs, signal) => {
    if (!accountUsageOwnerIsResolved(owner)) {
      return Promise.reject(new Error('Account usage owner is unresolved'))
    }

    return requestForSessionProfile(owner, ambientRequest, method, params, timeoutMs, signal)
  }
}

interface UsageQueryState {
  state: {
    error: unknown
    fetchFailureCount: number
  }
}

export class AccountUsageUnavailableError extends Error {
  constructor() {
    super('Account usage is unavailable')
    this.name = 'AccountUsageUnavailableError'
  }
}

export class AccountUsageUnsupportedError extends Error {
  constructor() {
    super('The active provider does not support account usage')
    this.name = 'AccountUsageUnsupportedError'
  }
}

export class AccountUsageMethodUnavailableError extends Error {
  constructor() {
    super('The connected Hermes backend does not support account usage')
    this.name = 'AccountUsageMethodUnavailableError'
  }
}

export function accountUsageQueryKey({
  connectionScope,
  profile,
  provider,
  sessionId
}: Pick<AccountUsageOptions, 'connectionScope' | 'profile' | 'provider' | 'sessionId'>) {
  return ['account-usage', connectionScope, profile, provider.trim().toLowerCase(), sessionId ?? ''] as const
}

function accountUsageFailureKey(queryKey: readonly unknown[]): string {
  return JSON.stringify(queryKey)
}

const ACCOUNT_USAGE_FAILURE_MAP_MAX = 200

function rememberAccountUsageFailure(failures: Map<string, number>, key: string): void {
  failures.set(key, (failures.get(key) ?? 0) + 1)
  // * Cap abandoned session keys after React Query GC. Map insertion order
  // * makes keys().next() the oldest entry.
  if (failures.size > ACCOUNT_USAGE_FAILURE_MAP_MAX) {
    const oldest = failures.keys().next().value
    if (oldest !== undefined) {
      failures.delete(oldest)
    }
  }
}

function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === 'AbortError') ||
    (error instanceof Error && error.name === 'AbortError')
  )
}

function isUnknownMethodError(error: unknown): boolean {
  const code =
    typeof error === 'object' && error !== null && 'code' in error
      ? Number((error as { code?: unknown }).code)
      : Number.NaN
  const message = error instanceof Error ? error.message : String(error)

  return code === -32601 || /unknown method|method not found|no such method/i.test(message)
}

function readAccountUsageSnapshot(response: AccountUsageResponse | null | undefined): AccountUsageSnapshot {
  // * Envelope: unsupported is terminal; unavailable (or a missing snapshot) is
  // * retryable. A missing `status` is the legacy Codex-only shape — presence
  // * of `account_usage` is enough.
  if (response?.status === 'unsupported') {
    throw new AccountUsageUnsupportedError()
  }

  const snapshot = response?.account_usage ?? null

  if (response?.status === 'unavailable' || !snapshot) {
    throw new AccountUsageUnavailableError()
  }

  return snapshot
}

export function accountUsageRefetchInterval(
  query: UsageQueryState,
  consecutiveFailures: number
): false | number {
  if (
    query.state.error instanceof AccountUsageMethodUnavailableError ||
    query.state.error instanceof AccountUsageUnsupportedError
  ) {
    return false
  }

  return consecutiveFailures >= 3 ? ACCOUNT_USAGE_BACKOFF_MS : ACCOUNT_USAGE_REFRESH_MS
}

export function useAccountUsage(options: AccountUsageOptions) {
  const provider = options.provider.trim().toLowerCase()
  const enabled =
    options.gatewayState === 'open' &&
    Boolean(options.sessionId) &&
    Boolean(provider) &&
    accountUsageOwnerIsResolved(options.owner)
  // * retry: false resets React Query's fetchFailureCount each cycle — this
  // * map is the consecutive-failure count that actually drives backoff.
  const consecutiveFailures = useRef(new Map<string, number>())

  const queryKey = accountUsageQueryKey({
    connectionScope: options.connectionScope,
    profile: options.profile,
    provider,
    sessionId: options.sessionId
  })
  const failureKey = accountUsageFailureKey(queryKey)

  const query = useQuery<AccountUsageSnapshot>({
    enabled,
    gcTime: ACCOUNT_USAGE_BACKOFF_MS,
    queryFn: async ({ signal }) => {
      try {
        const response = await options.requestGateway<AccountUsageResponse>(
          'session.account_usage',
          { session_id: options.sessionId },
          ACCOUNT_USAGE_REQUEST_TIMEOUT_MS,
          signal
        )

        const snapshot = readAccountUsageSnapshot(response)
        consecutiveFailures.current.delete(failureKey)
        return snapshot
      } catch (error) {
        if (isAbortError(error)) {
          throw error
        }

        const mapped =
          error instanceof AccountUsageUnsupportedError || error instanceof AccountUsageUnavailableError
            ? error
            : isUnknownMethodError(error)
              ? new AccountUsageMethodUnavailableError()
              : error

        rememberAccountUsageFailure(consecutiveFailures.current, failureKey)
        throw mapped
      }
    },
    queryKey,
    refetchInterval: queryState =>
      accountUsageRefetchInterval(
        queryState,
        consecutiveFailures.current.get(accountUsageFailureKey(queryState.queryKey)) ?? 0
      ),
    refetchIntervalInBackground: false,
    retry: false,
    staleTime: ACCOUNT_USAGE_REFRESH_MS
  })

  return {
    error: query.isError,
    loading: query.isFetching,
    methodUnavailable: query.error instanceof AccountUsageMethodUnavailableError,
    refresh: query.refetch,
    snapshot: query.data ?? null,
    unsupported: query.error instanceof AccountUsageUnsupportedError
  }
}
