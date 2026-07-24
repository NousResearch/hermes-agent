import { useQuery } from '@tanstack/react-query'

import type { AccountUsageResponse, AccountUsageSnapshot } from '@/types/hermes'

export const CODEX_USAGE_REFRESH_MS = 3 * 60_000
export const CODEX_USAGE_BACKOFF_MS = 15 * 60_000
export const CODEX_USAGE_REQUEST_TIMEOUT_MS = 45_000

export type GatewayRequester = <T = unknown>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<T>

export interface CodexAccountUsageOptions {
  connectionScope: string
  gatewayState: string
  profile: string
  provider: string
  requestGateway: GatewayRequester
  sessionId: null | string
}

interface UsageQueryState {
  state: {
    error: unknown
    fetchFailureCount: number
  }
}

export class AccountUsageUnavailableError extends Error {
  constructor() {
    super('Codex account usage is unavailable')
    this.name = 'AccountUsageUnavailableError'
  }
}

export class AccountUsageMethodUnavailableError extends Error {
  constructor() {
    super('The connected Hermes backend does not support account usage')
    this.name = 'AccountUsageMethodUnavailableError'
  }
}

export function codexAccountUsageQueryKey({
  connectionScope,
  profile,
  provider,
  sessionId
}: Pick<CodexAccountUsageOptions, 'connectionScope' | 'profile' | 'provider' | 'sessionId'>) {
  return [
    'account-usage',
    'openai-codex',
    connectionScope,
    profile,
    sessionId ?? '',
    provider.trim().toLowerCase()
  ] as const
}

function isUnknownMethodError(error: unknown): boolean {
  const code =
    typeof error === 'object' && error !== null && 'code' in error
      ? Number((error as { code?: unknown }).code)
      : Number.NaN
  const message = error instanceof Error ? error.message : String(error)

  return code === -32601 || /unknown method|method not found|no such method/i.test(message)
}

export function codexUsageRefetchInterval(query: UsageQueryState): false | number {
  if (query.state.error instanceof AccountUsageMethodUnavailableError) {
    return false
  }

  return query.state.fetchFailureCount >= 3 ? CODEX_USAGE_BACKOFF_MS : CODEX_USAGE_REFRESH_MS
}

export function useCodexAccountUsage(options: CodexAccountUsageOptions) {
  const provider = options.provider.trim().toLowerCase()
  const enabled = options.gatewayState === 'open' && provider === 'openai-codex' && Boolean(options.sessionId)
  const query = useQuery<AccountUsageSnapshot>({
    enabled,
    queryFn: async ({ signal }) => {
      try {
        const response = await options.requestGateway<AccountUsageResponse>(
          'session.account_usage',
          { session_id: options.sessionId },
          CODEX_USAGE_REQUEST_TIMEOUT_MS,
          signal
        )
        const snapshot = response?.account_usage ?? null

        if (!snapshot || snapshot.available === false) {
          throw new AccountUsageUnavailableError()
        }

        return snapshot
      } catch (error) {
        if (isUnknownMethodError(error)) {
          throw new AccountUsageMethodUnavailableError()
        }
        throw error
      }
    },
    queryKey: codexAccountUsageQueryKey({
      connectionScope: options.connectionScope,
      profile: options.profile,
      provider,
      sessionId: options.sessionId
    }),
    refetchInterval: codexUsageRefetchInterval,
    refetchIntervalInBackground: false,
    retry: false,
    staleTime: CODEX_USAGE_REFRESH_MS
  })

  return {
    error: query.isError,
    loading: query.isFetching,
    methodUnavailable: query.error instanceof AccountUsageMethodUnavailableError,
    refresh: query.refetch,
    snapshot: query.data ?? null
  }
}
