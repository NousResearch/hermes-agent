/**
 * Selects the cheapest readiness contract supported by a Hermes backend.
 *
 * Local Desktop children started by a current runtime expose `/api/ready`, a
 * narrow authenticated event-loop probe. Older runtimes do not, so the probe
 * falls back once to the existing rich `/api/status` contract and remembers
 * that capability decision for subsequent retries. The fallback also covers a
 * 401 from an old local runtime: legacy backends may regenerate the spawn
 * token, and Desktop can only adopt the served token after public readiness
 * succeeds. Remote connections retain `/api/status`: they use its
 * public/auth-aware contract and update on a separate release clock.
 */

export type HermesReadinessPath = '/api/ready' | '/api/status'

export interface HermesReadinessProbeOptions {
  preferReadyEndpoint: boolean
  request: (path: HermesReadinessPath) => Promise<unknown>
}

export interface CompleteLocalBackendHandshakeOptions {
  waitForReady: () => Promise<unknown>
  markReady: () => void
  adoptServedToken: () => Promise<string>
}

export function shouldFallbackFromReadyEndpoint(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return (
    /^(?:401|404)(?:\s|:)/.test(message) ||
    message.includes('The endpoint is likely missing on the Hermes backend.')
  )
}

export function createHermesReadinessProbe({
  preferReadyEndpoint,
  request
}: HermesReadinessProbeOptions): () => Promise<void> {
  let path: HermesReadinessPath = preferReadyEndpoint ? '/api/ready' : '/api/status'

  return async () => {
    try {
      await request(path)
    } catch (error) {
      if (path !== '/api/ready' || !shouldFallbackFromReadyEndpoint(error)) {
        throw error
      }

      path = '/api/status'
      await request(path)
    }
  }
}

export async function completeLocalBackendHandshake({
  waitForReady,
  markReady,
  adoptServedToken
}: CompleteLocalBackendHandshakeOptions): Promise<string> {
  await waitForReady()
  markReady()

  return adoptServedToken()
}
