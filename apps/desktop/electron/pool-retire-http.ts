interface PoolBackend {
  process?: unknown
  port?: number | null
  token?: string | null
  /** Present on a pooled descriptor entry (no local child): the resolved remote descriptor. */
  connectionPromise?: null | Promise<{ authMode?: string; baseUrl?: string; headers?: Record<string, string>; token?: string | null }>
}

interface RetirementReply {
  ok?: boolean
  idle?: boolean | null
  token?: string
}

type RequestJson = (
  url: string,
  token: string,
  options: { method: string; body: Record<string, string>; timeoutMs: number }
) => Promise<unknown>

/** Descriptor-aware transport (``fetchJsonForBackend``): never raw-fetch a descriptor URL. */
type RequestJsonForDescriptor = (
  descriptor: { authMode?: string; baseUrl?: string; headers?: Record<string, string>; token?: string | null },
  path: string,
  options: { method: string; timeoutMs: number; body: Record<string, string> }
) => Promise<unknown>

/** Bound on awaiting a pooled entry's connect before the fence is asked. */
const CONNECT_DEADLINE_MS = 3000

/**
 * A connect that never settles must not hold the arbiter's serial queue open:
 * `retireIdle` awaits the fence, so one hung descriptor would stall every later
 * reclamation. No descriptor in time = no authority, exactly like a rejection.
 */
function settledWithin<T>(promise: Promise<T>, ms: number): Promise<T | null> {
  return new Promise(resolve => {
    const timer = setTimeout(() => resolve(null), ms)

    promise.then(
      value => {
        clearTimeout(timer)
        resolve(value)
      },
      () => {
        clearTimeout(timer)
        resolve(null)
      }
    )
  })
}

/**
 * Use the app's authenticated transport, never a descriptor's remote URL.
 *
 * A pooled backend with no local child (registry/remote descriptor) is a REAL
 * backend process too — the retired admission endpoint lives behind the same
 * descriptor its own calls use, so the second transport is optional only for
 * callers that never pool a descriptor shape.
 */
export function createPoolRetirementClient(
  requestJson: RequestJson,
  requestJsonForDescriptor?: RequestJsonForDescriptor,
  connectDeadlineMs: number = CONNECT_DEADLINE_MS
) {
  async function request(entry: PoolBackend, action: string, token?: string): Promise<RetirementReply | null> {
    const options = { method: 'POST', body: { action, ...(token ? { token } : {}) }, timeoutMs: 3000 }

    try {
      if (entry.process) {
        if (!entry.port || !entry.token) {
          return null
        }

        return await requestJson(`http://127.0.0.1:${entry.port}/api/health/retirement`, entry.token, options) as RetirementReply | null
      }

      // A descriptor entry has no loopback port of its own; the fence is the
      // backend behind its resolved connection.
      if (!entry.connectionPromise || !requestJsonForDescriptor) {
        return null
      }

      const descriptor = await settledWithin(entry.connectionPromise, connectDeadlineMs)

      if (!descriptor?.baseUrl) {
        return null
      }

      const { method, body, timeoutMs } = options

      return await requestJsonForDescriptor(descriptor, '/api/health/retirement', { method, body, timeoutMs }) as RetirementReply | null
    } catch {
      // An older runtime, transport failure or unreadable reply grants no
      // authority. Prepared permits expire; committed ones remain recoverable
      // through idempotent prepare/commit on this exact backend generation.
      return null
    }
  }

  return {
    prepare: async (_key: string, entry: PoolBackend): Promise<string | null> => {
      const reply = await request(entry, 'prepare')

      return reply?.ok === true && reply.idle === true && typeof reply.token === 'string' && reply.token
        ? reply.token : null
    },
    commit: async (_key: string, entry: PoolBackend, token: string): Promise<boolean> =>
      (await request(entry, 'commit', token))?.ok === true,
    cancel: async (_key: string, entry: PoolBackend, token: string): Promise<void> => {
      await request(entry, 'cancel', token)
    },
  }
}
