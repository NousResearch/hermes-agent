interface PoolBackend {
  process?: unknown
  port?: number | null
  token?: string | null
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

/** Use the app's authenticated transport, never a descriptor's remote URL. */
export function createPoolRetirementClient(requestJson: RequestJson) {
  async function request(entry: PoolBackend, action: string, token?: string): Promise<RetirementReply | null> {
    if (!entry.process || !entry.port || !entry.token) {
      return null
    }

    try {
      return await requestJson(`http://127.0.0.1:${entry.port}/api/health/retirement`, entry.token, {
        method: 'POST',
        body: { action, ...(token ? { token } : {}) },
        timeoutMs: 3000,
      }) as RetirementReply | null
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
