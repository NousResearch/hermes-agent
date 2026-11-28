import { canonicalCreateParams } from './canonical-create-params'
import { DurableSendRefusal, type DurableSendTransport } from './durable-send-delivery'
import type { DurableSendOwner } from './durable-send-types'

export type DurableRpc = ((
  route: Pick<DurableSendOwner, 'connectionId' | 'profile'>,
  method: string,
  params: Record<string, unknown>
) => Promise<any>) & { acquire?: DurableSendTransport['acquire'] }

export const DURABLE_RPC_METHODS = new Set([
  'durable.describe',
  'durable.create',
  'session.resume',
  'durable.upload',
  'durable.submit',
  'durable.cancel'
])

/** The same scoped RPC protocol as the Python authority fixture. No version-based promises,
 * legacy upload, ambient socket, or renderer-provided terminal receipt is accepted. */
export function durableRpcTransport(rpc: DurableRpc): DurableSendTransport {
  const fence = (owner: DurableSendOwner) => {
    const [store_id, profile_id] = JSON.parse(owner.gatewayId)

    if (!store_id || !profile_id) {
      throw new DurableSendRefusal('Invalid owner identity')
    }

    return { store_id, profile_id, principal_id: owner.principalId }
  }

  const resume = async (owner: DurableSendOwner, sessionId: string) => {
    const result = await rpc(owner, 'session.resume', { session_id: sessionId, omit_messages: true })

    if (result?.session_id !== sessionId) {
      throw new DurableSendRefusal('Session destination changed')
    }
  }

  return {
    acquire: rpc.acquire,
    async describe(owner) {
      let result

      try {
        result = await rpc(owner, 'durable.describe', {})
      } catch (error) {
        if (error instanceof DurableSendRefusal) {
          return { owner, capabilities: [], uploadReplayVerified: false }
        }

        throw error
      }

      if (
        !result ||
        typeof result.store_id !== 'string' ||
        !result.store_id ||
        typeof result.instance_id !== 'string' ||
        typeof result.profile_id !== 'string' ||
        typeof result.principal_id !== 'string' ||
        !Array.isArray(result.capabilities)
      ) {
        throw new DurableSendRefusal('Owner negotiation unavailable')
      }

      return {
        owner: {
          connectionId: owner.connectionId,
          profile: owner.profile,
          gatewayId: JSON.stringify([result.store_id, result.profile_id]),
          principalId: result.principal_id
        },
        capabilities: result.capabilities,
        uploadReplayVerified: result.capabilities.includes('durable-desktop-send-v1')
      }
    },
    async create(owner, target) {
      const params = {
        ...canonicalCreateParams(target.params),
        request_id: target.createRequestId,
        source: 'gui'
      } as Record<string, unknown>

      delete params.profile
      const result = await rpc(owner, 'durable.create', { owner: fence(owner), params })

      if (typeof result?.session_id !== 'string' || !result.session_id) {
        throw new DurableSendRefusal('Missing session identity')
      }

      return result.session_id
    },
    async upload(owner, { sessionId, submissionId, file, bytes }) {
      await resume(owner, sessionId)
      let offset = 0

      do {
        const chunk = bytes.subarray(offset, offset + 1024 * 1024)

        const receipt = await rpc(owner, 'durable.upload', {
          owner: fence(owner),
          session_id: sessionId,
          submission_id: submissionId,
          file,
          offset,
          data: Buffer.from(chunk).toString('base64')
        })

        if (
          receipt.sha256 !== file.sha256 ||
          receipt.size !== file.size ||
          !Number.isSafeInteger(receipt.offset) ||
          receipt.offset < offset + chunk.length ||
          receipt.offset > file.size
        ) {
          throw new DurableSendRefusal('Upload receipt mismatch')
        }

        offset = receipt.offset

        if (offset === file.size) {
          if (!receipt.complete || typeof receipt.reference !== 'string') {
            throw new DurableSendRefusal('Incomplete upload')
          }

          return { reference: receipt.reference, sha256: receipt.sha256, size: receipt.size }
        }
      } while (offset < bytes.length)

      throw new DurableSendRefusal('Incomplete upload')
    },
    async cancel(owner, { sessionId, submissionId, files }) {
      await resume(owner, sessionId)
      const result = await rpc(owner, 'durable.cancel', { owner: fence(owner), session_id: sessionId, submission_id: submissionId, files })

      return result?.cancelled === true
    },
    async submit(owner, { sessionId, submissionId, text, attachments }) {
      await resume(owner, sessionId)

      const result = await rpc(owner, 'durable.submit', {
        owner: fence(owner),
        session_id: sessionId,
        submission_id: submissionId,
        text,
        attachments
      })

      if (typeof result?.admission_id !== 'string') {
        throw new DurableSendRefusal('Admission receipt unavailable')
      }

      return { admissionId: result.admission_id, status: result.status }
    }
  }
}
