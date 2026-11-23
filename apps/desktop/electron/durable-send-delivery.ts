import type { DurableSendStore } from './durable-send-store'
import type { DurableSendFile, DurableSendOwner, DurableSendRecord, DurableSendTarget } from './durable-send-types'

export interface DurableSendTransport {
  /** Authenticated, destination-bound negotiation. uploadReplayVerified is set only by an
   * adapter that has verified content-addressed/idempotent upload and exact-ACK reconciliation.
   * It is NOT inferred from a version string or durable-admission-v1 alone. */
  describe: (
    owner: DurableSendOwner
  ) => Promise<{ owner: DurableSendOwner; capabilities: string[]; uploadReplayVerified: boolean }>
  create: (owner: DurableSendOwner, target: Exclude<DurableSendTarget, { sessionId: string }>) => Promise<string>
  upload: (
    owner: DurableSendOwner,
    request: { sessionId: string; submissionId: string; file: DurableSendFile; bytes: Uint8Array }
  ) => Promise<{ reference: string; sha256: string; size: number }>
  submit: (
    owner: DurableSendOwner,
    request: {
      sessionId: string
      submissionId: string
      text: string
      attachments: Array<DurableSendFile & { reference: string }>
    }
  ) => Promise<{ admissionId: string; status: 'queued' | 'started' | 'terminal' | 'unknown' }>
}

export class DurableSendRefusal extends Error {}

class AttachmentIntegrityError extends Error {}
// One consumer across all windows/store instances of an origin in the owning main process.
const draining = new Set<string>()

const sameOwner = (a: DurableSendOwner, b: DurableSendOwner) =>
  ['connectionId', 'gatewayId', 'profile', 'principalId'].every(
    key => a[key as keyof DurableSendOwner] === b[key as keyof DurableSendOwner]
  )

const lane = (entry: DurableSendRecord) =>
  JSON.stringify([
    entry.intent.owner,
    'sessionId' in entry.intent.target ? entry.intent.target.sessionId : entry.intent.target.draftId
  ])

/** Fault-safe delivery engine. No network is performed until the supplied adapter negotiates
 * the exact owner's guarantees. Shell startup/reconnect wiring deliberately belongs to its caller;
 * never call this with an ambient focused-chat transport or a legacy fallback. */
export async function drainDurableSends(
  store: DurableSendStore,
  transport: DurableSendTransport,
  now = Date.now()
): Promise<void> {
  if (draining.has(store.root)) {
    return
  }

  draining.add(store.root)
  const held = new Set<string>()

  try {
    for (const candidate of store.list()) {
      const entry = store.read(candidate.intent.id)

      if (entry.state === 'confirmed' || entry.state === 'cancelled') {
        store.cleanup(entry.intent.id)

        continue
      }

      const scope = lane(entry)

      if (held.has(scope)) {
        continue
      }

      // Preserve FIFO even if the head is still copying, backoff-delayed, or needs a human decision.
      held.add(scope)

      if (
        entry.state === 'staging' ||
        entry.retryAt > now ||
        (entry.state === 'blocked' && entry.problem !== 'capability')
      ) {
        continue
      }

      const { id, owner, target, files, text } = entry.intent

      try {
        const negotiated = await transport.describe(owner)

        if (
          !sameOwner(owner, negotiated.owner) ||
          !negotiated.capabilities.includes('durable-admission-v1') ||
          (!('sessionId' in target) && !negotiated.capabilities.includes('local-cli-create-v1')) ||
          (files.length > 0 && !negotiated.uploadReplayVerified)
        ) {
          store.update(id, { state: 'blocked', problem: 'capability' })

          continue
        }

        store.update(id, { state: 'sending', attempts: entry.attempts + 1, problem: undefined })
        let sessionId = entry.sessionId ?? ('sessionId' in target ? target.sessionId : undefined)

        if (!sessionId && !('sessionId' in target)) {
          sessionId = await transport.create(owner, target)
          // Persist the create result before any upload or prompt; a lost create ACK repeats
          // the SAME request id and frozen parameters on this SAME owner.
          store.update(id, { sessionId })
        }

        if (!sessionId) {
          throw new DurableSendRefusal('Missing durable session identity')
        }

        for (const file of files) {
          let bytes: Uint8Array

          // Verify even a previously uploaded file: corrupted local state is never sent as text-only.
          try {
            bytes = store.bytes(id, file.id)
          } catch {
            throw new AttachmentIntegrityError('Attachment integrity failure')
          }

          if (store.read(id).uploads[file.id]) {
            continue
          }

          const result = await transport.upload(owner, { sessionId, submissionId: id, file, bytes })

          if (
            !result.reference ||
            result.reference.length > 16_384 ||
            result.sha256 !== file.sha256 ||
            result.size !== file.size
          ) {
            throw new AttachmentIntegrityError('Upload receipt does not identify the original bytes')
          }

          store.update(id, { uploads: { ...store.read(id).uploads, [file.id]: result } })
        }

        const uploaded = store.read(id)
        const attachments = files.map(file => ({ ...file, reference: uploaded.uploads[file.id].reference }))
        // Once this is durable, cancellation cannot discard the only local copy on an uncertain ACK.
        store.update(id, { submitStarted: true })
        const receipt = await transport.submit(owner, { sessionId, submissionId: id, text, attachments })

        if (!receipt.admissionId || !['queued', 'started', 'terminal', 'unknown'].includes(receipt.status)) {
          throw new Error('Unrecognized admission receipt')
        }

        if (receipt.status === 'unknown') {
          store.update(id, { state: 'blocked', problem: 'uncertain', admissionId: receipt.admissionId })

          continue
        }

        store.update(id, { state: 'confirmed', admissionId: receipt.admissionId, problem: undefined, retryAt: 0 })
        store.cleanup(id)
        held.delete(scope)
      } catch (error) {
        // Never expose credentials from transport exceptions in persisted UI error text.
        const current = store.read(id)

        if (current.state === 'confirmed' || current.state === 'cancelled') {
          continue
        }

        const problem =
          error instanceof AttachmentIntegrityError
            ? 'integrity'
            : error instanceof DurableSendRefusal
              ? 'refused'
              : 'connection'

        store.update(id, {
          state: problem === 'connection' ? 'waiting' : 'blocked',
          problem,
          retryAt: problem === 'connection' ? now + Math.min(60_000, 1000 * 2 ** Math.min(current.attempts, 6)) : 0
        })
      }
    }
  } finally {
    draining.delete(store.root)
  }
}
