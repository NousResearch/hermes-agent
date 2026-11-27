import type { CanonicalClarifyBatch, CanonicalClarifyIdentity, ClarifyLockResult } from './gateway-contract.generated'
import type { ServerRequest } from './json-rpc-channel'

interface Owner {
  sessionId: string
  profile: string
  profileId: string
}
interface Entry {
  owner: Owner
  batch: CanonicalClarifyBatch
  id: string
  generation: number
  pending: boolean
  serial: number
  openedSerial: number
  cursor?: { epoch: string; sequence: number }
}
const object = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v)
const keys = (v: Record<string, unknown>, allowed: string[]) => Object.keys(v).every(k => allowed.includes(k))
const text = (v: unknown): v is string => typeof v === 'string' && v.length > 0
const positive = (v: unknown) => typeof v === 'number' && Number.isSafeInteger(v) && v > 0
export function batchIdentity(v: unknown): v is CanonicalClarifyIdentity {
  return (
    object(v) &&
    keys(v, ['profile_id', 'authority_epoch', 'stream_id', 'execution_generation']) &&
    text(v.profile_id) &&
    text(v.stream_id) &&
    positive(v.authority_epoch) &&
    positive(v.execution_generation)
  )
}
function validQuestion(q: unknown, ids: Set<string>): q is CanonicalClarifyBatch['questions'][number] {
  return (
    object(q) &&
    keys(q, ['qid', 'question', 'choices', 'multi_select']) &&
    text(q.qid) &&
    text(q.question) &&
    !ids.has(q.qid) &&
    q.qid.trim() === q.qid &&
    !!q.question.trim() &&
    (q.multi_select === undefined || typeof q.multi_select === 'boolean') &&
    (q.choices === undefined || q.choices === null || (Array.isArray(q.choices) && q.choices.every(text)))
  )
}
function parseBatch(v: unknown, owner: Owner): CanonicalClarifyBatch | null {
  if (
    !object(v) ||
    !keys(v, ['request_id', 'identity', 'questions', 'answers', 'replayed']) ||
    !text(v.request_id) ||
    !batchIdentity(v.identity) ||
    v.identity.profile_id !== owner.profileId ||
    typeof v.replayed !== 'boolean' ||
    !Array.isArray(v.questions) ||
    !v.questions.length ||
    !object(v.answers)
  )
    return null
  const ids = new Set<string>()
  for (const q of v.questions) {
    if (!validQuestion(q, ids)) return null
    ids.add(q.qid)
  }
  if (!Object.entries(v.answers).every(([k, a]) => ids.has(k) && (a === null || typeof a === 'string'))) return null
  return structuredClone(v) as unknown as CanonicalClarifyBatch
}
const identityKey = (o: Owner, id: string, i: CanonicalClarifyIdentity) =>
  JSON.stringify([o.profileId, o.sessionId, i.authority_epoch, i.stream_id, i.execution_generation, id])
let nextAdapterId = 0

/** One whole native form per exact authenticated owner/request. No mutation retries. */
export class CanonicalBatchAdapter {
  private readonly adapterId = ++nextAdapterId
  private generation = 0
  private serial = 0
  // replay_epoch names a SessionEvents stream lifetime, not a profile-wide counter.
  private watermarks = new Map<string, number>()
  private watermarkKey(owner: Owner, epoch: string): string {
    return JSON.stringify([owner.profileId, owner.sessionId, epoch])
  }
  checkpoint(): number {
    return this.serial
  }
  reset(): void {
    this.generation++
    for (const e of this.entries.values()) this.io.withdraw(e.id, e.owner)
    this.entries.clear()
    this.watermarks.clear()
  }
  private entries = new Map<string, Entry>()
  private tombstones = new Set<string>()
  constructor(
    private readonly io: {
      deliver: (request: ServerRequest, profile: string) => void
      withdraw: (id: string, owner: Owner) => void
      rpc: <T>(method: string, params: Record<string, unknown>) => Promise<T>
      recover: (owner: Owner) => Promise<unknown>
    }
  ) {}

  open(owner: Owner, value: unknown, replayed: boolean, cursor?: { epoch: string; sequence: number }): void {
    const batch = parseBatch(value, owner)
    if (!batch) return
    if (cursor && cursor.sequence <= (this.watermarks.get(this.watermarkKey(owner, cursor.epoch)) ?? -1)) return
    const key = identityKey(owner, batch.request_id, batch.identity)
    if (this.tombstones.has(key) || this.entries.has(key)) return
    const entry: Entry = {
      owner,
      batch,
      // The native panel retains this ID in its lock callback. Never give a new
      // entry the previous panel's capability, even for identical backend identity.
      id: `canonical-batch:${this.adapterId}:${this.generation}:${this.serial + 1}:${key}`,
      generation: this.generation,
      pending: false,
      serial: ++this.serial,
      openedSerial: this.serial,
      cursor
    }
    this.entries.set(key, entry)
    this.io.deliver(
      {
        id: entry.id,
        method: 'clarify',
        params: { session_id: owner.sessionId, questions: batch.questions, answers: batch.answers },
        replayed: replayed || batch.replayed,
        respond: result => {
          void this.respond(entry, result).catch(() => undefined)
        },
        fail: () => {
          if (this.entries.get(key) === entry) this.entries.delete(key)
        }
      },
      owner.profile
    )
  }

  snapshot(owner: Owner, values: unknown, checkpoint: number, watermark?: { epoch: string; sequence: number }): void {
    if (!Array.isArray(values)) return
    const parsed = values.map(v => parseBatch(v, owner))
    if (parsed.some(v => !v || !v.replayed)) return
    const batches = parsed as CanonicalClarifyBatch[]
    const present = new Set(batches.map(b => identityKey(owner, b.request_id, b.identity)))
    if (present.size !== batches.length) return
    if (watermark) {
      const key = this.watermarkKey(owner, watermark.epoch)
      if ((this.watermarks.get(key) ?? -1) > watermark.sequence) return
      this.watermarks.set(key, watermark.sequence)
    }
    for (const [key, entry] of this.entries) {
      if (this.snapshotPreserves(entry, owner, checkpoint, watermark)) continue
      if (!present.has(key)) {
        this.tombstones.add(key)
        this.entries.delete(key)
        this.io.withdraw(entry.id, entry.owner)
      } else {
        const next = batches.find(b => identityKey(owner, b.request_id, b.identity) === key)!
        if (!entry.pending && JSON.stringify(next.answers) !== JSON.stringify(entry.batch.answers)) {
          this.entries.delete(key)
          this.io.withdraw(entry.id, entry.owner)
        }
      }
    }
    for (const batch of batches) this.open(owner, batch, true)
  }

  private snapshotPreserves(entry: Entry, owner: Owner, checkpoint: number, watermark?: { epoch: string; sequence: number }): boolean {
    const sameStream = entry.cursor && watermark && entry.cursor.epoch === watermark.epoch
    // A newer opening or accepted local lock wins over an older read. An
    // older queued opening first seen during that read does not win merely
    // because it was assigned a later local arrival serial.
    return (
      entry.owner.profileId !== owner.profileId ||
      entry.owner.sessionId !== owner.sessionId ||
      !!(sameStream && entry.cursor!.sequence > watermark!.sequence) ||
      (entry.serial > checkpoint && (!sameStream || entry.serial !== entry.openedSerial))
    )
  }

  cancel(owner: Owner, id: unknown, identity: unknown): string | null {
    if (!text(id) || !batchIdentity(identity) || identity.profile_id !== owner.profileId) return null
    const key = identityKey(owner, id, identity)
    this.tombstones.add(key)
    const local = this.entries.get(key)?.id ?? `canonical-batch:${key}`
    this.entries.delete(key)
    return local
  }

  private current(entry: Entry): boolean {
    return (
      entry.generation === this.generation &&
      this.entries.get(identityKey(entry.owner, entry.batch.request_id, entry.batch.identity)) === entry
    )
  }
  private params(entry: Entry): Record<string, unknown> {
    return {
      session_id: entry.owner.sessionId,
      profile: entry.owner.profile,
      request_id: entry.batch.request_id,
      identity: entry.batch.identity
    }
  }
  owns(id: unknown): boolean {
    return typeof id === 'string' && id.startsWith('canonical-batch:')
  }
  private find(id: unknown): Entry {
    const e = [...this.entries.values()].find(e => e.id === id)
    if (!e || !this.current(e) || e.pending) throw new Error('Clarification is stale or awaiting recovery')
    return e
  }
  async lock(id: unknown, question: unknown, answer: unknown): Promise<ClarifyLockResult> {
    const entry = this.find(id)
    if (
      !entry.batch.questions.some(q => q.qid === question) ||
      !(
        answer === null ||
        typeof answer === 'string' ||
        (Array.isArray(answer) && answer.every(a => typeof a === 'string'))
      )
    ) {
      throw new Error('Invalid clarification lock')
    }
    return this.mutate(entry, 'clarify.batch.lock', { question_id: question, answer }) as Promise<ClarifyLockResult>
  }
  private async respond(entry: Entry, result: Record<string, unknown>): Promise<void> {
    if (!this.current(entry) || entry.pending) return
    // Existing native Skip/Stop sends {}; its whole-batch meaning is explicit cancel-all.
    const terminal = Object.keys(result).length === 0 ? { answers: {}, outcome: 'cancelled' } : result
    if (
      !keys(terminal, ['answers', 'outcome']) ||
      !object(terminal.answers) ||
      !['submitted', 'cancelled', 'timed_out'].includes(String(terminal.outcome)) ||
      !Object.entries(terminal.answers).every(
        ([qid, a]) => entry.batch.questions.some(q => q.qid === qid) && (a === null || typeof a === 'string')
      )
    )
      return
    await this.mutate(entry, 'clarify.batch.respond', { result: terminal })
  }
  private validateAcknowledgement(entry: Entry, method: string, extra: Record<string, unknown>, result: unknown): void {
    if (
      !object(result) ||
      !keys(result, method === 'clarify.batch.lock' ? ['status', 'remaining'] : ['status']) ||
      !['ok', 'expired'].includes(String(result.status)) ||
      (method === 'clarify.batch.lock' &&
        (result.status === 'ok' || result.remaining !== undefined) &&
        (!Array.isArray(result.remaining) ||
          new Set(result.remaining).size !== result.remaining.length ||
          !result.remaining.every(q => entry.batch.questions.some(e => e.qid === q)) ||
          (result.status === 'ok' && result.remaining.includes(extra.question_id))))
    ) {
      throw new Error('Malformed clarification acknowledgement')
    }
  }
  private async mutate(entry: Entry, method: string, extra: Record<string, unknown>): Promise<unknown> {
    entry.pending = true
    try {
      const result = await this.io.rpc<Record<string, unknown>>(method, { ...this.params(entry), ...extra })
      if (
        entry.generation !== this.generation ||
        (!this.current(entry) &&
          !this.tombstones.has(identityKey(entry.owner, entry.batch.request_id, entry.batch.identity)))
      ) {
        throw new Error('Clarification connection changed')
      }
      this.validateAcknowledgement(entry, method, extra, result)
      entry.pending = false
      if (
        result.status === 'expired' ||
        method === 'clarify.batch.respond' ||
        (Array.isArray(result.remaining) && result.remaining.length === 0)
      ) {
        this.cancel(entry.owner, entry.batch.request_id, entry.batch.identity)
        this.io.withdraw(entry.id, entry.owner)
      } else {
        entry.batch.answers[String(extra.question_id)] = Array.isArray(extra.answer)
          ? JSON.stringify(extra.answer)
          : (extra.answer as string | null)
        entry.serial = ++this.serial
      }
      return result
    } catch (error) {
      if (this.current(entry)) {
        this.entries.delete(identityKey(entry.owner, entry.batch.request_id, entry.batch.identity))
        this.io.withdraw(entry.id, entry.owner)
        try {
          await this.io.recover(entry.owner)
        } catch {
          /* Reattach will reconcile; never repeat the write. */
        }
      }
      throw error
    }
  }
}
