import { JsonRpcGatewayError, randomUuid } from '@hermes/shared'

import { readJson, readKey, writeJson, writeKey } from '@/lib/storage'

import type { LongFrameBucket } from './desktop-metrics'
import type { SharedMetricsRequester } from './shared-metrics'

const DAY_MS = 24 * 60 * 60_000
const RETENTION_MS = 7 * DAY_MS
const BUCKETS: readonly LongFrameBucket[] = ['50ms_to_100ms', '100ms_to_250ms', '250ms_to_1s', '1s_to_5s', 'gte_5s']
const utcDay = (now: number) => new Date(now).toISOString().slice(0, 10)

interface FrameDay {
  frames: Partial<Record<LongFrameBucket, number>>
  visibleMs: number
}

interface FrameReport extends FrameDay {
  day: string
  epoch: string
}

interface Journal {
  days: Record<string, FrameDay>
  epoch: string
  maxSpanMs: number
  revision: number
  since: number | null
  /** Wall-clock high-water mark, retained across hidden/paused tracking transitions. */
  updatedAt: number
}

interface Settlement {
  epoch: string
}

interface ScopeState extends Settlement {
  /** A permit follows the logical profile through renames; scope ownership does not. */
  consent?: string
  importing?: { epoch: string; from: string }
}

interface ImportSnapshot {
  entries: [string, Record<string, unknown>][]
  epoch: string
}

export interface DesktopFrameConsent {
  active: boolean
  consent: string
  epoch: string
}

export interface DesktopFrameRevocation {
  after: DesktopFrameConsent | null
  before: DesktopFrameConsent | null
  complete: boolean
}

let lockDatabase: Promise<IDBDatabase> | undefined

/** The callback is synchronous. IDB's readwrite transaction is our fallback mutex, not a place
 * to await network work: ownership starts at request success and ends after this event task. */
async function metricsPhase<T>(run: () => T): Promise<T> {
  const locks = typeof navigator === 'undefined' ? undefined : navigator.locks

  if (locks) {
    return locks.request('hermes-desktop-metrics-coordination', run)
  }

  if (!lockDatabase) {
    const opened = Promise.withResolvers<IDBDatabase>()
    const request = indexedDB.open('hermes-desktop-metrics-coordination', 1)

    lockDatabase = opened.promise
    request.onupgradeneeded = () => request.result.createObjectStore('phases')

    request.onerror = () => {
      lockDatabase = undefined
      opened.reject(request.error)
    }

    request.onsuccess = () => {
      request.result.onversionchange = () => {
        request.result.close()
        lockDatabase = undefined
      }

      opened.resolve(request.result)
    }
  }

  const database = await lockDatabase
  const finished = Promise.withResolvers<T>()
  // ponytail: one store serializes all profile scopes; phases are short and contain no RPC.
  const transaction = database.transaction('phases', 'readwrite')
  let result: T
  let failure: unknown

  transaction.oncomplete = () => finished.resolve(result)
  transaction.onabort = () => finished.reject(failure ?? transaction.error)

  transaction.objectStore('phases').get('owner').onsuccess = () => {
    try {
      result = run()
    } catch (error) {
      failure = error
      transaction.abort()
    }
  }

  return finished.promise
}

function frameKeys(prefix: string): string[] {
  const keys: string[] = []

  try {
    const storage = window.localStorage

    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index)

      if (key?.startsWith(prefix)) {
        keys.push(key)
      }
    }
  } catch {
    // Unavailable storage cannot contribute a persisted journal.
  }

  return keys
}

/** One writer per journal. Native Web Locks or an IndexedDB readwrite transaction serialize
 * publication, immutable report freezing, and acknowledgement. The RPC runs outside ownership;
 * every writer honors a durably frozen day, including while its acknowledgement is outstanding. */
export class DesktopFrameMetrics {
  private static readonly pendingRevocations = new Set<string>()
  private readonly key: string
  private readonly slot: string
  private readonly journal: Journal
  readonly ready: Promise<void>
  private initializing = true
  private consent: string | undefined
  private stopped = false
  private closing = false
  private published = false
  private flushing = false
  private flushAgain = false
  private prunedDay = ''

  constructor(private readonly stateKey: string, expected?: DesktopFrameConsent) {
    this.key = `${stateKey}:frames`
    this.slot = `${this.key}:${randomUuid()}`
    const captured = expected ?? DesktopFrameMetrics.capture(stateKey)
    const pending = expected === undefined && DesktopFrameMetrics.pendingRevocations.has(captured.consent)

    this.consent = captured.consent
    this.journal = { days: {}, epoch: captured.epoch, maxSpanMs: 0, revision: 0, since: null, updatedAt: Date.now() }

    if (pending) {
      void DesktopFrameMetrics.purge(stateKey, captured)
    }

    this.ready = metricsPhase(() => {
      try {
        DesktopFrameMetrics.repairOutgoingOwned(stateKey)
        const stored = readJson<ScopeState>(this.key)
        let value = stored ?? { epoch: 'initial' }

        if ((expected !== undefined && !captured.active) || (!pending && value.epoch !== captured.epoch) ||
          DesktopFrameMetrics.pendingRevocations.has(value.consent ?? `${this.key}:consent:${value.epoch}`)) {
          this.stopped = true

          return
        }

        if (value.importing) {
          if (!DesktopFrameMetrics.finishImportOwned(stateKey)) {
            this.stopped = true

            return
          }

          value = readJson<ScopeState>(this.key)!
        }

        const active = value.consent !== undefined && readKey(value.consent) === '1'

        if (!pending && captured.active && (!active || captured.consent !== value.consent)) {
          this.stopped = true

          return
        }

        if (!active) {
          const prepared = DesktopFrameMetrics.prepareCohortOwned(stateKey)

          if (!prepared.active) {
            this.stopped = true

            return
          }

          value = readJson<ScopeState>(this.key)!
        }

        this.consent = value.consent
        this.journal.epoch = value.epoch
      } finally {
        this.initializing = false
      }
    }).catch(() => {
      this.initializing = false
      this.stopped = true
    })
  }

  static capture(stateKey: string): DesktopFrameConsent {
    const value = readJson<ScopeState>(`${stateKey}:frames`) ?? { epoch: 'initial' }

    const consent = value.consent ?? `${stateKey}:frames:consent:${value.epoch}`

    return { active: readKey(consent) === '1', consent, epoch: value.epoch }
  }

  /** Allocate the immutable cohort before a consent RPC, not when its delayed result arrives.
   * This marker authorizes no telemetry by itself: collection still requires a positive result. */
  private static prepareCohortOwned(stateKey: string): DesktopFrameConsent {
    const key = `${stateKey}:frames`
    const stored = readJson<ScopeState>(key)
    const captured = DesktopFrameMetrics.capture(stateKey)

    if (captured.active || stored?.importing || DesktopFrameMetrics.pendingRevocations.has(captured.consent)) {
      return captured
    }

    const epoch = stored?.consent ? randomUuid() : captured.epoch
    const consent = `${key}:consent:${randomUuid()}`

    if (stored) {
      DesktopFrameMetrics.clearPayloadOwned(stateKey)
    }

    writeKey(consent, '1')
    writeJson(key, { consent, epoch })

    if (readKey(consent) !== '1' || readJson<ScopeState>(key)?.consent !== consent) {
      writeKey(consent, null)
      // Never leave optional, unowned legacy measurements behind a reusable failed grant.
      DesktopFrameMetrics.retireScopeOwned(stateKey)
    }

    return DesktopFrameMetrics.capture(stateKey)
  }

  static async captureCoordinated(stateKey: string, prepare = true): Promise<DesktopFrameConsent> {
    const captured = DesktopFrameMetrics.capture(stateKey)

    if (DesktopFrameMetrics.pendingRevocations.has(captured.consent)) {
      await DesktopFrameMetrics.purge(stateKey, captured)
    }

    try {
      return await metricsPhase(() => {
        DesktopFrameMetrics.repairOutgoingOwned(stateKey)

        return prepare ? DesktopFrameMetrics.prepareCohortOwned(stateKey) : DesktopFrameMetrics.capture(stateKey)
      })
    } catch {
      // Consent UI remains usable when telemetry's coordinator or storage is unavailable.
      return DesktopFrameMetrics.capture(stateKey)
    }
  }

  private settlement(): Settlement | null {
    if (this.published && !readJson<Journal>(this.slot)) {
      this.journal.days = {}
      this.journal.since = null
      this.published = false

      // GC and withdrawal are indistinguishable once a slot is gone. Only a fresh positive
      // consent read may create a replacement collector; elapsed time is not authorization.
      this.stopped = true
    }

    const value = readJson<ScopeState>(this.key) ?? { epoch: 'initial' }

    return !this.initializing && !this.stopped && !value.importing &&
      value.epoch === this.journal.epoch && value.consent === this.consent &&
      this.consent !== undefined && !DesktopFrameMetrics.pendingRevocations.has(this.consent) &&
      readKey(this.consent) === '1' ? value : null
  }

  isCurrent(): boolean {
    return !this.closing && !this.stopped && (this.initializing || this.settlement() !== null)
  }

  private isClosed(day: string): boolean {
    return readJson<Settlement>(`${this.key}:settled:${day}`)?.epoch === this.journal.epoch ||
      readJson<FrameReport>(`${this.key}:report:${day}`)?.epoch === this.journal.epoch
  }

  private day(day: string): FrameDay | null {
    const settlement = this.settlement()

    if (!settlement || this.isClosed(day) || day < utcDay(Date.now() - RETENTION_MS)) {
      return null
    }

    return (this.journal.days[day] ??= { frames: {}, visibleMs: 0 })
  }

  private save(now: number): void {
    const settlement = this.settlement()

    if (!settlement) {
      return
    }

    for (const day of Object.keys(this.journal.days)) {
      if (this.isClosed(day) || day < utcDay(now - RETENTION_MS)) {
        delete this.journal.days[day]
      }
    }

    this.journal.updatedAt = Math.max(this.journal.updatedAt, now)
    const empty = this.closing && this.journal.since === null && Object.keys(this.journal.days).length === 0

    this.journal.revision += 1
    writeJson(this.slot, empty ? null : this.journal)
    this.published = !empty

    if (!empty && readJson<Journal>(this.slot)?.revision !== this.journal.revision) {
      this.journal.days = {}
      this.journal.since = null
      this.stopped = true

      return
    }

    if (!this.settlement()) {
      writeJson(this.slot, null)

      return
    }

    if (this.prunedDay !== utcDay(now)) {
      this.prunedDay = utcDay(now)
      this.peers(now)
    }
  }

  /** Serialize synchronous aggregate read/modify/write with journal publication and retirement.
   * Promise continuations (including RPCs) run only after browser ownership is released. */
  private withOwnership<T>(change: () => T): Promise<T | undefined> | undefined {
    if (this.closing || this.stopped || (!this.initializing && !this.settlement())) {
      return undefined
    }

    return metricsPhase(() => this.settlement() ? change() : undefined).catch(() => {
      // In particular, a failed initial track must not leave a frame-only collector alive.
      this.stopped = true
      this.journal.since = null
      this.journal.days = {}

      return undefined
    })
  }

  /** Capture UI boundaries now, but decide span eligibility using the execution-time clock. */
  private mutate(change: (now: number) => void): void {
    void this.withOwnership(() => change(Date.now()))
  }

  private advance(at: number, visible: boolean, now: number): void {
    let since = this.journal.since

    this.journal.since = this.journal.maxSpanMs > 0 && visible
      ? Math.max(at, this.journal.updatedAt)
      : null

    if (since !== null && at > since && at - since <= this.journal.maxSpanMs && now - since <= this.journal.maxSpanMs) {
      while (since < at) {
        const day = utcDay(since)
        const until = Math.min(at, Date.parse(day) + DAY_MS)
        const entry = this.day(day)

        if (entry) {
          entry.visibleMs += until - since
        }

        since = until
      }
    }
  }

  checkpoint<T = void>(at = Date.now(), writeAggregate?: () => T): Promise<T | undefined> | undefined {
    const visible = document.visibilityState === 'visible'

    return this.withOwnership(() => {
      this.advance(at, visible, Date.now())
      this.save(at)

      return this.settlement() ? writeAggregate?.() : undefined
    })
  }

  track(maxSpanMs: number): void {
    const at = Date.now()
    const visible = document.visibilityState === 'visible'

    this.mutate(now => {
      this.advance(at, visible, now)
      this.journal.maxSpanMs = maxSpanMs
      this.journal.since = maxSpanMs > 0 && visible ? Math.max(at, this.journal.updatedAt) : null
      this.save(at)
    })
  }

  record(buckets: readonly LongFrameBucket[]): void {
    const at = Date.now()
    const visible = document.visibilityState === 'visible'

    this.mutate(now => {
      this.advance(at, visible, now)
      // Keep the observed UTC day. A sample that arrives after that day was settled (for example
      // after a backwards clock correction) is dropped, never reassigned to a different day.
      const day = this.day(utcDay(at))

      if (day) {
        for (const bucket of buckets) {
          day.frames[bucket] = (day.frames[bucket] ?? 0) + 1
        }
      }

      this.save(at)
    })
  }

  /** Keep the legacy source until the deterministic copy (or its frozen report) is durable. */
  adopt(day: string, source: FrameDay): boolean {
    if (this.isClosed(day) || day < utcDay(Date.now() - RETENTION_MS)) {
      return true
    }

    const key = `${this.key}:legacy-${day}`
    const existing = readJson<Journal>(key)
    const copied = existing?.epoch === this.journal.epoch ? existing.days[day] : undefined

    if (copied && copied.visibleMs === source.visibleMs &&
      BUCKETS.every(bucket => (copied.frames[bucket] ?? 0) === (source.frames[bucket] ?? 0))) {
      return true
    }

    this.mutate(now => {
      if (!this.isClosed(day)) {
        writeJson(key, {
          days: { [day]: source },
          epoch: this.journal.epoch,
          maxSpanMs: 0,
          since: null,
          updatedAt: now
        })

        if (!this.settlement()) {
          writeJson(key, null)
        }
      }
    })

    return false
  }

  stop(): void {
    const at = Date.now()

    this.mutate(now => {
      this.advance(at, false, now)
      this.journal.maxSpanMs = 0
      this.save(at)
      this.stopped = true
    })
    this.closing = true
  }

  /** Permits follow a logical profile; ownership epochs retire the old name. A pending copy is
   * durable before any destination payload, so the next attach can finish after a renderer crash. */
  static async migrate(from: string, to: string): Promise<void> {
    if (from === to) {
      return
    }

    const captured = DesktopFrameMetrics.capture(from)

    try {
      await metricsPhase(() => {
        const existing = readJson<ScopeState>(`${to}:frames`)

        if (existing?.importing?.from === from) {
          DesktopFrameMetrics.finishImportOwned(to)

          return
        }

        if (readJson<ScopeState>(`${from}:frames`)?.importing) {
          DesktopFrameMetrics.finishImportOwned(from)
        }

        let source = DesktopFrameMetrics.capture(from)

        const sourceState = readJson<ScopeState>(`${from}:frames`)

        if (!DesktopFrameMetrics.revokeConsentOwned(to, DesktopFrameMetrics.capture(to))) {
          DesktopFrameMetrics.revokeConsentOwned(from, source)

          return
        }

        if (DesktopFrameMetrics.pendingRevocations.has(source.consent)) {
          DesktopFrameMetrics.revokeConsentOwned(from, source)

          return
        }

        if (sourceState?.consent && !source.active) {
          DesktopFrameMetrics.retireScopeOwned(from)

          return
        }

        // Legacy data receives an immutable local cohort before it moves. Neither profile
        // names nor permits are included in the metric RPC.
        if (!source.active) {
          source = DesktopFrameMetrics.prepareCohortOwned(from)

          if (!source.active) {
            DesktopFrameMetrics.revokeConsentOwned(from, source)

            return
          }
        }

        const destination: ScopeState = {
          consent: source.consent,
          epoch: randomUuid(),
          importing: { epoch: source.epoch, from }
        }

        writeJson(`${to}:frames`, destination)

        if (readJson<ScopeState>(`${to}:frames`)?.epoch === destination.epoch) {
          DesktopFrameMetrics.finishImportOwned(to)
        } else {
          DesktopFrameMetrics.revokeConsentOwned(from, source)
        }
      })
    } catch {
      // Never fail an already-successful backend rename. This immutable permit removal is
      // the only uncoordinated fallback: it cannot replace a newer owner's token or revive data.
      DesktopFrameMetrics.revokePermitAfterFailure(captured)
    }
  }

  private static repairOutgoingOwned(stateKey: string): void {
    const source = readJson<ScopeState>(`${stateKey}:frames`)
    const prefix = stateKey.slice(0, stateKey.lastIndexOf(':') + 1)

    for (const key of frameKeys(prefix)) {
      if (!key.endsWith(':frames')) {
        continue
      }

      const candidate = readJson<ScopeState>(key)

      if (candidate?.importing?.from === stateKey && candidate.importing.epoch === source?.epoch &&
        candidate.consent === source?.consent) {
        DesktopFrameMetrics.finishImportOwned(key.slice(0, -':frames'.length))
      }
    }
  }

  /** Stage the complete cohort before the source name can be retired or reused. */
  private static stageImportOwned(
    snapshotKey: string,
    destination: ScopeState,
    source: ScopeState | null,
    captured: DesktopFrameConsent
  ): ImportSnapshot | null {
    const move = destination.importing!

    if (source?.epoch !== move.epoch || source.consent !== destination.consent) {
      DesktopFrameMetrics.revokeConsentOwned(move.from, captured)

      return null
    }

    const inherited = source.importing
      ? readJson<ImportSnapshot>(`${move.from}:frames:import:${source.epoch}`)
      : null

    if (source.importing && (!inherited || inherited.epoch !== source.epoch)) {
      DesktopFrameMetrics.revokeConsentOwned(move.from, captured)

      return null
    }

    const entries: ImportSnapshot['entries'] = inherited?.entries ?? frameKeys(move.from).flatMap(sourceKey => {
      if ((sourceKey !== move.from && !sourceKey.startsWith(`${move.from}:frames:`)) ||
        sourceKey.startsWith(`${move.from}:frames:consent:`) ||
        sourceKey.startsWith(`${move.from}:frames:import:`)) {
        return []
      }

      const value = readJson<Record<string, unknown>>(sourceKey)

      return value && (!('epoch' in value) || value.epoch === move.epoch)
        ? [[sourceKey.slice(move.from.length), value] as [string, Record<string, unknown>]]
        : []
    })

    const snapshot: ImportSnapshot = {
      entries: entries.map(([suffix, value]): [string, Record<string, unknown>] => [
        suffix, 'epoch' in value ? { ...value, epoch: destination.epoch } : value
      ]),
      epoch: destination.epoch
    }

    const raw = JSON.stringify(snapshot)

    writeKey(snapshotKey, raw)

    if (readKey(snapshotKey) !== raw) {
      // Optional measurements must not remain attached to a reusable source name when no
      // durable, cohort-isolated retry copy fits. Never touch sessions or other user content.
      DesktopFrameMetrics.revokeConsentOwned(move.from, captured)

      return null
    }

    return snapshot
  }

  private static finishImportOwned(to: string, visiting = new Set<string>()): boolean {
    const key = `${to}:frames`
    const destination = readJson<ScopeState>(key)
    const move = destination?.importing

    if (!destination || !move) {
      return true
    }

    if (visiting.has(to)) {
      return false
    }

    visiting.add(to)

    const captured: DesktopFrameConsent = {
      active: true,
      consent: destination.consent!,
      epoch: move.epoch
    }

    if (!destination.consent || readKey(destination.consent) !== '1') {
      DesktopFrameMetrics.retireScopeOwned(to)

      return false
    }

    const snapshotKey = `${key}:import:${destination.epoch}`
    let snapshot = readJson<ImportSnapshot>(snapshotKey)
    let source = readJson<ScopeState>(`${move.from}:frames`)

    if (source?.importing && !snapshot) {
      DesktopFrameMetrics.finishImportOwned(move.from, visiting)
      source = readJson<ScopeState>(`${move.from}:frames`)
    }

    if (!snapshot) {
      snapshot = DesktopFrameMetrics.stageImportOwned(snapshotKey, destination, source, captured)

      if (!snapshot) {
        return false
      }
    }

    if (snapshot.epoch !== destination.epoch) {
      return false
    }

    // Retire the name before copying individual destination entries. A failed copy can now
    // retry only this immutable quarantine, never a newly recreated source profile's data.
    if (source?.epoch === move.epoch && source.consent === destination.consent &&
      !DesktopFrameMetrics.retireScopeOwned(move.from)) {
      DesktopFrameMetrics.revokeConsentOwned(move.from, captured)

      return false
    }

    for (const [suffix, value] of snapshot.entries) {
      const destinationKey = `${to}${suffix}`
      const raw = JSON.stringify(value)

      writeKey(destinationKey, raw)

      if (readKey(destinationKey) !== raw) {
        return false
      }
    }

    writeJson(key, { consent: destination.consent, epoch: destination.epoch })
    const complete = readJson<ScopeState>(key)
    const ready = complete?.epoch === destination.epoch && !complete.importing

    if (ready) {
      writeKey(snapshotKey, null)
    }

    return ready
  }

  private static clearPayloadOwned(stateKey: string): void {
    for (const key of frameKeys(`${stateKey}:frames:`)) {
      if (!key.startsWith(`${stateKey}:frames:consent:`)) {
        writeKey(key, null)
      }
    }

    writeJson(stateKey, null)
  }

  private static retireScopeOwned(stateKey: string): boolean {
    const key = `${stateKey}:frames`
    const epoch = randomUuid()

    writeJson(key, { epoch })
    DesktopFrameMetrics.clearPayloadOwned(stateKey)

    if (readJson<ScopeState>(key)?.epoch !== epoch) {
      // Removal can reclaim space when quota prevented the first marker write.
      writeJson(key, { epoch })
    }

    return readJson<ScopeState>(key)?.epoch === epoch
  }

  private static revokeConsentOwned(stateKey: string, captured: DesktopFrameConsent): boolean {
    writeKey(captured.consent, null)

    if (readKey(captured.consent) !== null) {
      return false
    }

    let retired = true
    const prefix = stateKey.slice(0, stateKey.lastIndexOf(':') + 1)

    // Only the captured permit is withdrawn. A recreated old name has a different permit.
    for (const key of frameKeys(prefix)) {
      if (key.endsWith(':frames') && readJson<ScopeState>(key)?.consent === captured.consent) {
        retired = DesktopFrameMetrics.retireScopeOwned(key.slice(0, -':frames'.length)) && retired
      }
    }

    const original = DesktopFrameMetrics.capture(stateKey)

    if (original.epoch === captured.epoch && original.consent === captured.consent) {
      retired = DesktopFrameMetrics.retireScopeOwned(stateKey) && retired
    }

    return retired
  }

  private static revokePermitAfterFailure(captured: DesktopFrameConsent): void {
    DesktopFrameMetrics.pendingRevocations.add(captured.consent)

    try {
      const storage = window.localStorage

      storage.removeItem(captured.consent)

      if (storage.getItem(captured.consent) === null) {
        DesktopFrameMetrics.pendingRevocations.delete(captured.consent)
      }
    } catch {
      // Keep this renderer disabled when even an allocation-free removal is unavailable.
    }
  }

  static async purge(
    stateKey: string,
    captured = DesktopFrameMetrics.capture(stateKey)
  ): Promise<DesktopFrameRevocation> {
    DesktopFrameMetrics.pendingRevocations.add(captured.consent)

    try {
      return await metricsPhase(() => {
        const before = DesktopFrameMetrics.capture(stateKey)
        const complete = DesktopFrameMetrics.revokeConsentOwned(stateKey, captured)
        const after = DesktopFrameMetrics.capture(stateKey)

        if (complete) {
          DesktopFrameMetrics.pendingRevocations.delete(captured.consent)
        }

        return { after, before, complete }
      })
    } catch {
      // No namespace resolution or replacement writes outside ownership. This immutable
      // permit removal cannot revoke a new cohort that appeared while coordination failed.
      DesktopFrameMetrics.revokePermitAfterFailure(captured)

      return { after: null, before: null, complete: false }
    }
  }

  private peers(now: number): Journal[] {
    const peers: Journal[] = []

    for (const key of frameKeys(`${this.key}:`)) {
      if (key.startsWith(`${this.key}:consent:`)) {
        continue
      }

      if (key.startsWith(`${this.key}:import:`)) {
        writeKey(key, null)

        continue
      }

      if (key.startsWith(`${this.key}:settled:`) || key.startsWith(`${this.key}:report:`)) {
        if (key.slice(-10) < utcDay(now - RETENTION_MS)) {
          writeJson(key, null)
        }

        continue
      }

      const peer = readJson<Journal>(key)

      if (!peer || peer.updatedAt < now - RETENTION_MS) {
        writeJson(key, null)
      } else if (peer.epoch === this.journal.epoch) {
        peers.push(peer)
      }
    }

    return peers
  }

  private freeze(): FrameReport | null {
    DesktopFrameMetrics.repairOutgoingOwned(this.stateKey)

    if (!this.settlement()) {
      return null
    }

    const now = Date.now()

    if (!this.published) {
      this.save(now)

      if (!this.settlement() || !this.published) {
        return null
      }
    }

    const peers = this.peers(now)

    const days = [...new Set([
      ...peers.flatMap(peer => Object.keys(peer.days)),
      ...frameKeys(`${this.key}:report:`).map(key => key.slice(-10))
    ])].sort()

    for (const day of days) {
      if (day >= utcDay(now) || day < utcDay(now - RETENTION_MS) ||
        readJson<Settlement>(`${this.key}:settled:${day}`)?.epoch === this.journal.epoch) {
        continue
      }

      const key = `${this.key}:report:${day}`
      const frozen = readJson<FrameReport>(key)

      if (frozen?.epoch === this.journal.epoch) {
        return frozen
      }

      const end = Date.parse(day) + DAY_MS

      if (peers.some(peer => peer.since !== null && peer.since < end && now - peer.since <= peer.maxSpanMs)) {
        continue
      }

      const report: FrameReport = { day, epoch: this.journal.epoch, frames: {}, visibleMs: 0 }

      for (const peer of peers) {
        const entry = peer.days[day]

        if (entry) {
          report.visibleMs += entry.visibleMs

          for (const bucket of BUCKETS) {
            report.frames[bucket] = (report.frames[bucket] ?? 0) + (entry.frames[bucket] ?? 0)
          }
        }
      }

      const raw = JSON.stringify(report)

      writeKey(key, raw)

      if (!this.settlement()) {
        writeJson(key, null)

        return null
      }

      // Quota/private-mode failures must not create a report whose only copy is in memory.
      return readKey(key) === raw ? report : null
    }

    return null
  }

  async flush(requester: SharedMetricsRequester, current: () => boolean): Promise<void> {
    await this.ready

    if (this.flushing) {
      this.flushAgain = true

      return
    }

    if (!this.settlement() || !current()) {
      return
    }

    this.flushing = true

    try {
      while (this.settlement() && current()) {
        const report = await metricsPhase(() => current() ? this.freeze() : null)

        if (!report || !this.settlement() || !current()) {
          return
        }

        if (report.visibleMs > 0 || Object.values(report.frames).some(count => (count ?? 0) > 0)) {
          try {
            const result = await requester<{ recorded?: boolean }>('shared_metrics.desktop_long_frames', {
              day: report.day,
              frames: BUCKETS.filter(bucket => (report.frames[bucket] ?? 0) > 0)
                .map(bucket => ({ bucket, count: report.frames[bucket]! })),
              visible_ms: report.visibleMs
            })

            if (result?.recorded !== true) {
              return
            }
          } catch (error) {
            if (!(error instanceof JsonRpcGatewayError && error.code === -32601)) {
              return
            }
          }
        }

        const acknowledged = await metricsPhase(() => {
          if (!this.settlement() || !current()) {
            return false
          }

          const key = `${this.key}:settled:${report.day}`

          writeJson(key, { epoch: report.epoch })

          if (readJson<Settlement>(key)?.epoch !== report.epoch) {
            return false
          }

          writeJson(`${this.key}:report:${report.day}`, null)

          return true
        })

        if (!acknowledged) {
          return
        }
      }
    } catch {
      // Unavailable coordination/storage leaves durable journals and reports for the next attach.
    } finally {
      this.flushing = false

      if (this.flushAgain) {
        this.flushAgain = false
        await this.flush(requester, current)
      }
    }
  }
}
