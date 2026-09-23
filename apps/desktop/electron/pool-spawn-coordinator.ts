import type { WaitableChild } from './backend-child'

export type ReleaseLocalBackendSlot = () => void

export type LocalBackendSpawnPriority = 'foreground' | 'background'

/** Single normalization point for wire priority values: anything but an
 *  explicit 'foreground' dials background, so hydration can never consume the
 *  pool's reserved foreground slot by accident. */
export function spawnPriorityFrom(value: unknown): LocalBackendSpawnPriority {
  return value === 'foreground' ? 'foreground' : 'background'
}

export type LocalBackendSpawnRequest = {
  acquired: Promise<ReleaseLocalBackendSlot>
  cancel: () => boolean
  promote: (priority: LocalBackendSpawnPriority) => boolean
  /** False when the slot was granted without waiting behind the queue. */
  queued: boolean
}

type Waiter = {
  key: string
  priority: LocalBackendSpawnPriority
  resolve: (release: ReleaseLocalBackendSlot) => void
  reject: (error: Error) => void
  timer: ReturnType<typeof setTimeout> | null
}

const SLOT_WAIT_TIMEOUT_MESSAGE = (key: string) =>
  `Local backend start for "${key}" timed out while waiting for a free slot.`

/**
 * Slot-wait timeout. Background hydrations set `silent` so call sites can fail
 * quiet instead of toasting a user-visible backend-start failure.
 */
export class LocalBackendSlotWaitTimeoutError extends Error {
  readonly priority: LocalBackendSpawnPriority
  readonly silent: boolean

  constructor(key: string, priority: LocalBackendSpawnPriority) {
    const suffix = priority === 'background' ? ' (background)' : ''
    super(`${SLOT_WAIT_TIMEOUT_MESSAGE(key)}${suffix}`)
    this.name = 'LocalBackendSlotWaitTimeoutError'
    this.priority = priority
    this.silent = priority === 'background'
  }
}

export function isBackgroundSlotWaitTimeout(error: unknown): boolean {
  return error instanceof LocalBackendSlotWaitTimeoutError && error.silent
}

/** A retry deferment is expected background control flow, not a start failure. */
export class BackgroundSlotRetryDeferredError extends Error {
  constructor(key: string) {
    super(`Local backend start for "${key}" is backing off after slot saturation.`)
    this.name = 'BackgroundSlotRetryDeferredError'
  }
}

export function isBackgroundSlotRetryDeferred(error: unknown): boolean {
  return error instanceof BackgroundSlotRetryDeferredError
}

/**
 * Per-profile cooldown for background hydration after a pool-slot timeout.
 *
 * A full local pool is commonly structural (more profiles than the configured
 * cap), so retrying every roster refresh creates a permanent timeout and log
 * storm. Foreground opens bypass this guard; a successful slot acquisition
 * clears it. Keeping this state outside the coordinator preserves its role as
 * a fair slot allocator rather than teaching it about hydration policy.
 */
export class BackgroundSlotRetryBackoff {
  #failures = new Map<string, { nextRetryAt: number; attempts: number }>()
  readonly #baseDelayMs: number
  readonly #maxDelayMs: number

  constructor({ baseDelayMs = 60_000, maxDelayMs = 15 * 60_000 } = {}) {
    if (!Number.isFinite(baseDelayMs) || baseDelayMs < 1 || !Number.isFinite(maxDelayMs) || maxDelayMs < baseDelayMs) {
      throw new RangeError('Background slot retry delays must be positive and ordered.')
    }

    this.#baseDelayMs = baseDelayMs
    this.#maxDelayMs = maxDelayMs
  }

  canAttempt(key: string, now = Date.now()): boolean {
    return (this.#failures.get(key)?.nextRetryAt ?? 0) <= now
  }

  recordFailure(key: string, now = Date.now()): number {
    const attempts = (this.#failures.get(key)?.attempts ?? 0) + 1
    const delay = Math.min(this.#baseDelayMs * 2 ** (attempts - 1), this.#maxDelayMs)
    this.#failures.set(key, { attempts, nextRetryAt: now + delay })

    return delay
  }

  clear(key: string): void {
    this.#failures.delete(key)
  }
}

/** Register at spawn, before claiming ownership or awaiting readiness. */
export function registerLocalBackendExitFinalizer<Entry extends { process: WaitableChild | null }>(
  pool: Map<string, Entry>,
  key: string,
  entry: Entry,
  release: ReleaseLocalBackendSlot
): void {
  const child = entry.process

  const finalize = () => {
    child?.removeListener('exit', finalize)
    child?.removeListener('error', spawnFailed)

    if (pool.get(key) === entry) {
      pool.delete(key)
    }

    release()
  }

  const spawnFailed = () => {
    // Failed spawn has no PID and emits error, but never exit. A signal error
    // against an existing child is not evidence of exit and must keep its slot.
    if (!child?.pid) {
      finalize()
    }
  }

  child?.once('exit', finalize)
  child?.once('error', spawnFailed)
}

export async function releaseLocalBackendSlotAfterExit(
  release: ReleaseLocalBackendSlot,
  waitForExit: () => Promise<void>
): Promise<void> {
  await waitForExit()
  release()
}

/**
 * Bounds the number of local profile backends that are starting or running.
 *
 * A lease is acquired immediately before local start work and is held until
 * the child exits or the start fails. Remote descriptors never call request().
 *
 * When the cap is at least 2, one slot is reserved for foreground (user-open)
 * requests so background roster hydration cannot occupy the whole pool.
 * Untagged acquire() is foreground, so existing cap tests still fill `limit`.
 */
export class LocalBackendSpawnCoordinator {
  #limit: number
  #activeForeground = 0
  #activeBackground = 0
  #queue: Waiter[] = []
  #listeners = new Set<() => void>()

  get foregroundWaiters(): ReadonlySet<string> {
    return new Set(this.#queue.filter(waiter => waiter.priority === 'foreground').map(waiter => waiter.key))
  }

  onChange(listener: () => void): () => void {
    this.#listeners.add(listener)

    return () => {
      this.#listeners.delete(listener)
    }
  }

  #changed(): void {
    for (const listener of this.#listeners) {
      listener()
    }
  }

  constructor(limit: number) {
    if (!Number.isInteger(limit) || limit < 1) {
      throw new RangeError('Local backend spawn limit must be a positive integer.')
    }

    this.#limit = limit
  }

  get activeCount(): number {
    return this.#activeForeground + this.#activeBackground
  }

  get limit(): number {
    return this.#limit
  }

  /**
   * Adopt a new cap at runtime (the pool size is a live device preference).
   * Raising it drains waiters into the newly freed slots immediately; lowering
   * it never revokes a granted slot — the running backends simply stay over
   * the cap until they exit, and LRU eviction (main.ts) converges the pool.
   */
  setLimit(limit: number): void {
    if (!Number.isInteger(limit) || limit < 1) {
      throw new RangeError('Local backend spawn limit must be a positive integer.')
    }

    this.#limit = limit
    this.#drain()
  }

  get queuedCount(): number {
    return this.#queue.length
  }

  request(
    key: string,
    options: { timeoutMs?: number; priority?: LocalBackendSpawnPriority } = {}
  ): LocalBackendSpawnRequest {
    if (options.timeoutMs !== undefined && (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 1)) {
      throw new RangeError('Local backend spawn timeout must be a positive number.')
    }

    const priority: LocalBackendSpawnPriority = options.priority === 'background' ? 'background' : 'foreground'

    if (this.#queue.length === 0 && this.#canGrant(priority)) {
      return {
        acquired: Promise.resolve(this.#grant(priority)),
        cancel: () => false,
        promote: () => false,
        queued: false
      }
    }

    let waiter!: Waiter

    const acquired = new Promise<ReleaseLocalBackendSlot>((resolve, reject) => {
      waiter = { key, priority, resolve, reject, timer: null }
      this.#queue.push(waiter)

      if (options.timeoutMs !== undefined) {
        waiter.timer = setTimeout(() => {
          this.#rejectWaiter(waiter, this.#timeoutError(waiter))
        }, options.timeoutMs)
        waiter.timer.unref?.()
      }
    })

    this.#drain()

    return {
      acquired,
      cancel: () =>
        this.#rejectWaiter(waiter, new Error(`Local backend start for "${key}" was cancelled while queued.`)),
      promote: (nextPriority: LocalBackendSpawnPriority) => this.#promoteWaiter(waiter, nextPriority),
      queued: this.#queue.includes(waiter)
    }
  }

  acquire(key: string): Promise<ReleaseLocalBackendSlot> {
    return this.request(key).acquired
  }

  #timeoutError(waiter: Waiter): Error {
    if (waiter.priority === 'background') {
      return new LocalBackendSlotWaitTimeoutError(waiter.key, 'background')
    }

    return new Error(SLOT_WAIT_TIMEOUT_MESSAGE(waiter.key))
  }

  #backgroundLimit(): number {
    return this.#limit >= 2 ? this.#limit - 1 : this.#limit
  }

  #canGrant(priority: LocalBackendSpawnPriority): boolean {
    if (this.activeCount >= this.#limit) {
      return false
    }

    if (priority === 'background' && this.#activeBackground >= this.#backgroundLimit()) {
      return false
    }

    return true
  }

  #promoteWaiter(waiter: Waiter, priority: LocalBackendSpawnPriority): boolean {
    if (!this.#queue.includes(waiter)) {
      return false
    }

    if (waiter.priority === priority) {
      return false
    }

    waiter.priority = priority
    this.#drain()

    return true
  }

  #rejectWaiter(waiter: Waiter, error: Error): boolean {
    const index = this.#queue.indexOf(waiter)

    if (index === -1) {
      return false
    }

    this.#queue.splice(index, 1)
    this.#clearTimer(waiter)
    waiter.reject(error)
    this.#changed()

    return true
  }

  #clearTimer(waiter: Waiter): void {
    if (waiter.timer) {
      clearTimeout(waiter.timer)
      waiter.timer = null
    }
  }

  #grant(priority: LocalBackendSpawnPriority): ReleaseLocalBackendSlot {
    if (priority === 'background') {
      this.#activeBackground += 1
    } else {
      this.#activeForeground += 1
    }

    let released = false

    return () => {
      if (released) {
        return
      }

      released = true

      if (priority === 'background') {
        this.#activeBackground -= 1
      } else {
        this.#activeForeground -= 1
      }

      this.#drain()
    }
  }

  #takeWaiter(priority: LocalBackendSpawnPriority): Waiter | undefined {
    const index = this.#queue.findIndex(waiter => waiter.priority === priority)

    if (index === -1) {
      return undefined
    }

    return this.#queue.splice(index, 1)[0]
  }

  /** Hand free slots to queued waiters. Foreground waiters always go first. */
  #drain(): void {
    while (this.#canGrant('foreground')) {
      const next = this.#takeWaiter('foreground')

      if (!next) {
        break
      }

      this.#clearTimer(next)
      next.resolve(this.#grant('foreground'))
    }

    while (this.#canGrant('background')) {
      const next = this.#takeWaiter('background')

      if (!next) {
        break
      }

      this.#clearTimer(next)
      next.resolve(this.#grant('background'))
    }

    this.#changed()
  }
}
