export type ReleaseLocalBackendSlot = () => void

export type LocalBackendSpawnRequest = {
  acquired: Promise<ReleaseLocalBackendSlot>
  cancel: () => boolean
}

type Waiter = {
  key: string
  resolve: (release: ReleaseLocalBackendSlot) => void
  reject: (error: Error) => void
  timer: ReturnType<typeof setTimeout> | null
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
 */
export class LocalBackendSpawnCoordinator {
  #limit: number
  #active = 0
  #queue: Waiter[] = []

  constructor(limit: number) {
    if (!Number.isInteger(limit) || limit < 1) {
      throw new RangeError('Local backend spawn limit must be a positive integer.')
    }

    this.#limit = limit
  }

  get activeCount(): number {
    return this.#active
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

  request(key: string, options: { timeoutMs?: number } = {}): LocalBackendSpawnRequest {
    if (options.timeoutMs !== undefined && (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 1)) {
      throw new RangeError('Local backend spawn timeout must be a positive number.')
    }

    if (this.#active < this.#limit) {
      return {
        acquired: Promise.resolve(this.#grant()),
        cancel: () => false
      }
    }

    let waiter!: Waiter

    const acquired = new Promise<ReleaseLocalBackendSlot>((resolve, reject) => {
      waiter = { key, resolve, reject, timer: null }
      this.#queue.push(waiter)

      if (options.timeoutMs !== undefined) {
        waiter.timer = setTimeout(() => {
          this.#rejectWaiter(
            waiter,
            new Error(`Local backend start for "${key}" timed out while waiting for a free slot.`)
          )
        }, options.timeoutMs)
        waiter.timer.unref?.()
      }
    })

    return {
      acquired,
      cancel: () =>
        this.#rejectWaiter(waiter, new Error(`Local backend start for "${key}" was cancelled while queued.`))
    }
  }

  acquire(key: string): Promise<ReleaseLocalBackendSlot> {
    return this.request(key).acquired
  }

  #rejectWaiter(waiter: Waiter, error: Error): boolean {
    const index = this.#queue.indexOf(waiter)

    if (index === -1) {
      return false
    }

    this.#queue.splice(index, 1)
    this.#clearTimer(waiter)
    waiter.reject(error)

    return true
  }

  #clearTimer(waiter: Waiter): void {
    if (waiter.timer) {
      clearTimeout(waiter.timer)
      waiter.timer = null
    }
  }

  #grant(): ReleaseLocalBackendSlot {
    this.#active += 1
    let released = false

    return () => {
      if (released) {
        return
      }

      released = true
      this.#active -= 1
      this.#drain()
    }
  }

  /** Hand free slots to queued waiters while under the (possibly lowered) cap. */
  #drain(): void {
    while (this.#active < this.#limit && this.#queue.length > 0) {
      const next = this.#queue.shift()!
      this.#clearTimer(next)
      next.resolve(this.#grant())
    }
  }
}
