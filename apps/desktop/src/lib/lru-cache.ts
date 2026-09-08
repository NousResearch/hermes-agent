/**
 * A `Map` with a ceiling. The renderer stays open for days, so a module-level
 * cache keyed by anything user-driven (a URL, a connection id, an equation)
 * grows for the life of the window unless something evicts.
 *
 * Only for values that can be REGENERATED: eviction costs a recompute or a
 * refetch, never correctness. State whose loss changes behaviour — an unsent
 * draft, an unread watermark, a freshness fence — is a record, not a cache,
 * and belongs in a plain `Map`.
 */
export class LruCache<K, V> {
  private readonly entries = new Map<K, V>()
  private readonly max: number

  constructor(max: number) {
    this.max = max
  }

  get size(): number {
    return this.entries.size
  }

  /** Read and mark most-recently-used. */
  get(key: K): undefined | V {
    const value = this.entries.get(key)

    if (value === undefined) {
      return undefined
    }

    // Map iterates in insertion order, so re-inserting moves the entry to the
    // tail and leaves the least recently used one at the head.
    this.entries.delete(key)
    this.entries.set(key, value)

    return value
  }

  /** Membership WITHOUT touching recency — for the `has` / `set` / `get`
   *  populate-on-miss shape, where the `get` does the touching. */
  has(key: K): boolean {
    return this.entries.has(key)
  }

  /** Write, evicting the least recently used entry once the cache is full. */
  set(key: K, value: V): void {
    if (this.entries.has(key)) {
      this.entries.delete(key)
    } else if (this.entries.size >= this.max) {
      const oldest = this.entries.keys().next().value

      if (oldest !== undefined) {
        this.entries.delete(oldest)
      }
    }

    this.entries.set(key, value)
  }

  delete(key: K): boolean {
    return this.entries.delete(key)
  }

  keys(): IterableIterator<K> {
    return this.entries.keys()
  }
}
