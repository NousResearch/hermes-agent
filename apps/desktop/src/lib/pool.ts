/**
 * `Promise.all(items.map(fn))` with a concurrency cap: at most `limit` calls run
 * at once, results stay in input order. Keeps a many-repo probe from spawning a
 * `git` process per repo all at once.
 */
export async function mapPool<T, R>(items: readonly T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length)
  let next = 0

  const worker = async () => {
    while (next < items.length) {
      const i = next++
      out[i] = await fn(items[i])
    }
  }

  await Promise.all(Array.from({ length: Math.min(Math.max(1, limit), items.length) }, worker))

  return out
}
