import type { WriteOptions } from '@/lib/storage'

/** Persist a store: the value it loads with (its default, or a heal of what
 *  was stored) as provisional, every later change as the user's. */
export function persistStore<T>(
  store: { subscribe(listener: (value: T) => void): () => void },
  persist: (value: T, options: WriteOptions) => void
): () => void {
  let loading = true
  const stop = store.subscribe(value => persist(value, { provisional: loading }))
  loading = false

  return stop
}
