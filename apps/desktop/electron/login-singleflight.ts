/** Concurrent requests for one auth scope share one login flow. */
export function createLoginSingleFlight<T>() {
  const pending = new Map<string, Promise<T>>()

  return (scope: string, start: () => Promise<T>): Promise<T> => {
    const existing = pending.get(scope)
    if (existing) return existing

    const flow = Promise.resolve().then(start)
    const tracked = flow.then(
      result => {
        if (pending.get(scope) === tracked) pending.delete(scope)
        return result
      },
      error => {
        if (pending.get(scope) === tracked) pending.delete(scope)
        throw error
      }
    )
    pending.set(scope, tracked)
    return tracked
  }
}
