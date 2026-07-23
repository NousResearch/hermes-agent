export function remainingLinkTitleMs(deadline: number, now: () => number = Date.now): number {
  return Math.max(0, deadline - now())
}

export function withLinkTitleTimeout<T>(operation: PromiseLike<T>, timeoutMs: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      reject(new Error('Link title operation timed out'))

      return
    }

    let settled = false

    const finish = (callback: () => void) => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      callback()
    }

    const timer = setTimeout(() => {
      finish(() => reject(new Error('Link title operation timed out')))
    }, timeoutMs)

    Promise.resolve(operation).then(
      value => finish(() => resolve(value)),
      error => finish(() => reject(error))
    )
  })
}
