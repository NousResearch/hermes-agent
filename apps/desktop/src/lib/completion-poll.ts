interface CompletionPollOptions<T> {
  delayMs: number
  poll: (signal: AbortSignal) => Promise<T>
  publish: (value: T) => void
}

/** Serialize a producer across effect lifecycles without publishing stale work. */
export function createSerialTask<Args, Result>(
  task: (args: Args) => Promise<Result>
): (args: Args, signal: AbortSignal) => Promise<Result> {
  let tail = Promise.resolve()

  return (args, signal) => {
    const result = tail.then(() => {
      if (signal.aborted) {
        throw new DOMException('Task stopped', 'AbortError')
      }

      return task(args)
    })

    tail = result.then(
      () => undefined,
      () => undefined
    )

    return result
  }
}

/** Poll immediately, then wait one full cadence after each request settles. */
export function startCompletionPoll<T>({ delayMs, poll, publish }: CompletionPollOptions<T>): () => void {
  const controller = new AbortController()
  let timer: number | undefined

  const run = async () => {
    try {
      const value = await poll(controller.signal)

      if (!controller.signal.aborted) {
        publish(value)
      }
    } catch {
      // A transient producer failure keeps the previous value visible.
    }

    if (!controller.signal.aborted) {
      timer = window.setTimeout(() => void run(), delayMs)
    }
  }

  void run()

  return () => {
    controller.abort()

    if (timer !== undefined) {
      window.clearTimeout(timer)
    }
  }
}
