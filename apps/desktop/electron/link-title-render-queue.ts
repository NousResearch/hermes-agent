export interface LinkTitleRenderQueue {
  close(): void
  enqueue(url: string): Promise<string>
}

interface QueueItem {
  deadline: number
  resolve(value: string): void
  settled: boolean
  timer: ReturnType<typeof setTimeout>
  url: string
}

export function createLinkTitleRenderQueue(options: {
  concurrency: number
  now?: () => number
  run(url: string, deadline: number): Promise<string>
  timeoutMs: number
}): LinkTitleRenderQueue {
  const concurrency = Math.max(1, Math.floor(options.concurrency))
  const now = options.now ?? Date.now
  const pending: QueueItem[] = []
  let active = 0
  let closed = false

  const settle = (item: QueueItem, value: string) => {
    if (item.settled) {
      return
    }

    item.settled = true
    clearTimeout(item.timer)
    item.resolve(value)
  }

  const dequeue = () => {
    while (!closed && active < concurrency && pending.length) {
      const item = pending.shift()

      if (!item || item.settled) {
        continue
      }

      if (now() >= item.deadline) {
        settle(item, '')

        continue
      }

      clearTimeout(item.timer)
      active += 1
      void options
        .run(item.url, item.deadline)
        .catch(() => '')
        .then(value => settle(item, value))
        .finally(() => {
          active -= 1
          dequeue()
        })
    }
  }

  return {
    close() {
      if (closed) {
        return
      }

      closed = true

      for (const item of pending.splice(0)) {
        settle(item, '')
      }
    },
    enqueue(url) {
      if (closed) {
        return Promise.resolve('')
      }

      return new Promise(resolve => {
        const deadline = now() + options.timeoutMs

        const item: QueueItem = {
          deadline,
          resolve,
          settled: false,
          timer: setTimeout(() => settle(item, ''), Math.max(0, deadline - now())),
          url
        }

        pending.push(item)
        dequeue()
      })
    }
  }
}
