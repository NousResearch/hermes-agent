import {
  type DesignPageEvent,
  parsePreviewDesignBridgeEvent,
  PREVIEW_DESIGN_COMMAND_CHANNEL,
  PREVIEW_DESIGN_EVENT_CHANNEL,
  type PreviewDesignCommand
} from '@hermes/shared/preview-design'

interface PreviewDesignWebview extends EventTarget {
  send?: (channel: string, payload: PreviewDesignCommand) => void
}

export interface PreviewDesignGuest {
  addEventListener(type: 'ipc-message', listener: EventListener): void
  removeEventListener(type: 'ipc-message', listener: EventListener): void
  send(command: PreviewDesignCommand): void
}

export interface PreviewDesignSession {
  selectLayer(index: number): boolean
  stop(): Promise<void>
  wait(): Promise<DesignPageEvent>
}

export function createPreviewDesignGuest(webview: PreviewDesignWebview | null): PreviewDesignGuest | null {
  if (!webview || typeof webview.send !== 'function') {
    return null
  }

  return {
    addEventListener: (type, listener) => webview.addEventListener(type, listener),
    removeEventListener: (type, listener) => webview.removeEventListener(type, listener),
    send: command => webview.send!(PREVIEW_DESIGN_COMMAND_CHANNEL, command)
  }
}

let nextSession = 0

function deferred<T>() {
  let reject!: (reason?: unknown) => void
  let resolve!: (value: T | PromiseLike<T>) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, reject, resolve }
}

function timeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(message)), ms)

    promise.then(
      value => {
        window.clearTimeout(timer)
        resolve(value)
      },
      error => {
        window.clearTimeout(timer)
        reject(error)
      }
    )
  })
}

/** Start the controller in the guest preload's isolated world and accept only
 * bounded, schema-validated DOM context back from it. */
export async function installDesignOverlay(
  guest: PreviewDesignGuest,
  signal: AbortSignal
): Promise<PreviewDesignSession> {
  signal.throwIfAborted()
  const session = ++nextSession
  const ready = deferred<void>()
  const stopped = deferred<void>()
  const queue: DesignPageEvent[] = []
  let waiting: ((event: DesignPageEvent) => void) | null = null
  let stopPromise: Promise<void> | null = null
  let closed = false

  const emit = (event: DesignPageEvent) => {
    if (waiting) {
      const resolve = waiting
      waiting = null
      resolve(event)
    } else {
      queue.push(event)
    }
  }

  const cleanup = () => {
    if (closed) {
      return
    }

    closed = true
    guest.removeEventListener('ipc-message', onMessage)
    signal.removeEventListener('abort', onAbort)
  }

  const stop = () => {
    if (stopPromise) {
      return stopPromise
    }

    stopPromise = (async () => {
      try {
        guest.send({ session, type: 'stop' })
        await timeout(stopped.promise, 1_500, 'Design Mode cleanup timed out.')
      } catch {
        // A navigation or guest teardown can remove the preload before it can
        // acknowledge. Host cleanup still retires the session deterministically.
      } finally {
        emit({ type: 'end' })
        cleanup()
      }
    })()

    return stopPromise
  }

  function onAbort() {
    void stop()
  }

  function onMessage(event: Event) {
    const detail = event as Event & { args?: unknown[]; channel?: string }

    if (detail.channel !== PREVIEW_DESIGN_EVENT_CHANNEL) {
      return
    }

    const message = parsePreviewDesignBridgeEvent(detail.args?.[0])

    if (!message) {
      ready.reject(new Error('Design Mode returned invalid DOM context.'))
      void stop()

      return
    }

    if (message.session !== session) {
      return
    }

    if (message.type === 'ready') {
      ready.resolve()
    } else if (message.type === 'select') {
      emit({ selection: message.selection, type: 'select' })
    } else if (message.type === 'end') {
      emit({ type: 'end' })
    } else if (message.type === 'stopped') {
      stopped.resolve()
    } else {
      ready.reject(new Error(message.message))
      emit({ type: 'end' })
    }
  }

  const designSession: PreviewDesignSession = {
    selectLayer: index => {
      if (closed || !Number.isSafeInteger(index) || index < 0 || index >= 16) {
        return false
      }

      guest.send({ index, session, type: 'select-layer' })

      return true
    },
    stop,
    wait: () =>
      queue.length
        ? Promise.resolve(queue.shift()!)
        : new Promise(resolve => {
            waiting = resolve
          })
  }

  guest.addEventListener('ipc-message', onMessage)
  signal.addEventListener('abort', onAbort, { once: true })

  try {
    guest.send({ session, type: 'start' })
    await timeout(ready.promise, 4_000, 'Design Mode did not become ready.')
    signal.throwIfAborted()

    return designSession
  } catch (error) {
    await stop()
    throw error
  }
}