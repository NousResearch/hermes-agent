// A cancelled protocol body must abort Electron's fetch, not just its reader:
// otherwise an unread response can retain a gateway HTTP connection indefinitely.
export async function fetchRemoteMedia(
  fetcher: (headers: Headers, signal: AbortSignal) => Promise<Response>,
  headers: Headers,
  signal?: AbortSignal
): Promise<Response> {
  const controller = new AbortController()
  const abort = () => controller.abort(signal?.reason)
  const cleanup = () => signal?.removeEventListener('abort', abort)
  signal?.addEventListener('abort', abort, { once: true })

  if (signal?.aborted) {
    abort()
  }

  try {
    controller.signal.throwIfAborted()
    const response = await fetcher(headers, controller.signal)

    if (!response.body) {
      cleanup()

      return response
    }

    const reader = response.body.getReader()
    let cancelled = false

    const body = new ReadableStream<Uint8Array>({
      async pull(target) {
        try {
          const next = await reader.read()

          if (cancelled) {
            return
          }

          if (next.done) {
            cleanup()
            reader.releaseLock()
            target.close()
          } else {
            target.enqueue(next.value)
          }
        } catch (error) {
          cleanup()
          controller.abort(error)

          if (!cancelled) {
            reader.releaseLock()
            target.error(error)
          }
        }
      },
      async cancel(reason) {
        cancelled = true
        controller.abort(reason)
        cleanup()

        try {
          await reader.cancel(reason)
        } finally {
          reader.releaseLock()
        }
      }
    })

    return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers })
  } catch (error) {
    cleanup()
    controller.abort(error)
    throw error
  }
}
