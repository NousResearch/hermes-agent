// Match the image attachment byte budget without limiting browser-managed files.
export const BROWSER_IMAGE_DOWNLOAD_MAX_BYTES = 25 * 1024 * 1024
export const BROWSER_IMAGE_DOWNLOAD_TIMEOUT_MS = 30_000

/** Fetch only cross-origin images here; gateway files stay browser-managed. */
export async function fetchBrowserImage(url: URL): Promise<Blob> {
  const controller = new AbortController()
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
  let timer: number | undefined

  const deadline = new Promise<never>((_, reject) => {
    timer = window.setTimeout(() => {
      reject(new Error('Image download timed out after 30 seconds. Try again or open the image in a separate tab.'))
    }, BROWSER_IMAGE_DOWNLOAD_TIMEOUT_MS)
  })

  const download = async () => {
    const response = await fetch(url, { credentials: 'omit', mode: 'cors', signal: controller.signal })
    controller.signal.throwIfAborted()
    reader = response.body?.getReader()

    if (!response.ok) {throw new Error(`Image download failed (${response.status})`)}

    if (!reader) {throw new Error('Image download has no readable body. Try opening the image in a separate tab.')}

    const chunks: Uint8Array<ArrayBuffer>[] = []
    let received = 0

    while (true) {
      const { done, value } = await reader.read()
      controller.signal.throwIfAborted()

      if (done) {break}
      received += value.byteLength

      // Content-Length may be absent, incorrect, or describe compressed bytes.
      if (received > BROWSER_IMAGE_DOWNLOAD_MAX_BYTES) {
        throw new Error('Image download exceeds 25 MiB. Open the image in a separate tab to save it.')
      }

      chunks.push(value.slice())
    }

    return new Blob(chunks, { type: response.headers.get('content-type') || '' })
  }

  try {
    return await Promise.race([download(), deadline])
  } catch (error) {
    // Do not wait for a stalled cancellation handshake before reporting failure.
    void reader?.cancel(error).catch(() => undefined)
    controller.abort(error)
    throw error
  } finally {
    window.clearTimeout(timer)
    reader?.releaseLock()
  }
}
