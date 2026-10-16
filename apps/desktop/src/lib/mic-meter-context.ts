// The mic level meters (the recorder's VAD and the barge-in monitor) each open
// an AudioContext on the capture stream. close() is async, so a context torn
// down at the end of one turn can still hold the device when the next turn
// opens its own. Overlapping capture contexts are what make Chromium raise
// "The AudioContext encountered an error from the audio device" (#75329), so
// closes go through here and a new meter waits for them first.

// A close() on a wedged device must not hold the next turn forever.
const CLOSE_WAIT_CAP_MS = 1_000

let pendingClose: Promise<void> = Promise.resolve()

export function closeMeterContext(context: AudioContext | null | undefined) {
  if (!context || context.state === 'closed') {
    return
  }

  let closing: Promise<void>

  try {
    closing = context.close().catch(() => undefined)
  } catch {
    return
  }

  const previous = pendingClose
  pendingClose = Promise.all([previous, closing]).then(() => undefined)
}

/** Resolves once every meter context closed so far has finished closing. */
export function meterContextsClosed(): Promise<void> {
  return Promise.race([pendingClose, new Promise<void>(resolve => window.setTimeout(resolve, CLOSE_WAIT_CAP_MS))])
}
