/**
 * Optional IPC channels (telemetry drains, capability probes) may be missing
 * from the running main process when the renderer is newer than the process
 * serving it — a source-tree update while the app keeps running leaves the
 * reloaded renderer invoking channels the in-memory bundle never registers.
 * Electron prints `Error occurred in handler for '<channel>'` in the MAIN
 * process console for every such invoke, before any renderer-side catch can
 * swallow it, so the only way to quiet the stream is to stop invoking.
 *
 * These helpers give optional-channel callers a per-session latch: the first
 * `No handler registered` rejection marks the channel dead, and every later
 * call resolves as if the channel were absent (the caller's optional-chaining
 * path). A reload of the renderer resets the latch, which is correct: the
 * next preload may come from a build whose main does register the channel.
 */

const NO_HANDLER_MARKER = 'No handler registered'

const deadChannels = new Set<string>()

function isNoHandlerRejection(error: unknown): boolean {
  return error instanceof Error && error.message.includes(NO_HANDLER_MARKER)
}

/**
 * Invoke an optional channel through a bridge accessor, latching the channel
 * dead for the session once the running main reports it has no handler.
 * Returns undefined when the channel is dead, the bridge is missing, or the
 * method is absent — the same shape the caller already treats as "nothing to
 * drain", so an old main costs one probe per channel per session.
 */
export async function invokeOptionalIpc<T>(
  channel: string,
  invoke: () => Promise<T> | undefined
): Promise<T | undefined> {
  if (deadChannels.has(channel)) {
    return undefined
  }

  try {
    return await invoke()
  } catch (error) {
    if (isNoHandlerRejection(error)) {
      deadChannels.add(channel)
    }

    throw error
  }
}

/** Test-only: forget the per-session latch so suites start unbiased. */
export function resetOptionalIpcLatchesForTests(): void {
  deadChannels.clear()
}
