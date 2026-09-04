/**
 * IPC error envelope for the `hermes:api` channel.
 *
 * Electron logs EVERY rejected `ipcMain.handle` promise itself, with a full
 * main-process stack trace and no way to opt out per channel. That is right for
 * genuine faults and wrong for the one case the renderer treats as ordinary
 * control flow: a REST 404.
 *
 * `session-tile.tsx` deliberately probes a freshly created draft by id and
 * retries across the persist lag ("404s harmlessly for an in-memory draft that
 * hasn't persisted a turn yet"), and `resolveStoredSession` is fail-closed on a
 * miss by design. A discarded draft — `SessionDB.delete_session_if_empty()`
 * removes an empty one, while its id lives on in the renderer's journal /
 * owner-hint storage — therefore produces up to seven 404s per tile, each one
 * logged as
 *
 *   Error occurred in handler for 'hermes:api': Error: 404: {"detail":"Session not found"}
 *
 * so a normal session of work buries the log in stack traces for a case the
 * code already handles.
 *
 * The fix keeps BOTH renderer-visible contracts intact and changes only what
 * crosses the wire:
 *
 *  - the handler RESOLVES with an envelope instead of rejecting, so Electron
 *    sees a settled promise and logs nothing;
 *  - preload rethrows an `Error` whose `message` is byte-identical to the one
 *    the rejection produced, because 404 detection is message-based
 *    (`isMissingRestEndpoint` in `src/lib/gateway-rpc.ts` matches
 *    `/(?:^\s*|error:\s*)404\b/`) and `inlineErrorMessage` /
 *    `notifications.ts` unwrap the `Error invoking remote method '…': Error: …`
 *    shape. `statusCode` is preserved on the rethrown error.
 *
 * Only 404 is enveloped. Every other failure keeps rejecting and keeps being
 * logged — a 500, a timeout or a transport error is a real fault and must stay
 * visible.
 */

/** Marker on the envelope. Unlikely to collide with any REST response body. */
export const API_ERROR_ENVELOPE = '__hermesApiError'

export interface HermesApiErrorEnvelope {
  [API_ERROR_ENVELOPE]: true
  message: string
  statusCode: number
}

/** HTTP statuses the renderer handles as control flow, so they must not be
 *  logged as main-process faults. Deliberately just 404: a missing row or route
 *  is an answer, whereas 5xx/timeouts are faults that must stay noisy. */
function isEnvelopedStatus(statusCode: unknown): boolean {
  return statusCode === 404
}

/**
 * Wrap a caught `hermes:api` error for transport when it is one the renderer
 * treats as control flow; return null to let the caller keep rejecting.
 */
export function apiErrorEnvelope(error: unknown): HermesApiErrorEnvelope | null {
  const statusCode = (error as { statusCode?: unknown } | null)?.statusCode

  if (!isEnvelopedStatus(statusCode)) {
    return null
  }

  return {
    [API_ERROR_ENVELOPE]: true,
    message: error instanceof Error ? error.message : String(error ?? ''),
    statusCode: statusCode as number
  }
}

/** True when a resolved IPC value is an enveloped error rather than a payload. */
export function isApiErrorEnvelope(value: unknown): value is HermesApiErrorEnvelope {
  return Boolean(
    value &&
      typeof value === 'object' &&
      (value as Record<string, unknown>)[API_ERROR_ENVELOPE] === true &&
      typeof (value as HermesApiErrorEnvelope).message === 'string'
  )
}

/**
 * The main-process side: run the real dispatch and convert a control-flow
 * failure into a resolved envelope so Electron has no rejected handler promise
 * to log. Every other failure is rethrown and stays logged.
 *
 * `ipcMain.handle('hermes:api')` wraps its dispatch in this, and the wiring test
 * exercises THIS function — not a copy of the handler body.
 */
export async function settleApiRequest<T>(dispatch: () => Promise<T>): Promise<HermesApiErrorEnvelope | T> {
  try {
    return await dispatch()
  } catch (error) {
    const envelope = apiErrorEnvelope(error)

    if (envelope) {
      return envelope
    }

    throw error
  }
}

/**
 * The preload side: unwrap an envelope back into the exact rejection the caller
 * would have received, and pass any real payload through untouched.
 */
export async function unwrapApiResult<T>(invoke: () => Promise<HermesApiErrorEnvelope | T>): Promise<T> {
  const result = await invoke()

  if (isApiErrorEnvelope(result)) {
    throw apiErrorFromEnvelope(result)
  }

  return result as T
}

/**
 * Rebuild the Error the renderer would have received from a rejection. The
 * message is passed through verbatim so message-based 404 detection and the
 * remote-method unwrappers behave exactly as before.
 */
export function apiErrorFromEnvelope(envelope: HermesApiErrorEnvelope): Error {
  const error = new Error(envelope.message) as Error & { statusCode?: number }
  error.statusCode = envelope.statusCode

  return error
}
