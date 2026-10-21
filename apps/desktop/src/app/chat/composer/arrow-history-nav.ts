/** A request to walk the composer's sent-message history ring. */
export type HistoryStepRequest =
  | { step: 'forward'; browsing: boolean; enabled: boolean }
  | { step: 'backward'; browsing: boolean; draft: string; enabled: boolean }

/**
 * Whether an arrow press may walk sent-message history.
 *
 * ↓ only ever walks back to the present, so it needs an open ring. ↑ opens the
 * ring from an untouched composer and keeps stepping while browsing, but never
 * hijacks a typed draft the user did not recall. Both are off when the user has
 * turned the arrows off in Settings.
 */
export function historyStepAllowed(request: HistoryStepRequest): boolean {
  if (!request.enabled) {
    return false
  }

  if (request.step === 'forward') {
    return request.browsing
  }

  return request.browsing || request.draft.trim().length === 0
}
