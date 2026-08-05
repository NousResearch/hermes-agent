// The `hermes:api` expected-404 seam, shared by the main-process handler and
// preload so both sides agree on one literal.
//
// Electron logs "Error occurred in handler for 'hermes:api'" with a full stack
// trace for every rejected `ipcMain.handle` invoke, and offers no way to opt a
// handler out. Desktop's session resolution is a deliberate probe ladder
// (`resolveStoredSession`: cache → active backend → each other profile), so a
// 404 there is a normal rung outcome the renderer already handles — but each
// one printed a stack into the launching terminal.
//
// The handler therefore RESOLVES with this sentinel for that one expected case,
// and preload rethrows it as an ordinary `Error` carrying the identical
// `404: <body>` message. The renderer sees exactly the rejection it saw before;
// only Electron's logging is bypassed. Any other failure rejects as usual and
// still logs in full.

const HERMES_API_EXPECTED_404 = '__hermesExpected404__'

// True when `value` is a handler-resolved expected-404 sentinel, not real
// response data. Kept narrow: a plain object whose ONLY key is the sentinel and
// whose value is a string, so a backend payload can't be mistaken for one.
function isExpectedNotFoundSentinel(value: unknown): value is Record<string, string> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const keys = Object.keys(value as Record<string, unknown>)

  return keys.length === 1 && keys[0] === HERMES_API_EXPECTED_404 && typeof (value as any)[HERMES_API_EXPECTED_404] === 'string'
}

// Restore the caller-visible contract: a sentinel becomes the rejection the
// renderer expects; anything else passes through untouched.
function unwrapExpectedNotFound(value: unknown): unknown {
  if (isExpectedNotFoundSentinel(value)) {
    throw new Error(value[HERMES_API_EXPECTED_404])
  }

  return value
}

export { HERMES_API_EXPECTED_404, isExpectedNotFoundSentinel, unwrapExpectedNotFound }
