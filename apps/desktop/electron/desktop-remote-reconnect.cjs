/**
 * desktop-remote-reconnect.cjs
 *
 * Focused utilities for remote-backend reconnect behavior used by
 * electron/main.cjs: bounded remote startup probing and recovery-state
 * accounting for transient /api/status misses.
 *
 * Kept as a pure module so we can unit-test timing, threshold, and state
 * transitions with deterministic clocks instead of booting the Electron shell.
 */

const DEFAULT_REMOTE_PROBE_TIMEOUT_MS = 90_000
const DEFAULT_REMOTE_PROBE_INITIAL_DELAY_MS = 500
const DEFAULT_REMOTE_PROBE_MAX_DELAY_MS = 2_000
const DEFAULT_REMOTE_PROBE_MAX_ATTEMPTS = 16
const DEFAULT_REMOTE_REVALIDATE_FAILURE_LIMIT = 3
const DEFAULT_REMOTE_REVALIDATE_FAILURE_WINDOW_MS = 20_000

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function toPositiveInteger(value, fallback) {
  return Number.isFinite(value) && value > 0 ? Math.trunc(value) : fallback
}

function backoffDelay(attempt, initialDelayMs, maxDelayMs) {
  const capped = Math.pow(2, Math.max(0, attempt - 1))
  return Math.min(initialDelayMs * capped, maxDelayMs)
}

async function waitForRemoteHermes(baseUrl, token, options = {}) {
  const fetcher = options.fetcher
  if (typeof fetcher !== 'function') {
    throw new Error('waitForRemoteHermes requires a fetcher function')
  }

  const timeoutMs = toPositiveInteger(options.timeoutMs, DEFAULT_REMOTE_PROBE_TIMEOUT_MS)
  const maxAttempts = toPositiveInteger(options.maxAttempts, DEFAULT_REMOTE_PROBE_MAX_ATTEMPTS)
  const initialDelayMs = toPositiveInteger(options.initialDelayMs, DEFAULT_REMOTE_PROBE_INITIAL_DELAY_MS)
  const maxDelayMs = toPositiveInteger(options.maxDelayMs, DEFAULT_REMOTE_PROBE_MAX_DELAY_MS)
  const deadline = Date.now() + timeoutMs
  const wait = options.sleep || sleep
  let attempts = 0
  let lastError = null

  while (Date.now() < deadline) {
    try {
      await fetcher(`${baseUrl}/api/status`, token, options.fetcherOptions)
      return
    } catch (error) {
      lastError = error
      attempts += 1

      if (attempts >= maxAttempts) {
        break
      }

      const delay = backoffDelay(attempts, initialDelayMs, maxDelayMs)
      const remainingMs = deadline - Date.now()
      if (remainingMs <= delay) {
        break
      }
      await wait(delay)
    }
  }

  const message = lastError?.message || 'timeout'
  throw new Error(`Remote Hermes backend did not become ready after ${attempts} attempts: ${message}`)
}

function createRemoteRevalidateState() {
  return {
    failures: 0,
    windowStartedAt: 0
  }
}

function resetRemoteRevalidateState(state) {
  if (!state) return
  state.failures = 0
  state.windowStartedAt = 0
}

function recordRemoteRevalidateFailure(state, options = {}) {
  if (!state) {
    throw new Error('recordRemoteRevalidateFailure requires a state object')
  }

  const now = toPositiveInteger(options.now, Date.now())
  const failureLimit = toPositiveInteger(options.failureLimit, DEFAULT_REMOTE_REVALIDATE_FAILURE_LIMIT)
  const failureWindowMs = toPositiveInteger(options.failureWindowMs, DEFAULT_REMOTE_REVALIDATE_FAILURE_WINDOW_MS)
  const inWindow = state.failures > 0 && state.windowStartedAt > 0 && now - state.windowStartedAt <= failureWindowMs

  if (!inWindow) {
    state.failures = 1
    state.windowStartedAt = now
    return { failures: 1, shouldReset: false }
  }

  state.failures += 1
  return {
    failures: state.failures,
    shouldReset: state.failures >= failureLimit
  }
}

function shouldRemoteRevalidateKeepState(state) {
  return state && state.failures > 0
}

module.exports = {
  createRemoteRevalidateState,
  DEFAULT_REMOTE_PROBE_INITIAL_DELAY_MS,
  DEFAULT_REMOTE_PROBE_MAX_ATTEMPTS,
  DEFAULT_REMOTE_PROBE_MAX_DELAY_MS,
  DEFAULT_REMOTE_PROBE_TIMEOUT_MS,
  DEFAULT_REMOTE_REVALIDATE_FAILURE_LIMIT,
  DEFAULT_REMOTE_REVALIDATE_FAILURE_WINDOW_MS,
  recordRemoteRevalidateFailure,
  resetRemoteRevalidateState,
  shouldRemoteRevalidateKeepState,
  waitForRemoteHermes
}
