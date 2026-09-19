import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import { ERROR_CODE_KEYS, errorRecoveryPlan, type ErrorSurface, formatErrorDiagnostics, parseErrorSurface } from './error-surface'
import { errorCardText } from './error-surface-copy'

describe('parseErrorSurface', () => {
  it('accepts a valid descriptor', () => {
    expect(parseErrorSurface({ layer: 'streaming', code: 'stream_drop', retryable: true })).toEqual({
      layer: 'streaming',
      code: 'stream_drop',
      retryable: true
    })
  })

  it('accepts every documented layer', () => {
    for (const layer of ['provider', 'endpoint', 'streaming', 'auth', 'billing', 'gateway', 'runtime', 'disk']) {
      expect(parseErrorSurface({ layer, code: 'x', retryable: false })?.layer).toBe(layer)
    }
  })

  it('rejects unknown layers and non-objects', () => {
    expect(parseErrorSurface({ layer: 'blockchain', code: 'x', retryable: true })).toBeNull()
    expect(parseErrorSurface('provider')).toBeNull()
    expect(parseErrorSurface(null)).toBeNull()
    expect(parseErrorSurface(undefined)).toBeNull()
    expect(parseErrorSurface(7)).toBeNull()
  })

  it('defaults code and retryable when missing', () => {
    expect(parseErrorSurface({ layer: 'gateway' })).toEqual({ layer: 'gateway', code: 'unknown', retryable: true })
  })

  it('honors retryable=false', () => {
    expect(parseErrorSurface({ layer: 'auth', code: 'auth_permanent', retryable: false })?.retryable).toBe(false)
  })

  it('carries the failing session identity when present', () => {
    const surface = parseErrorSurface({
      layer: 'provider',
      code: 'rate_limit',
      retryable: true,
      provider: 'openrouter',
      model: 'test/m1'
    })

    expect(surface?.provider).toBe('openrouter')
    expect(surface?.model).toBe('test/m1')
    // Absent identity yields no keys, not empty strings.
    expect(parseErrorSurface({ layer: 'provider', code: 'x', retryable: true })?.provider).toBeUndefined()
  })
})

describe('formatErrorDiagnostics', () => {
  it('includes layer, code, model and error', () => {
    const text = formatErrorDiagnostics({
      errorText: 'boom',
      model: 'anthropic/claude-opus-4.6',
      surface: { layer: 'provider', code: 'rate_limit', retryable: true }
    })

    expect(text).toContain('layer: provider')
    expect(text).toContain('code: rate_limit')
    expect(text).toContain('model: anthropic/claude-opus-4.6')
    expect(text).toContain('error: boom')
  })

  it('prefers the descriptor identity over the caller fallback', () => {
    const text = formatErrorDiagnostics({
      errorText: 'boom',
      // Foreground composer atom — potentially stale by click time.
      model: 'some/other-model',
      surface: { layer: 'provider', code: 'rate_limit', retryable: true, provider: 'openrouter', model: 'failed/model' }
    })

    expect(text).toContain('provider: openrouter')
    expect(text).toContain('model: failed/model')
    expect(text).not.toContain('some/other-model')
  })

  it('omits absent fields without leaving blank lines', () => {
    const text = formatErrorDiagnostics({ errorText: 'boom' })

    expect(text).not.toContain('layer:')
    expect(text).not.toContain('model:')
    expect(text.split('\n').every(line => line.trim().length > 0)).toBe(true)
  })
})

// The card body and the buttons under it must agree: a body that says "retry"
// while the plan hides the Retry button leaves the user with an instruction
// they cannot follow. Walks every code the backend can send (plus the layer
// fallbacks) with the non-retryable verdict the classifier stamps for it.
describe('error copy never names a hidden Retry', () => {
  const thread = en.assistant.thread
  const RETRY_WORDS = /\bretry\b|\btry again\b/i

  // Verdicts the classifier stamps as deterministic (agent/error_surface.py
  // `_NON_RETRYABLE_REASONS`); everything else arrives retryable.
  const NON_RETRYABLE = new Set([
    'auth',
    'auth_permanent',
    'billing',
    'content_policy_blocked',
    'provider_policy_blocked',
    'model_not_found',
    'format_error',
    'ssl_cert_verification',
    'context_overflow',
    'interpreter_shutdown'
  ])

  const surfaces: ErrorSurface[] = [
    ...ERROR_CODE_KEYS.map(code => ({ code, layer: 'provider' as const, retryable: !NON_RETRYABLE.has(code) })),
    { code: 'auth', layer: 'auth', retryable: false },
    { code: 'auth_permanent', layer: 'auth', retryable: false },
    { code: 'ssl_cert_verification', layer: 'endpoint', retryable: false },
    { code: 'interpreter_shutdown', layer: 'gateway', retryable: false },
    { code: 'interpreter_shutdown', layer: 'runtime', retryable: false },
    { code: 'unknown', layer: 'endpoint', retryable: false }
  ]

  it.each(surfaces.map(surface => [surface.code, surface.layer, surface] as const))(
    '%s on %s',
    (_code, _layer, surface) => {
      const plan = errorRecoveryPlan(surface)
      const { body } = errorCardText(thread, surface)

      if (!plan.retry) {
        expect(body).not.toMatch(RETRY_WORDS)
      }
    }
  )

  it('a credential rejection keeps Retry, so its body may still say retry', () => {
    const surface: ErrorSurface = { authKind: 'api_key', code: 'auth', layer: 'auth', provider: 'openai', retryable: false }
    expect(errorRecoveryPlan(surface).retry).toBe(true)
  })
})

describe('free-tier refusals', () => {
  const surface = parseErrorSurface({
    code: 'free_tier_disabled',
    layer: 'provider',
    message: '  Using Hermes without signing in is switched off right now. To sign in: /login. ',
    provider: 'nous',
    retryable: false
  })!

  it('carries the backend sentence and offers the free sign-in, never an OAuth re-login', () => {
    expect(surface.message).toBe('Using Hermes without signing in is switched off right now. To sign in: /login.')
    const plan = errorRecoveryPlan(surface)
    expect(plan.signInFreeTier).toBe(true)
    expect(plan.signInAgain).toBe(false)
    expect(plan.retry).toBe(false)
    expect(plan.switchProvider).toBe(true)
  })

  it('renders the backend sentence as the body under its own title', () => {
    const { body, title } = errorCardText(en.assistant.thread, surface)
    expect(title).toBe(en.assistant.thread.errorCodes.free_tier_disabled.title)
    expect(body).toBe(surface.message)
  })

  it('falls back to the table body when an older backend sent no sentence', () => {
    const bare = parseErrorSurface({ code: 'free_tier_rate_limited', layer: 'provider', retryable: true })!
    expect(errorCardText(en.assistant.thread, bare).body).toBe(en.assistant.thread.errorCodes.free_tier_rate_limited.body)
    expect(errorRecoveryPlan(bare).retry).toBe(true)
  })

  it('every free-tier code has copy and the copy never blames the free model', () => {
    for (const code of ERROR_CODE_KEYS.filter(key => key.startsWith('free_tier_'))) {
      const copy = en.assistant.thread.errorCodes[code]
      const text = `${typeof copy.title === 'string' ? copy.title : ''} ${typeof copy.body === 'string' ? copy.body : ''}`.toLowerCase()
      expect(text).not.toMatch(/free (service|model|tier) is (off|switched off|unavailable|down)/)
      expect(text).not.toMatch(/anonymous|guest|credential|token|rate limit/)
    }
  })
})
