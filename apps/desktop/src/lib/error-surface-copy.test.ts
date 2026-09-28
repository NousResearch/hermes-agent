import { expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n'

import { parseErrorSurface } from './error-surface'
import { errorCardText } from './error-surface-copy'

const codes = [
  'provider_policy_blocked',
  'content_policy_blocked',
  'format_error',
  'invalid_response',
  'empty_response',
  'rate_limit',
  'upstream_rate_limit',
  'overloaded',
  'server_error',
  'timeout',
  'ssl_cert_verification'
] as const

it.each(['zh', 'zh-hant', 'ja'] as const)(
  'renders provider errors in %s without losing the failing provider identity',
  locale => {
    for (const layer of ['network', 'provider', 'endpoint', 'streaming'] as const) {
      const surface = parseErrorSurface({ layer, code: 'unknown_failure', retryable: true })
      const copy = errorCardText(TRANSLATIONS[locale].assistant.thread, surface)
      const english = errorCardText(TRANSLATIONS.en.assistant.thread, surface)
      expect(copy.title, layer).not.toBe(english.title)
      expect(copy.body, layer).not.toBe(english.body)
    }

    expect(TRANSLATIONS[locale].assistant.thread.errorGenericProvider).not.toBe(
      TRANSLATIONS.en.assistant.thread.errorGenericProvider
    )

    for (const code of codes) {
      const surface = parseErrorSurface({
        layer: 'provider',
        code,
        provider: 'fixture-provider',
        provider_label: 'Provider Ω',
        retryable: true
      })

      const copy = errorCardText(TRANSLATIONS[locale].assistant.thread, surface)
      const english = errorCardText(TRANSLATIONS.en.assistant.thread, surface)
      expect(copy.title, code).not.toBe(english.title)
      expect(copy.body, code).not.toBe(english.body)
      expect(copy.body, code).toContain('Provider Ω')
      expect(copy.body, code).not.toContain('fixture-provider')
    }
  }
)

it('does not blame a slow reply when the provider could not be reached', () => {
  // The backend stamps `timeout` for refused connections and DNS failures too
  // (agent/error_classifier.py _CONNECTION_MESSAGE_PATTERNS).
  const surface = parseErrorSurface({
    layer: 'provider',
    code: 'timeout',
    provider_label: 'OpenCode Go',
    retryable: true
  })

  const copy = errorCardText(TRANSLATIONS.en.assistant.thread, surface)

  expect(copy.title).toBe('Could not reach the AI service')
  expect(copy.body).toContain('OpenCode Go could not be reached')
})
