import { expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n'

import { ERROR_CODE_KEYS, parseErrorSurface } from './error-surface'
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

it('renders every classified Russian error card in Russian, preserving provider names', () => {
  const russian = TRANSLATIONS.ru.assistant.thread
  const english = TRANSLATIONS.en.assistant.thread

  for (const code of ERROR_CODE_KEYS) {
    const surface = parseErrorSurface({ layer: 'provider', code, provider_label: 'Provider Ω' })
    const copy = errorCardText(russian, surface)
    const source = errorCardText(english, surface)
    expect(copy.title, code).not.toBe(source.title)
    expect(copy.body, code).not.toBe(source.body)
    if (source.title.includes('Provider Ω')) expect(copy.title, code).toContain('Provider Ω')
    if (source.body.includes('Provider Ω')) expect(copy.body, code).toContain('Provider Ω')
  }

  for (const layer of ['provider', 'auth', 'billing', 'gateway', 'disk', 'streaming'] as const) {
    const surface = parseErrorSurface({ layer, code: 'unknown_failure' })
    const copy = errorCardText(russian, surface)
    const source = errorCardText(english, surface)
    expect(copy.title, layer).not.toBe(source.title)
    expect(copy.body, layer).not.toBe(source.body)
  }

  for (const auth_kind of ['api_key', 'oauth'] as const) {
    const surface = parseErrorSurface({ layer: 'auth', code: 'auth', auth_kind, provider_label: 'Provider Ω' })
    const copy = errorCardText(russian, surface)
    const source = errorCardText(english, surface)
    expect(copy.title, auth_kind).not.toBe(source.title)
    expect(copy.body, auth_kind).not.toBe(source.body)
    expect(copy.title, auth_kind).toContain('Provider Ω')
    expect(copy.body, auth_kind).toContain('Provider Ω')
  }
})
