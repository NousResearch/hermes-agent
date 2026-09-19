// Resolves the error card's title/body for a failed turn from the i18n tables
// (`assistant.thread.errorCodes` / `errorAuthKinds` / `errorLayers`), so the
// inline card and the global gateway-error toast read the same words for the
// same failure. The message schema is derived from `i18n/en.ts` and the error
// code list lives in error-surface.ts — hence the sibling module.

import type { Translations } from '@/i18n'

import { errorCardKey, type ErrorSurface, isFreeTierSurface } from './error-surface'

/** One error-card entry: a short title and one plain sentence. Either may
 *  take the failing provider's display name (falls back to "the AI service"). */
export interface ErrorCardCopy {
  title: string | ((provider: string) => string)
  body: string | ((provider: string) => string)
}

export interface ErrorCardText {
  title: string
  body: string
}

const render = (value: ErrorCardCopy['title'], provider: string) =>
  typeof value === 'function' ? value(provider) : value

/** The failing provider's display name for copy — the descriptor's label,
 *  else its id, else the generic "the AI service". */
export function errorProviderName(thread: Translations['assistant']['thread'], surface?: ErrorSurface | null): string {
  return surface?.providerLabel || surface?.provider || thread.errorGenericProvider
}

export function errorCardText(
  thread: Translations['assistant']['thread'],
  surface: ErrorSurface | null | undefined
): ErrorCardText {
  const provider = errorProviderName(thread, surface)

  // A credential rejection is worded by HOW the provider is credentialed
  // (key vs sign-in), which the code alone (`auth`) cannot tell.
  if (surface?.layer === 'auth' && surface.authKind === 'oauth') {
    return { body: thread.errorOauthExpired(provider), title: render(thread.errorAuthKinds.oauth.title, provider) }
  }

  if (surface?.layer === 'auth' && surface.authKind === 'api_key') {
    const copy = thread.errorAuthKinds.api_key

    return { body: render(copy.body, provider), title: render(copy.title, provider) }
  }

  const key = errorCardKey(surface)

  if ('code' in key) {
    const copy = thread.errorCodes[key.code]

    // A free-tier refusal arrives with the backend's own sentence (the wait, the
    // model, the way forward); the table body is only the fallback for an older backend.
    return {
      body: (isFreeTierSurface(surface) && surface?.message) || render(copy.body, provider),
      title: render(copy.title, provider)
    }
  }

  return { body: thread.errorLayerBodies[key.layer], title: thread.errorLayers[key.layer] }
}
