/**
 * Backend-delivered language packs. A Hermes language-pack plugin (or a user
 * overlay under `$HERMES_HOME/locales/`) ships `<lang>.desktop.yaml`; the
 * gateway serves the flattened strings over `i18n.catalog {lang, surface:
 * 'desktop'}` and the selectable languages over `i18n.languages`. Both land
 * in the app-locale registry under source `backend`, merged over the bundled
 * catalog exactly like a plugin registration.
 *
 * Pure: the caller hands in the JSON-RPC door. Older gateways predate both
 * methods — method-not-found is silence, not an error, and any other failure
 * leaves whatever the registry already holds (never blank the UI over i18n).
 */

import { isRecord } from '@hermes/shared/i18n'

import { isMissingRpcMethod } from '@/lib/gateway-rpc'

import { type AppLocaleRegistration, normalizeLocaleId, replaceAppLocaleSource } from './registry'

export type BackendLocaleRequest = <T>(method: string, params: Record<string, unknown>) => Promise<T>

/** One row of `i18n.languages` (`agent.i18n.language_options()`). */
export interface BackendLanguageOption {
  id: string
  endonym?: string
  rtl?: boolean
  source?: string
}

/** `i18n.catalog` reply: the pack + overlay layer for one surface, flat. */
export interface BackendLocaleCatalog {
  lang: string
  surface: string
  messages: Record<string, string>
}

export const DESKTOP_SURFACE = 'desktop'

function languageRows(value: unknown): BackendLanguageOption[] {
  const rows = isRecord(value) && Array.isArray(value.languages) ? value.languages : Array.isArray(value) ? value : []

  return rows.filter(
    (row): row is BackendLanguageOption => isRecord(row) && typeof row.id === 'string' && row.id.trim() !== ''
  )
}

function catalogMessages(value: unknown): Record<string, string> {
  if (!isRecord(value) || !isRecord(value.messages)) {
    return {}
  }

  const messages: Record<string, string> = {}

  for (const [key, text] of Object.entries(value.messages)) {
    if (typeof text === 'string') {
      messages[key] = text
    }
  }

  return messages
}

async function tolerant<T>(call: () => Promise<T>): Promise<T | null> {
  try {
    return await call()
  } catch (error) {
    if (!isMissingRpcMethod(error)) {
      console.debug('[i18n] backend language pack unavailable', error)
    }

    return null
  }
}

/**
 * Pull the backend's language list plus the desktop pack for `lang` and swap
 * them into the registry's `backend` layer. `isCurrent` guards the apply: a
 * profile switch or a newer language pick mid-flight must not write a stale
 * pack over the winner's.
 */
export async function syncBackendLocalePacks(
  request: BackendLocaleRequest,
  lang: null | string,
  isCurrent: () => boolean = () => true
): Promise<void> {
  const wanted = lang ? normalizeLocaleId(lang) : ''

  const [languages, catalog] = await Promise.all([
    tolerant(() => request<unknown>('i18n.languages', {})),
    wanted ? tolerant(() => request<unknown>('i18n.catalog', { lang: wanted, surface: DESKTOP_SURFACE })) : null
  ])

  if (!isCurrent()) {
    return
  }

  if (languages === null && catalog === null) {
    // Nothing answered (old backend, socket dropped): keep the current layer.
    return
  }

  const registrations = new Map<string, { id: string } & AppLocaleRegistration>()

  for (const row of languageRows(languages)) {
    const id = normalizeLocaleId(row.id)

    if (id) {
      registrations.set(id, {
        id,
        endonym: typeof row.endonym === 'string' ? row.endonym : undefined,
        rtl: typeof row.rtl === 'boolean' ? row.rtl : undefined
      })
    }
  }

  const messages = catalogMessages(catalog)

  if (wanted && Object.keys(messages).length) {
    const existing = registrations.get(wanted) ?? { id: wanted }
    registrations.set(wanted, { ...existing, translations: messages })
  }

  replaceAppLocaleSource('backend', [...registrations.values()])
}
