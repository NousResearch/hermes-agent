// Pure, framework-free helpers for the Provider Manager's custom-provider form.
//
// Persistence and enablement are handled by the canonical REST endpoints API
// (`/api/providers/custom-endpoints`), which writes the keyed `providers:` schema
// and stores API keys in `.env` behind `key_env` — see `use-provider-config.ts`.
// This module only holds the form's data types and the name/id normalization
// helpers, so they can be unit tested without React or the backend.

export type CustomProviderApiMode = 'chat_completions' | 'anthropic_messages'

/** A single model exposed by a custom provider. */
export interface CustomProviderModel {
  /** Stable model id sent to the backend (e.g. "gpt-4o"). */
  id: string
  /** Optional display name shown in the UI instead of the raw id. */
  name?: string
  /** Optional provider-specific advanced parameters (unused for now). */
  advanced?: Record<string, unknown>
}

export interface CustomProviderEntry {
  /** Display + identity name. Unique after normalization. */
  name: string
  /** OpenAI-compatible (or Anthropic) base URL, e.g. https://my.host/v1 */
  base_url: string
  /** Optional secret API key. Empty string means "leave existing". */
  api_key?: string
  api_mode?: CustomProviderApiMode
  /** Models the user wants this provider to expose. */
  models: CustomProviderModel[]
}

// Mirror the backend's _normalize_custom_pool_name: lowercased, spaces -> '-'.
export function normalizeProviderName(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '-')
}

/**
 * Generate a clean, unique internal provider id from a display name.
 *
 * The custom-endpoint identity (the `providers.<id>` key, surfaced as catalog
 * slug `custom:<id>`) is derived from the name. The Add-Provider modal lets the
 * user type a friendly name and silently turns it into a machine-safe, unique id
 * stored as the provider's name.
 *
 * Normalizes (lowercase, spaces → '-', strip anything not alnum/'-'), then
 * appends a numeric suffix (-2, -3, …) until the id is unique among
 * `existingIds`. Falls back to "provider" when the name normalizes to nothing.
 */
export function generateProviderId(displayName: string, existingIds: string[]): string {
  const base =
    normalizeProviderName(displayName)
      .replace(/[^a-z0-9-]/g, '')
      .replace(/-+/g, '-')
      .replace(/^-+|-+$/g, '') || 'provider'

  const taken = new Set(existingIds.map(normalizeProviderName))

  if (!taken.has(base)) {
    return base
  }

  let n = 2
  while (taken.has(`${base}-${n}`)) {
    n++
  }

  return `${base}-${n}`
}
