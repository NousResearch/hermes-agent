import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'
import type { ModelOptionProvider } from '@/types/hermes'

function isValidHttpUrl(value: string): boolean {
  try {
    const url = new URL(value)
    return url.protocol === 'http:' || url.protocol === 'https:'
  } catch {
    return false
  }
}

export interface ProviderConfigurePanelProps {
  provider: ModelOptionProvider
  /** True while the configure → discover round-trip is in flight. */
  working: boolean
  /** Error message surfaced by the parent after a failed save. */
  error: string | null
  /** Persist the API key (+ optional base URL override) and discover models. */
  onConfigure: (apiKey: string, baseUrl?: string) => void
}

/**
 * Inline configuration panel shown in the Provider Manager's right pane when an
 * unconfigured (but inline-configurable) built-in provider is selected. Lets the
 * user paste an API key, optionally override the base URL, and save — which
 * writes the env var, re-probes the catalog, and discovers the provider's models.
 *
 * The panel is intentionally NOT a modal: it occupies the same content slot as
 * the model list, so the transition from "configure" → "model list" after a
 * successful save is a seamless in-place swap driven by the refreshed catalog.
 */
export function ProviderConfigurePanel({ provider, working, error, onConfigure }: ProviderConfigurePanelProps) {
  const { t } = useI18n()
  const p = t.providerManager

  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState('')

  // Reset the fields whenever the user switches to a different provider so a
  // half-typed key never leaks across providers.
  useEffect(() => {
    setApiKey('')
    setBaseUrl('')
  }, [provider.slug])

  const urlInvalid = baseUrl.trim() !== '' && !isValidHttpUrl(baseUrl.trim())
  const canSubmit = !working && apiKey.trim() !== '' && !urlInvalid

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (!canSubmit) {
      return
    }

    onConfigure(apiKey.trim(), baseUrl.trim() || undefined)
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto px-3 py-2">
      <div className="mb-1 text-sm font-medium">{provider.name}</div>
      <p className="mb-4 text-[0.6875rem] leading-relaxed text-(--ui-text-tertiary)">{p.configureDescription}</p>

      <form className="grid max-w-md gap-4" onSubmit={handleSubmit}>
        <div className="grid gap-1.5">
          <label className="text-xs font-medium" htmlFor="configure-panel-api-key">
            {p.apiKey}
          </label>
          <Input
            autoFocus
            disabled={working}
            id="configure-panel-api-key"
            onChange={event => setApiKey(event.target.value)}
            placeholder={p.apiKeyPlaceholder}
            type="password"
            value={apiKey}
          />
        </div>

        <div className="grid gap-1.5">
          <label className="text-xs font-medium" htmlFor="configure-panel-base-url">
            {p.baseUrlOverride}
          </label>
          <Input
            aria-invalid={urlInvalid}
            disabled={working}
            id="configure-panel-base-url"
            onChange={event => setBaseUrl(event.target.value)}
            placeholder={p.baseUrlOverridePlaceholder}
            value={baseUrl}
          />
        </div>

        {(error || urlInvalid) && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <span>{urlInvalid ? p.invalidUrl : error}</span>
          </div>
        )}

        <div>
          <Button disabled={!canSubmit} type="submit">
            {working ? p.configuring : p.saveAndDiscover}
          </Button>
        </div>
      </form>
    </div>
  )
}
