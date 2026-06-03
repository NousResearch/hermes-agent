import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import {
  ApiKeyForm,
  type ApiKeyOption,
  FEATURED_ID,
  FeaturedProviderRow,
  KeyProviderRow,
  ProviderRow,
  sortProviders
} from '@/components/desktop-onboarding-overlay'
import { listOAuthProviders } from '@/hermes'
import { ChevronDown, Sparkles } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $desktopOnboarding, startManualProviderOAuth } from '@/store/onboarding'
import type { EnvVarInfo, OAuthProvider } from '@/types/hermes'

import { SettingsCategoryHeading, useEnvCredentials } from './env-credentials'
import { providerGroup, providerMeta, providerPriority } from './helpers'
import { LoadingState, SettingsContent } from './primitives'
import type { SearchProps } from './types'

// Sub-views surfaced as a sidebar subnav: account sign-in vs raw API keys.
export const PROVIDER_VIEWS = ['accounts', 'keys'] as const

export type ProviderView = (typeof PROVIDER_VIEWS)[number]

const isKeyVar = (key: string, info: EnvVarInfo) => info.is_password || /(?:_API_KEY|_TOKEN|_KEY)$/.test(key)

// Turn the env catalog into the same `ApiKeyOption` shape onboarding's curated
// list uses, so the shared `ApiKeyForm` can render the full provider catalog in
// the identical 2-column card grid. One card per provider group, keyed to its
// primary credential var; OAuth-only / no-key groups (e.g. Nous) are skipped.
function buildApiKeyOptions(vars: Record<string, EnvVarInfo>): ApiKeyOption[] {
  const groups = new Map<string, [string, EnvVarInfo][]>()

  for (const [key, info] of Object.entries(vars)) {
    if (info.category !== 'provider') {
      continue
    }

    const name = providerGroup(key)

    if (name === 'Other') {
      continue
    }

    groups.set(name, [...(groups.get(name) ?? []), [key, info]])
  }

  const options: ApiKeyOption[] = []

  for (const [name, entries] of groups) {
    const primary = entries.find(([k, i]) => !i.advanced && isKeyVar(k, i)) ?? entries.find(([k, i]) => isKeyVar(k, i))

    if (!primary) {
      continue
    }

    const meta = providerMeta(name)
    const description = meta?.description ?? primary[1].description

    options.push({
      description,
      docsUrl: meta?.docsUrl ?? primary[1].url ?? '',
      envKey: primary[0],
      id: name,
      name,
      short: description
    })
  }

  options.sort((a, b) => providerPriority(a.name) - providerPriority(b.name) || a.name.localeCompare(b.name))

  // Mirror onboarding's "Local / custom endpoint" option for bring-your-own
  // OpenAI-compatible servers.
  if (vars.OPENAI_BASE_URL) {
    options.push({
      description: 'Point Hermes at a local or self-hosted OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama, etc).',
      docsUrl: 'https://github.com/NousResearch/hermes-agent#bring-your-own-endpoint',
      envKey: 'OPENAI_BASE_URL',
      id: 'local',
      name: 'Local / custom endpoint',
      placeholder: 'http://127.0.0.1:8000/v1',
      short: 'self-hosted'
    })
  }

  return options
}

// Deliberately a near-1:1 replica of the first-run onboarding picker
// (`Picker` in desktop-onboarding-overlay): same recommended card, same
// provider rows, same "Other providers" disclosure, same OpenRouter quick-key
// row, and the same bottom-right "I have an API key" affordance. The leaf cards
// are the exact shared components, so the two surfaces stay visually identical.
// Selecting a provider hands off to the shared onboarding overlay, which runs
// that provider's real sign-in flow; the key affordances open the API-key
// catalog below.
function OAuthPicker({
  onWantApiKey,
  providers,
  query
}: {
  onWantApiKey: () => void
  providers: OAuthProvider[]
  query: string
}) {
  const [showAll, setShowAll] = useState(false)
  const ordered = useMemo(() => sortProviders(providers), [providers])

  if (ordered.length === 0) {
    return null
  }

  const select = (p: OAuthProvider) => startManualProviderOAuth(p.id)
  const q = query.trim().toLowerCase()

  // While searching, flatten the featured/connected/disclosure structure into a
  // single matched list so it reads the same as the API-keys grid does.
  if (q) {
    const matches = ordered.filter(p => p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q))

    return (
      <section className="grid gap-2">
        <SettingsCategoryHeading
          count={`${matches.length} match${matches.length === 1 ? '' : 'es'}`}
          icon={Sparkles}
          title="Accounts"
        />
        {matches.length === 0 ? (
          <EmptyMatches query={query} />
        ) : (
          matches.map(p => <ProviderRow key={p.id} onSelect={select} provider={p} />)
        )}
      </section>
    )
  }

  const featured = ordered.find(p => p.id === FEATURED_ID) ?? null
  const rest = featured ? ordered.filter(p => p.id !== FEATURED_ID) : ordered
  // Keep connected accounts grouped and always visible; only the unconnected
  // providers hide behind the disclosure, so the page leads with what's set up.
  const connected = rest.filter(p => p.status?.logged_in)
  const others = rest.filter(p => !p.status?.logged_in)
  const collapsible = others.length > 0
  const showOthers = !collapsible || showAll

  return (
    <section className="mb-5 grid gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <SettingsCategoryHeading icon={Sparkles} title="Connect an account" />
        <button
          className="text-[length:var(--conversation-caption-font-size)] font-medium text-muted-foreground transition hover:text-foreground"
          onClick={onWantApiKey}
          type="button"
        >
          Have an API key instead?
        </button>
      </div>
      <p className="-mt-2 mb-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        Sign in with a subscription — no API key to copy. Hermes runs the browser sign-in for you, right here in the
        app.
      </p>
      {featured && <FeaturedProviderRow onSelect={select} provider={featured} />}
      {connected.length > 0 && (
        <>
          <p className="mt-1 px-0.5 text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-tertiary)">
            Connected
          </p>
          {connected.map(p => (
            <ProviderRow key={p.id} onSelect={select} provider={p} />
          ))}
        </>
      )}
      {showOthers && (
        <>
          {others.map(p => (
            <ProviderRow key={p.id} onSelect={select} provider={p} />
          ))}
          <KeyProviderRow onClick={onWantApiKey} />
        </>
      )}
      {collapsible && (
        <button
          className="flex items-center justify-center gap-1.5 pt-1 text-[length:var(--conversation-caption-font-size)] font-medium text-muted-foreground transition hover:text-foreground"
          onClick={() => setShowAll(v => !v)}
          type="button"
        >
          {showAll ? 'Collapse' : connected.length > 0 ? 'Connect another provider' : 'Other providers'}
          <ChevronDown className={cn('size-3.5 transition', showAll && 'rotate-180')} />
        </button>
      )}
    </section>
  )
}

function EmptyMatches({ query }: { query: string }) {
  return (
    <div className="rounded-lg border border-dashed border-(--ui-stroke-tertiary) px-4 py-8 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
      No providers match “{query.trim()}”.
    </div>
  )
}

export function ProvidersSettings({ onViewChange, query, view }: ProvidersSettingsProps) {
  const { rowProps, saveValue, vars } = useEnvCredentials()
  const [oauthProviders, setOauthProviders] = useState<OAuthProvider[]>([])
  // The onboarding overlay owns the OAuth flow. Watch its `manual` flag so we
  // re-read connection state when the user finishes (or dismisses) a sign-in
  // they launched from this page — otherwise the cards keep their stale status.
  const onboardingActive = useStore($desktopOnboarding).manual

  useEffect(() => {
    if (onboardingActive) {
      return
    }

    let cancelled = false

    // OAuth providers are best-effort — a failure here just hides the panel.
    void (async () => {
      try {
        const { providers } = await listOAuthProviders()

        if (!cancelled) {
          setOauthProviders(providers)
        }
      } catch {
        // Ignore — the OAuth panel just won't render.
      }
    })()

    return () => void (cancelled = true)
  }, [onboardingActive])

  if (!vars) {
    return <LoadingState label="Loading providers..." />
  }

  const q = query.trim().toLowerCase()
  const hasOauth = oauthProviders.length > 0
  // The sidebar subnav owns the Accounts/API-keys split now; with no OAuth
  // providers there's nothing for the "Accounts" view to show, so fall to keys.
  const showApiKeyForm = view === 'keys' || !hasOauth

  // Search stays inside the active sub-view and uses its native card format —
  // the OAuth picker filters its rows, the key catalog filters its grid — so
  // results never drop back to a different-looking accordion list.
  const allKeyOptions = buildApiKeyOptions(vars)
  const keyOptions = q
    ? allKeyOptions.filter(o => o.name.toLowerCase().includes(q) || o.envKey.toLowerCase().includes(q))
    : allKeyOptions

  if (showApiKeyForm) {
    return (
      <SettingsContent>
        {q && (
          <SettingsCategoryHeading
            count={`${keyOptions.length} match${keyOptions.length === 1 ? '' : 'es'}`}
            icon={Sparkles}
            title="API keys"
          />
        )}
        {keyOptions.length > 0 ? (
          <ApiKeyForm
            canGoBack={hasOauth && !q}
            collapseAfter={q ? undefined : 6}
            isSet={envKey => Boolean(vars[envKey]?.is_set)}
            onBack={() => onViewChange('accounts')}
            onClear={rowProps.onClear}
            onSave={(envKey, value) => saveValue(envKey, value)}
            options={keyOptions}
            redactedValue={envKey => vars[envKey]?.redacted_value}
          />
        ) : (
          <EmptyMatches query={query} />
        )}
      </SettingsContent>
    )
  }

  return (
    <SettingsContent>
      <OAuthPicker onWantApiKey={() => onViewChange('keys')} providers={oauthProviders} query={query} />
    </SettingsContent>
  )
}

interface ProvidersSettingsProps extends SearchProps {
  onViewChange: (view: ProviderView) => void
  view: ProviderView
}
