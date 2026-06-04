import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import {
  FEATURED_ID,
  FeaturedProviderRow,
  KeyProviderRow,
  ProviderRow,
  sortProviders
} from '@/components/desktop-onboarding-overlay'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { listOAuthProviders } from '@/hermes'
import { ChevronDown, ExternalLink, Loader2, Save, Sparkles, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $desktopOnboarding, startManualProviderOAuth } from '@/store/onboarding'
import type { EnvVarInfo, OAuthProvider } from '@/types/hermes'

import { SettingsCategoryHeading, useEnvCredentials } from './env-credentials'
import { providerGroup, providerMeta, providerPriority } from './helpers'
import { LoadingState, SettingsContent } from './primitives'
import type { EnvRowProps } from './types'

// Sub-views surfaced as a sidebar subnav: account sign-in vs raw API keys.
export const PROVIDER_VIEWS = ['accounts', 'keys'] as const

export type ProviderView = (typeof PROVIDER_VIEWS)[number]

const isKeyVar = (key: string, info: EnvVarInfo) => info.is_password || /(?:_API_KEY|_TOKEN|_KEY)$/.test(key)

const friendlyFieldLabel = (key: string, info: EnvVarInfo) =>
  info.description?.trim() || key.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase())

// Group the env catalog by provider so the keys view can render one collapsible
// row per vendor: a primary key field inline, with any secondary / advanced vars
// (base URL overrides, alt tokens) revealed when the row is focused/expanded.
// Mirrors what Cursor's API-keys section does. Groups without a key field (e.g.
// Nous Portal's lone base-URL override) and the "Other" bucket are skipped.
function buildProviderKeyGroups(vars: Record<string, EnvVarInfo>): ProviderKeyGroup[] {
  const buckets = new Map<string, [string, EnvVarInfo][]>()

  for (const [key, info] of Object.entries(vars)) {
    if (info.category !== 'provider') {
      continue
    }

    const name = providerGroup(key)

    if (name === 'Other') {
      continue
    }

    buckets.set(name, [...(buckets.get(name) ?? []), [key, info]])
  }

  const groups: ProviderKeyGroup[] = []

  for (const [name, entries] of buckets) {
    const primary = entries.find(([k, i]) => !i.advanced && isKeyVar(k, i)) ?? entries.find(([k, i]) => isKeyVar(k, i))

    if (!primary) {
      continue
    }

    const meta = providerMeta(name)

    groups.push({
      // Advanced = the provider's non-key knobs (base URL, region, deployment).
      // Skip redundant alias key vars (e.g. ANTHROPIC_TOKEN vs ANTHROPIC_API_KEY)
      // so we never render a second "Paste key" input — unless one is already
      // set, in which case keep it visible so it stays clearable.
      advanced: entries
        .filter(([k, i]) => k !== primary[0] && (!isKeyVar(k, i) || i.is_set))
        .sort(([a], [b]) => a.localeCompare(b)),
      description: meta?.description ?? primary[1].description,
      docsUrl: meta?.docsUrl ?? primary[1].url ?? undefined,
      hasAnySet: entries.some(([, i]) => i.is_set),
      name,
      primary,
      priority: providerPriority(name)
    })
  }

  return groups.sort((a, b) => a.priority - b.priority || a.name.localeCompare(b.name))
}

// A single inline credential field: an always-visible input (Cursor-style)
// that shows the redacted current value as its placeholder so a set key reads
// as "•••• / 1234…wxyz" without an extra reveal click. Save appears once typed;
// otherwise a set key offers Remove.
function KeyField({
  info,
  label,
  placeholder,
  rowProps,
  varKey
}: {
  info: EnvVarInfo
  label?: string
  placeholder?: string
  rowProps: KeyRowProps
  varKey: string
}) {
  const { edits, onClear, onSave, saving, setEdits } = rowProps
  const draft = edits[varKey] ?? ''
  const dirty = draft.trim().length > 0
  const busy = saving === varKey

  return (
    <div className="grid gap-1">
      {label && (
        <label className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{label}</label>
      )}
      <div className="flex items-center gap-2">
        <Input
          className="h-8 min-w-0 flex-1 font-mono text-[0.6875rem] md:text-[0.6875rem]"
          onChange={e => setEdits(c => ({ ...c, [varKey]: e.target.value }))}
          onKeyDown={e => {
            if (e.key === 'Enter' && dirty) {
              void onSave(varKey)
            }
          }}
          placeholder={info.is_set ? (info.redacted_value ?? '••••••••') : (placeholder ?? 'Paste key')}
          type={info.is_password ? 'password' : 'text'}
          value={draft}
        />
        {dirty ? (
          <Button disabled={busy} onClick={() => void onSave(varKey)} size="sm">
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Save />}
            {busy ? 'Saving' : 'Save'}
          </Button>
        ) : info.is_set ? (
          <Button disabled={busy} onClick={() => void onClear(varKey)} size="icon-xs" title="Remove key" variant="ghost">
            <Trash2 />
          </Button>
        ) : null}
      </div>
    </div>
  )
}

function ProviderKeyCard({
  expanded,
  group,
  onToggle,
  rowProps
}: {
  expanded: boolean
  group: ProviderKeyGroup
  onToggle: () => void
  rowProps: KeyRowProps
}) {
  const hasOptions = group.advanced.length > 0

  return (
    <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/20">
      <div className="flex flex-wrap items-start gap-x-4 gap-y-2 p-3">
        <div className="flex min-w-44 flex-1 flex-col gap-0.5">
          <button
            className={cn('flex items-center gap-2 text-left', hasOptions ? 'cursor-pointer' : 'cursor-default')}
            onClick={() => hasOptions && onToggle()}
            type="button"
          >
            <span
              className={cn(
                'size-2 shrink-0 rounded-full',
                group.hasAnySet ? 'bg-primary' : 'bg-(--ui-stroke-secondary)'
              )}
            />
            <span className="truncate text-[length:var(--conversation-text-font-size)] font-medium">{group.name}</span>
            {hasOptions && (
              <ChevronDown
                className={cn('size-3.5 shrink-0 text-muted-foreground transition', expanded && 'rotate-180')}
              />
            )}
          </button>
          {(group.description || group.docsUrl) && (
            <span className="pl-4 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {group.description}
              {group.docsUrl && (
                <>
                  {group.description ? ' · ' : ''}
                  <a
                    className="inline-flex items-center gap-0.5 transition hover:text-foreground"
                    href={group.docsUrl}
                    rel="noreferrer"
                    target="_blank"
                  >
                    Get a key
                    <ExternalLink className="size-3" />
                  </a>
                </>
              )}
            </span>
          )}
        </div>
        <div className="w-full sm:w-80 sm:shrink-0">
          <KeyField
            info={group.primary[1]}
            placeholder={`Paste ${group.name} key`}
            rowProps={rowProps}
            varKey={group.primary[0]}
          />
        </div>
      </div>
      {hasOptions && expanded && (
        <div className="grid gap-3 border-t border-(--ui-stroke-tertiary) p-3">
          {group.advanced.map(([key, info]) => (
            <KeyField
              info={info}
              key={key}
              label={isKeyVar(key, info) ? key : friendlyFieldLabel(key, info)}
              rowProps={rowProps}
              varKey={key}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// Deliberately a near-1:1 replica of the first-run onboarding picker
// (`Picker` in desktop-onboarding-overlay): same recommended card, same
// provider rows, same "Other providers" disclosure, same OpenRouter quick-key
// row, and the same bottom-right "I have an API key" affordance. The leaf cards
// are the exact shared components, so the two surfaces stay visually identical.
// Selecting a provider hands off to the shared onboarding overlay, which runs
// that provider's real sign-in flow; the key affordances open the API-key
// catalog below.
function OAuthPicker({ onWantApiKey, providers }: { onWantApiKey: () => void; providers: OAuthProvider[] }) {
  const [showAll, setShowAll] = useState(false)
  const ordered = useMemo(() => sortProviders(providers), [providers])

  if (ordered.length === 0) {
    return null
  }

  const select = (p: OAuthProvider) => startManualProviderOAuth(p.id)

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

function NoProviderKeys() {
  return (
    <div className="rounded-lg border border-dashed border-(--ui-stroke-tertiary) px-4 py-8 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
      No provider API keys available.
    </div>
  )
}

export function ProvidersSettings({ onViewChange, view }: ProvidersSettingsProps) {
  const { rowProps, vars } = useEnvCredentials()
  const [oauthProviders, setOauthProviders] = useState<OAuthProvider[]>([])
  // Single-open accordion for the per-provider "advanced options" panels.
  const [openProvider, setOpenProvider] = useState<null | string>(null)
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

  const hasOauth = oauthProviders.length > 0
  // The sidebar subnav owns the Accounts/API-keys split now; with no OAuth
  // providers there's nothing for the "Accounts" view to show, so fall to keys.
  const showApiKeys = view === 'keys' || !hasOauth

  const keyGroups = buildProviderKeyGroups(vars)

  if (showApiKeys) {
    return (
      <SettingsContent>
        {keyGroups.length > 0 ? (
          <div className="grid gap-2">
            {keyGroups.map(group => (
              <ProviderKeyCard
                expanded={openProvider === group.name}
                group={group}
                key={group.name}
                onToggle={() => setOpenProvider(prev => (prev === group.name ? null : group.name))}
                rowProps={rowProps}
              />
            ))}
          </div>
        ) : (
          <NoProviderKeys />
        )}
      </SettingsContent>
    )
  }

  return (
    <SettingsContent>
      <OAuthPicker onWantApiKey={() => onViewChange('keys')} providers={oauthProviders} />
    </SettingsContent>
  )
}

type KeyRowProps = Omit<EnvRowProps, 'info' | 'varKey'>

interface ProviderKeyGroup {
  advanced: [string, EnvVarInfo][]
  description?: string
  docsUrl?: string
  hasAnySet: boolean
  name: string
  primary: [string, EnvVarInfo]
  priority: number
}

interface ProvidersSettingsProps {
  onViewChange: (view: ProviderView) => void
  view: ProviderView
}
