import { useStore } from '@nanostores/react'
import { type ChangeEvent, type KeyboardEvent, useEffect, useMemo, useState } from 'react'

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
import { ChevronDown, ExternalLink, KeyRound, Loader2, Save } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $desktopOnboarding, startManualProviderOAuth } from '@/store/onboarding'
import type { EnvVarInfo, OAuthProvider } from '@/types/hermes'

import { SettingsCategoryHeading, useEnvCredentials } from './env-credentials'
import { providerGroup, providerMeta, providerPriority, withoutKey } from './helpers'
import { LoadingState, SettingsContent } from './primitives'
import type { EnvRowProps } from './types'

// Sub-views surfaced as a sidebar subnav: account sign-in vs raw API keys.
export const PROVIDER_VIEWS = ['accounts', 'keys'] as const

export type ProviderView = (typeof PROVIDER_VIEWS)[number]

const isKeyVar = (key: string, info: EnvVarInfo) => info.is_password || /(?:_API_KEY|_TOKEN|_KEY)$/.test(key)

const friendlyFieldLabel = (key: string, info: EnvVarInfo) =>
  info.description?.trim() || key.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase())

// Advanced (non-primary) fields are mostly base-URL / endpoint overrides, not
// keys — so don't reuse the "Paste key" placeholder that makes them read as a
// duplicate key input. URL-ish vars get a URL hint; everything else stays optional.
const advancedPlaceholder = (key: string, info: EnvVarInfo): string =>
  isKeyVar(key, info) ? 'Paste key' : /URL$/i.test(key) ? 'https://…' : 'Optional'

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

// A single credential field: a set key shows as a filled read-only input
// (redacted value) that edits in place on click. Save appears once typed; a set
// key also offers Remove, and Esc cancels without closing the overlay.
function KeyField({
  compact = false,
  info,
  label,
  placeholder,
  rowProps,
  varKey
}: {
  compact?: boolean
  info: EnvVarInfo
  label?: string
  placeholder?: string
  rowProps: KeyRowProps
  varKey: string
}) {
  const { edits, onClear, onSave, saving, setEdits } = rowProps
  const editing = edits[varKey] !== undefined
  const draft = edits[varKey] ?? ''
  const dirty = draft.trim().length > 0
  const busy = saving === varKey
  const masked = info.redacted_value ?? '••••••••'
  const startEdit = () => setEdits(c => ({ ...c, [varKey]: '' }))
  const cancel = () => setEdits(c => withoutKey(c, varKey))
  const update = (e: ChangeEvent<HTMLInputElement>) => setEdits(c => ({ ...c, [varKey]: e.target.value }))

  // Enter saves; Esc cancels in place without bubbling to the overlay's window
  // Escape listener (which would otherwise close the whole settings panel).
  const keydown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && dirty) {
      void onSave(varKey)
    } else if (e.key === 'Escape' && editing) {
      e.preventDefault()
      e.stopPropagation()
      cancel()
    }
  }

  // Advanced overrides render quieter (xs) than the primary key field so the key
  // stays the visual anchor. Padding-driven sizing — no fixed heights.
  const inputSize = compact ? 'xs' : 'sm'
  const editType = info.is_password ? 'password' : 'text'

  // A set value reads as a single filled, read-only field (showing the redacted
  // value). Clicking it drops into edit mode in place — no Replace/Cancel chrome.
  const control =
    info.is_set && !editing ? (
      <Input
        className="cursor-pointer font-mono text-muted-foreground"
        onFocus={startEdit}
        readOnly
        size={inputSize}
        value={masked}
      />
    ) : (
      <div className="grid gap-1">
        <div className="flex items-center gap-2">
          <Input
            autoFocus={editing}
            className="min-w-0 flex-1 font-mono"
            onChange={update}
            onKeyDown={keydown}
            placeholder={placeholder ?? 'Paste key'}
            size={inputSize}
            type={editType}
            value={draft}
          />
          {dirty && (
            <Button disabled={busy} onClick={() => void onSave(varKey)} size="sm">
              {busy ? <Loader2 className="size-4 animate-spin" /> : <Save />}
              {busy ? 'Saving' : 'Save'}
            </Button>
          )}
        </div>
        {editing && (
          <div className="flex items-center gap-1 text-[0.6875rem]">
            {info.is_set && (
              <>
                <Button
                  className="h-auto px-0 py-0 text-[0.6875rem] text-destructive hover:text-destructive"
                  disabled={busy}
                  onClick={() => void onClear(varKey)}
                  type="button"
                  variant="text"
                >
                  Remove
                </Button>
                <span className="text-muted-foreground">or</span>
              </>
            )}
            <span className="text-muted-foreground">esc to cancel</span>
          </div>
        )}
      </div>
    )

  // Standard stacked form field: small muted label above, input below. Same shape
  // for the primary key and every advanced override — just smaller when compact.
  // Empty advanced inputs (not labels) fade back, brightening on hover/focus/set.
  const dim = compact && !info.is_set

  return (
    <div className="grid gap-1.5">
      {label && (
        <label className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {label}
        </label>
      )}
      {dim ? (
        <div className="opacity-55 transition-opacity focus-within:opacity-100 hover:opacity-100">{control}</div>
      ) : (
        control
      )}
    </div>
  )
}

function ProviderKeyCard({
  expanded,
  group,
  onExpand,
  onToggle,
  rowProps
}: {
  expanded: boolean
  group: ProviderKeyGroup
  onExpand: () => void
  onToggle: () => void
  rowProps: KeyRowProps
}) {
  // Expandable when there's anything to reveal — advanced overrides and/or a
  // "Get a key" docs link (which lives at the bottom of the expanded panel).
  const expandable = group.advanced.length > 0 || Boolean(group.docsUrl)

  return (
    <div
      className={cn(
        'group/card rounded-[6px] px-2 py-2 transition-colors',
        expandable && 'cursor-pointer',
        expandable && !expanded && 'hover:bg-(--ui-row-hover-background)',
        expanded && 'bg-(--ui-bg-quaternary) ring-1 ring-(--ui-stroke-secondary)'
      )}
      onClick={expandable ? onToggle : undefined}
      onKeyDown={
        expandable
          ? e => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onToggle()
              }
            }
          : undefined
      }
      role={expandable ? 'button' : undefined}
      tabIndex={expandable ? 0 : undefined}
    >
      <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
        <div className="flex min-w-44 flex-1 items-center gap-2 py-1">
          <span
            className={cn('size-2 shrink-0 rounded-full', group.hasAnySet ? 'bg-primary' : 'bg-(--ui-stroke-secondary)')}
          />
          <span className="truncate text-[length:var(--conversation-text-font-size)] font-medium">{group.name}</span>
          {expandable && (
            <ChevronDown
              className={cn(
                'size-3.5 shrink-0 text-muted-foreground transition',
                expanded ? 'rotate-180 opacity-100' : 'opacity-0 group-hover/card:opacity-100'
              )}
            />
          )}
        </div>
        <div
          className="w-full sm:w-80 sm:shrink-0"
          onClick={e => e.stopPropagation()}
          onFocus={() => {
            if (expandable && !expanded) {
              onExpand()
            }
          }}
        >
          <KeyField
            info={group.primary[1]}
            placeholder={`Paste ${group.name} key`}
            rowProps={rowProps}
            varKey={group.primary[0]}
          />
        </div>
      </div>
      {expandable && expanded && (
        <div className="mt-3 grid gap-2.5 pl-4" onClick={e => e.stopPropagation()}>
          {group.advanced.map(([key, info]) => (
            <KeyField
              compact
              info={info}
              key={key}
              label={isKeyVar(key, info) ? key : friendlyFieldLabel(key, info)}
              placeholder={advancedPlaceholder(key, info)}
              rowProps={rowProps}
              varKey={key}
            />
          ))}
          {group.docsUrl && (
            <a
              className="inline-flex w-fit items-center gap-1 justify-self-end text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary) underline-offset-4 transition-colors hover:text-foreground hover:underline"
              href={group.docsUrl}
              onClick={e => e.stopPropagation()}
              rel="noreferrer"
              target="_blank"
            >
              Get a key
              <ExternalLink className="size-3" />
            </a>
          )}
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
        <SettingsCategoryHeading icon={KeyRound} title="Connect an account" />
        <Button
          className="h-auto px-0 py-0 text-[length:var(--conversation-caption-font-size)]"
          onClick={onWantApiKey}
          type="button"
          variant="textStrong"
        >
          Have an API key instead?
        </Button>
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
        <Button
          className="h-auto px-0 py-1 text-[length:var(--conversation-caption-font-size)]"
          onClick={() => setShowAll(v => !v)}
          type="button"
          variant="text"
        >
          {showAll ? 'Collapse' : connected.length > 0 ? 'Connect another provider' : 'Other providers'}
          <ChevronDown className={cn('size-3.5 transition', showAll && 'rotate-180')} />
        </Button>
      )}
    </section>
  )
}

function NoProviderKeys() {
  return (
    <div className="grid min-h-32 place-items-center px-4 py-8 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
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
                onExpand={() => setOpenProvider(group.name)}
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
