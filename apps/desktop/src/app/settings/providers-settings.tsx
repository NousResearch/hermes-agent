import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { runInTerminal } from '@/app/right-sidebar/store'
import {
  FEATURED_ID,
  FeaturedProviderRow,
  FireworksProviderRow,
  LocalModelsProviderRow,
  OpenRouterProviderRow,
  ProviderRow,
  providerTitle
} from '@/components/onboarding'
import { Button } from '@/components/ui/button'
import { RowButton } from '@/components/ui/row-button'
import { SearchField } from '@/components/ui/search-field'
import { Tip } from '@/components/ui/tooltip'
import { disconnectOAuthProvider, listOAuthProviders } from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, ChevronDown, ChevronRight, Loader2, Terminal, Users, Trash2 } from '@/lib/icons'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { confirm } from '@/store/confirm'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { notify, notifyError } from '@/store/notifications'
import { $desktopOnboarding, startManualLocalEndpoint, startManualProviderOAuth } from '@/store/onboarding'
import { $settingsRequestProfile } from '@/store/settings-scope'
import type { EnvVarInfo, OAuthProvider } from '@/types/hermes'

import { isKeyVar, ProviderKeyRows } from './credential-key-ui'
import { CustomEndpointsSettings } from './custom-endpoints-settings'
import { SettingsCategoryHeading, useEnvCredentials } from './env-credentials'
import { providerGroup, providerMeta, providerPriority } from './helpers'
import { LocalModelsSettings } from './local-models-settings'
import { SettingsContent, SettingsSkeleton } from './primitives'
import { SettingsProfileScope } from './profile-scope'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

const canRunInTerminal = () => typeof window !== 'undefined' && Boolean(window.hermesDesktop?.terminal)

function GroupLabel({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 px-0.5 text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-tertiary)">
      {children}
    </p>
  )
}

export const PROVIDER_VIEWS = ['accounts', 'keys', 'custom-endpoints', 'local'] as const

export type ProviderView = (typeof PROVIDER_VIEWS)[number]

const providerKeyElementId = (name: string) => `provider-key-${name.replace(/\W+/g, '-')}`

function buildProviderKeyGroups(vars: Record<string, EnvVarInfo>): ProviderKeyGroup[] {
  const buckets = new Map<string, [string, EnvVarInfo][]>()

  for (const [key, info] of Object.entries(vars)) {
    if (info.category !== 'provider') {
      continue
    }

    const scopedInfos = info.provider_profiles?.length
      ? info.provider_profiles.map(profile => ({
          ...info,
          description: profile.description || info.description,
          provider: profile.provider,
          provider_label: profile.provider_label,
          provider_primary: profile.primary,
          url: profile.url ?? info.url
        }))
      : [info]

    for (const scopedInfo of scopedInfos) {
      const name = scopedInfo.provider_label?.trim() || scopedInfo.provider?.trim() || providerGroup(key)

      if (name === 'Other') {
        continue
      }

      buckets.set(name, [...(buckets.get(name) ?? []), [key, scopedInfo]])
    }
  }

  const groups: ProviderKeyGroup[] = []

  for (const [name, entries] of buckets) {
    const primary =
      entries.find(([k, i]) => i.provider_primary && isKeyVar(k, i)) ??
      entries.find(([k, i]) => !i.advanced && isKeyVar(k, i)) ??
      entries.find(([k, i]) => isKeyVar(k, i))

    if (!primary) {
      continue
    }

    const meta = providerMeta(name)

    groups.push({
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

function OAuthPicker({
  disconnecting,
  onDisconnect,
  onTerminalDisconnect,
  onWantApiKey,
  onWantLocalModels,
  providers,
  accountQuery,
  onAccountQueryChange,
  profile
}: {
  disconnecting: null | string
  onDisconnect: (provider: OAuthProvider) => void
  onTerminalDisconnect: (provider: OAuthProvider) => void
  onWantApiKey: () => void
  onWantLocalModels: () => void
  providers: OAuthProvider[]
  accountQuery: string
  onAccountQueryChange: (query: string) => void
  profile?: string
}) {
  const { t } = useI18n()
  const p = t.settings.providers
  const [showAll, setShowAll] = useState(false)
  const ordered = useMemo(() => {
    const query = normalize(accountQuery)
    return providers
      .filter(provider => !query || normalize(`${providerTitle(provider)} ${provider.id}`).includes(query))
      .sort((a, b) => providerTitle(a).localeCompare(providerTitle(b), undefined, { sensitivity: 'base' }))
  }, [accountQuery, providers])

  if (providers.length === 0) {
    return null
  }

  const select = (provider: OAuthProvider) => startManualProviderOAuth(provider.id, profile)
  const isConnected = (provider: OAuthProvider) =>
    Boolean(provider.status?.logged_in) && provider.status?.free_tier !== true
  const featured = ordered.find(provider => provider.id === FEATURED_ID && !isConnected(provider)) ?? null
  const rest = featured ? ordered.filter(provider => provider.id !== FEATURED_ID) : ordered
  const connected = rest.filter(isConnected)
  const others = rest.filter(provider => !isConnected(provider))
  const collapsible = others.length > 0
  const showOthers = !collapsible || showAll || Boolean(accountQuery)

  return (
    <section className="mb-5 grid gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <SettingsCategoryHeading icon={Users} title={p.connectAccount} />
        <Button
          className="text-[length:var(--conversation-caption-font-size)]"
          onClick={onWantApiKey}
          size="inline"
          type="button"
          variant="textStrong"
        >
          {p.haveApiKey}
        </Button>
      </div>
      <p className="-mt-2 mb-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {p.intro}
      </p>
      <SearchField
        aria-label={p.searchKeys}
        containerClassName="w-full"
        onChange={onAccountQueryChange}
        placeholder={p.searchKeys}
        value={accountQuery}
      />
      {ordered.length === 0 ? (
        <div className="grid min-h-24 place-items-center px-4 py-6 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
          {p.noKeysMatch}
        </div>
      ) : (
        <>
          {featured && <FeaturedProviderRow onSelect={select} provider={featured} />}
          {!accountQuery && $localModelsEnabled.get() && <LocalModelsProviderRow onClick={onWantLocalModels} />}
          {connected.length > 0 && (
            <>
              <GroupLabel>{p.connected}</GroupLabel>
              {connected.map(provider => (
                <ConnectedProviderRow
                  disconnecting={disconnecting === provider.id}
                  key={provider.id}
                  onDisconnect={onDisconnect}
                  onSelect={select}
                  onTerminalDisconnect={onTerminalDisconnect}
                  provider={provider}
                />
              ))}
            </>
          )}
          {showOthers && (
            <>
              {connected.length > 0 && <GroupLabel>{p.otherProviders}</GroupLabel>}
              {others.map(provider => (
                <ProviderRow key={provider.id} onSelect={select} provider={provider} />
              ))}
              {!accountQuery && <FireworksProviderRow onClick={onWantApiKey} />}
              {!accountQuery && <OpenRouterProviderRow onClick={onWantApiKey} />}
            </>
          )}
          {collapsible && !accountQuery && (
            <Button
              className="py-1 text-[length:var(--conversation-caption-font-size)]"
              onClick={() => setShowAll(value => !value)}
              size="inline"
              type="button"
              variant="text"
            >
              {showAll ? p.collapse : connected.length > 0 ? p.connectAnother : p.otherProviders}
              <ChevronDown className={cn('size-3.5 transition', showAll && 'rotate-180')} />
            </Button>
          )}
        </>
      )}
    </section>
  )
}

function ConnectedProviderRow({
  disconnecting,
  onDisconnect,
  onSelect,
  onTerminalDisconnect,
  provider
}: {
  disconnecting: boolean
  onDisconnect: (provider: OAuthProvider) => void
  onSelect: (provider: OAuthProvider) => void
  onTerminalDisconnect: (provider: OAuthProvider) => void
  provider: OAuthProvider
}) {
  const { t } = useI18n()
  const copy = t.settings.providers
  const title = providerTitle(provider)
  const Trail = provider.flow === 'external' ? Terminal : ChevronRight
  const canDisconnect = provider.disconnectable ?? provider.flow !== 'external'
  const terminalDisconnect = !canDisconnect && Boolean(provider.disconnect_command) && canRunInTerminal()
  const showHint = !canDisconnect && !terminalDisconnect

  return (
    <div className="group grid grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-[6px] transition-colors hover:bg-(--ui-control-hover-background)">
      <RowButton
        className="min-w-0 px-3 py-2.5 text-left"
        onClick={() => (terminalDisconnect ? onTerminalDisconnect(provider) : onSelect(provider))}
      >
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-[length:var(--conversation-text-font-size)] font-semibold">{title}</span>
          <span className="inline-flex shrink-0 items-center gap-1 bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
            <Check className="size-3" />
            {copy.connected}
          </span>
        </div>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">{t.onboarding.flowSubtitles[provider.flow]}</p>
        {showHint && (
          <p className="mt-0.5 truncate text-[0.68rem] leading-5 text-muted-foreground/70">
            {provider.flow === 'external' ? copy.removeExternalGeneric(title) : copy.removeKeyManaged(title)}
          </p>
        )}
      </RowButton>
      <div className="flex items-center gap-1 pr-2">
        {terminalDisconnect ? (
          <Button
            aria-label={`${copy.disconnect} ${title} in terminal`}
            onClick={() => onTerminalDisconnect(provider)}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Terminal className="size-4" />
          </Button>
        ) : (
          <Trail className="size-4 text-muted-foreground transition group-hover:text-foreground" />
        )}
        {canDisconnect && (
          <Button
            aria-label={`${t.common.remove} ${title}`}
            disabled={disconnecting}
            onClick={() => onDisconnect(provider)}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            {disconnecting ? <Loader2 className="size-3 animate-spin" /> : <Trash2 className="size-3" />}
          </Button>
        )}
        {terminalDisconnect && (
          <Tip label={copy.disconnectInTerminal}>
            <Button
              aria-label={`${copy.disconnect} ${title}`}
              onClick={() => onTerminalDisconnect(provider)}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Trash2 className="size-3" />
            </Button>
          </Tip>
        )}
      </div>
    </div>
  )
}

function NoProviderKeys() {
  const { t } = useI18n()

  return (
    <div className="grid min-h-32 place-items-center px-4 py-8 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
      {t.settings.providers.noProviderKeys}
    </div>
  )
}

function LocalEndpointRow({ onOpen }: { onOpen: (reason: null | string) => void }) {
  const { t } = useI18n()
  const copy = t.settings.providers.localEndpoint

  return (
    <RowButton
      className="group grid grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-[6px] px-3 py-2.5 text-left transition-colors hover:bg-(--ui-control-hover-background)"
      onClick={() => onOpen(null)}
    >
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="truncate text-[length:var(--conversation-text-font-size)] font-semibold">{copy.title}</span>
        <span className="truncate text-[length:var(--conversation-caption-font-size)] leading-5 text-muted-foreground">
          {copy.description}
        </span>
      </div>
      <ChevronRight className="size-4 text-muted-foreground transition group-hover:text-foreground" />
    </RowButton>
  )
}

export function ProvidersSettings({
  onClose,
  onConfigSaved,
  onMainModelChanged,
  onViewChange,
  view
}: ProvidersSettingsProps) {
  const { t } = useI18n()
  const scopeProfile = useStore($settingsRequestProfile)
  const { rowProps, vars } = useEnvCredentials(scopeProfile)
  const [oauthProviders, setOauthProviders] = useState<OAuthProvider[]>([])
  const [openProvider, setOpenProvider] = useState<null | string>(null)
  const [disconnecting, setDisconnecting] = useState<null | string>(null)
  const [keyQuery, setKeyQuery] = useState('')
  const [accountQuery, setAccountQuery] = useState('')
  const onboardingActive = useStore($desktopOnboarding).manual

  const keyGroupByEnv = useMemo(() => {
    const byEnv = new Map<string, string>()

    for (const group of vars ? buildProviderKeyGroups(vars) : []) {
      for (const [key] of [group.primary, ...group.advanced]) {
        byEnv.set(key, group.name)
      }
    }

    return byEnv
  }, [vars])

  const apiKeysShown = view === 'keys' || (oauthProviders.length === 0 && view !== 'custom-endpoints')

  useDeepLinkHighlight({
    elementId: key => providerKeyElementId(keyGroupByEnv.get(key) ?? ''),
    onResolve: key => {
      setKeyQuery('')
      setOpenProvider(keyGroupByEnv.get(key) ?? null)
    },
    param: 'key',
    ready: key => apiKeysShown && keyGroupByEnv.has(key)
  })

  const refreshOAuthProviders = useCallback(async () => {
    const { providers } = await listOAuthProviders(scopeProfile)
    setOauthProviders(providers)
  }, [scopeProfile])

  useEffect(() => {
    let cancelled = false

    void (async () => {
      if (onboardingActive) {
        return
      }

      try {
        const { providers } = await listOAuthProviders(scopeProfile)

        if (!cancelled) {
          setOauthProviders(providers)
        }
      } catch {
        // Ignore — the OAuth panel just won't render.
      }
    })()

    return () => void (cancelled = true)
  }, [onboardingActive, scopeProfile])

  async function handleTerminalDisconnect(provider: OAuthProvider) {
    const command = provider.disconnect_command

    if (!command) {
      return
    }

    const name = providerTitle(provider)
    const ok = await confirm({
      confirmLabel: t.settings.providers.disconnect,
      destructive: true,
      title: t.settings.providers.removeTerminalConfirm(name, command)
    })

    if (!ok) {
      return
    }

    onClose()
    runInTerminal(command)
    notify({
      kind: 'info',
      title: t.settings.providers.disconnect,
      message: t.settings.providers.removeTerminalRunning(name)
    })
  }

  async function handleDisconnect(provider: OAuthProvider) {
    const name = providerTitle(provider)
    const ok = await confirm({
      confirmLabel: t.settings.providers.disconnect,
      destructive: true,
      title: t.settings.providers.removeConfirm(name)
    })

    if (!ok) {
      return
    }

    setDisconnecting(provider.id)

    try {
      const result = await disconnectOAuthProvider(provider.id, scopeProfile)

      if (!result?.ok) {
        notifyError(new Error('No stored credentials were removed'), t.settings.providers.failedRemove(name))
        return
      }

      notify({
        durationMs: 3_000,
        kind: 'success',
        title: t.settings.providers.removedTitle,
        message: t.settings.providers.removedMessage(name)
      })
      await refreshOAuthProviders().catch(() => undefined)
    } catch (err) {
      notifyError(err, t.settings.providers.failedRemove(name))
    } finally {
      setDisconnecting(null)
    }
  }

  if (!vars) {
    return <SettingsSkeleton search sections={[{ rows: 6 }]} />
  }

  const hasOauth = oauthProviders.length > 0
  const showApiKeys = view === 'keys' || (!hasOauth && view !== 'custom-endpoints')
  const keyGroups = buildProviderKeyGroups(vars)

  if (showApiKeys) {
    const query = normalize(keyQuery)
    const visibleGroups = query
      ? keyGroups.filter(group => {
          const haystack = [group.name, group.description ?? '', group.primary[0], ...group.advanced.map(([key]) => key)]
          return haystack.some(value => value.toLowerCase().includes(query))
        })
      : keyGroups

    return (
      <SettingsContent>
        <SettingsProfileScope className="mb-5" />
        <LocalEndpointRow onOpen={reason => startManualLocalEndpoint(reason, scopeProfile)} />
        {keyGroups.length > 0 ? (
          <div className="grid gap-3">
            <SearchField
              aria-label={t.settings.providers.searchKeys}
              containerClassName="w-full"
              onChange={setKeyQuery}
              placeholder={t.settings.providers.searchKeys}
              value={keyQuery}
            />
            {visibleGroups.length > 0 ? (
              <div className="grid gap-2">
                {visibleGroups.map(group => (
                  <div className="scroll-mt-6 rounded-[6px]" id={providerKeyElementId(group.name)} key={group.name}>
                    <ProviderKeyRows
                      expanded={openProvider === group.name}
                      group={group}
                      onExpand={() => setOpenProvider(group.name)}
                      onToggle={() => setOpenProvider(previous => (previous === group.name ? null : group.name))}
                      rowProps={rowProps}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid min-h-24 place-items-center px-4 py-6 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
                {t.settings.providers.noKeysMatch}
              </div>
            )}
          </div>
        ) : (
          <NoProviderKeys />
        )}
      </SettingsContent>
    )
  }

  if (view === 'custom-endpoints') {
    return <CustomEndpointsSettings onConfigSaved={onConfigSaved} onMainModelChanged={onMainModelChanged} />
  }

  if (view === 'local') {
    return $localModelsEnabled.get() ? <LocalModelsSettings /> : null
  }

  return (
    <SettingsContent>
      <SettingsProfileScope className="mb-5" />
      <OAuthPicker
        accountQuery={accountQuery}
        disconnecting={disconnecting}
        onAccountQueryChange={setAccountQuery}
        onDisconnect={provider => void handleDisconnect(provider)}
        onTerminalDisconnect={provider => void handleTerminalDisconnect(provider)}
        onWantApiKey={() => onViewChange('keys')}
        onWantLocalModels={() => onViewChange('local')}
        profile={scopeProfile}
        providers={oauthProviders}
      />
    </SettingsContent>
  )
}

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
  onClose: () => void
  onConfigSaved?: () => void
  onMainModelChanged?: (provider: string, model: string) => void
  onViewChange: (view: ProviderView) => void
  view: ProviderView
}
