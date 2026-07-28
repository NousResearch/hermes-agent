import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { codiconIcon } from '@/components/ui/codicon'
import { Loader } from '@/components/ui/loader'
import { ScopedCommandSearch } from '@/components/ui/scoped-command-search'
import { Tip } from '@/components/ui/tooltip'
import { getHermesConfigDefaults, getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import {
  Archive,
  BarChart3,
  Bell,
  ChevronRight,
  Download,
  Globe,
  Info,
  Keyboard,
  KeyRound,
  Package,
  RefreshCw,
  Settings2,
  Upload,
  Wrench,
  Zap
} from '@/lib/icons'
import { notifyError } from '@/store/notifications'

import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { OverlayIconButton } from '../overlays/overlay-chrome'
import { OverlayMain, OverlayNav, type OverlayNavGroup, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'
import { SKILLS_ROUTE } from '../routes'

import { AboutSettings } from './about-settings'
import { AppearanceSettings } from './appearance-settings'
import { BillingSettings } from './billing'
import { ConfigSettings } from './config-settings'
import { SECTIONS } from './constants'
import { GatewaySettings } from './gateway-settings'
import { KeybindSettings } from './keybind-settings'
import { KEYS_VIEWS, KeysSettings, type KeysView } from './keys-settings'
import { NotificationsSettings } from './notifications-settings'
import { PluginsSettings } from './plugins-settings'
import { PROVIDER_VIEWS, ProvidersSettings, type ProviderView } from './providers-settings'
import { SessionsSettings } from './sessions-settings'
import type { SettingsPageProps, SettingsView as SettingsViewId } from './types'
import { useSettingsSearch } from './use-settings-search'

const SETTINGS_VIEWS: readonly SettingsViewId[] = [
  ...SECTIONS.map(s => `config:${s.id}` as SettingsViewId),
  'providers',
  'gateway',
  'keybinds',
  'keys',
  'notifications',
  'billing',
  'plugins',
  'sessions',
  'about'
]

export function SettingsView({ onClose, onConfigSaved, onMainModelChanged }: SettingsPageProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const { hash, pathname, search } = useLocation()
  const [settingsQuery, setSettingsQuery] = useState('')

  // MCP moved out of Settings into Capabilities (/skills?tab=mcp). Keep old
  // `/settings?tab=mcp` deep links working — `useRouteEnumParam` would silently
  // coerce the unknown tab to the default view otherwise. Preserve `server=` so
  // an old bookmark still lands on (and highlights) the selected server.
  useEffect(() => {
    const params = new URLSearchParams(search)

    if (params.get('tab') === 'mcp') {
      const server = params.get('server')
      const suffix = server ? `&server=${encodeURIComponent(server)}` : ''
      navigate(`${SKILLS_ROUTE}?tab=mcp${suffix}`, { replace: true })
    }
  }, [navigate, search])

  const [activeView, setActiveView] = useRouteEnumParam('tab', SETTINGS_VIEWS, 'config:model' as SettingsViewId)
  // Providers subnav (Accounts vs API keys) lives in its own param so each
  // sub-view is deep-linkable and survives a refresh.
  const [providerView, setProviderView] = useRouteEnumParam<ProviderView>('pview', PROVIDER_VIEWS, 'accounts')
  const [keysView] = useRouteEnumParam<KeysView>('kview', KEYS_VIEWS, 'tools')

  const selectActiveView = (view: SettingsViewId) => {
    setSettingsQuery('')
    setActiveView(view)
  }

  // Jump to a section + its sub-view in one navigate. Two sequential setters
  // would each read the same stale `search` and the second would clobber the
  // first's `tab` — so the sub-view never opened on narrow screens.
  const openSubView = (tab: SettingsViewId, param: string, value: string, fallback: string) => {
    setSettingsQuery('')

    const params = new URLSearchParams(search)
    params.set('tab', tab)

    if (value === fallback) {
      params.delete(param)
    } else {
      params.set(param, value)
    }

    const qs = params.toString()
    navigate({ hash, pathname, search: qs ? `?${qs}` : '' }, { replace: true })
  }

  const openProviderView = (view: ProviderView) => openSubView('providers', 'pview', view, 'accounts')
  const openKeysView = (view: KeysView) => openSubView('keys', 'kview', view, 'tools')

  const importInputRef = useRef<HTMLInputElement | null>(null)

  const exportConfig = async () => {
    try {
      const cfg = await getHermesConfigRecord()
      const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'hermes-config.json'
      a.click()
      URL.revokeObjectURL(url)
      triggerHaptic('success')
    } catch (err) {
      notifyError(err, t.settings.exportFailed)
    }
  }

  const resetConfig = async () => {
    if (!window.confirm(t.settings.resetConfirm)) {
      return
    }

    try {
      await saveHermesConfig(await getHermesConfigDefaults())
      triggerHaptic('success')
      onConfigSaved?.()
    } catch (err) {
      notifyError(err, t.settings.resetFailed)
    }
  }

  const navGroups: OverlayNavGroup[] = [
    ...SECTIONS.map(s => {
      const view = `config:${s.id}` as SettingsViewId

      return {
        active: activeView === view,
        icon: s.icon,
        id: view,
        label: t.settings.sections[s.id] ?? s.label,
        onSelect: () => selectActiveView(view)
      }
    }),
    {
      active: activeView === 'notifications',
      icon: Bell,
      id: 'notifications',
      label: t.settings.nav.notifications,
      onSelect: () => selectActiveView('notifications')
    },
    {
      active: activeView === 'billing',
      icon: BarChart3,
      id: 'billing',
      label: t.settings.nav.billing,
      onSelect: () => selectActiveView('billing')
    },
    {
      active: activeView === 'providers',
      children: [
        {
          active: activeView === 'providers' && providerView === 'accounts',
          icon: codiconIcon('account'),
          id: 'pview:accounts',
          label: t.settings.nav.providerAccounts,
          onSelect: () => openProviderView('accounts')
        },
        {
          active: activeView === 'providers' && providerView === 'keys',
          icon: KeyRound,
          id: 'pview:keys',
          label: t.settings.nav.providerApiKeys,
          onSelect: () => openProviderView('keys')
        },
        {
          active: activeView === 'providers' && providerView === 'custom-endpoints',
          icon: Globe,
          id: 'pview:custom-endpoints',
          label: t.settings.nav.providerCustomEndpoints,
          onSelect: () => openProviderView('custom-endpoints')
        }
      ],
      gapBefore: true,
      icon: Zap,
      id: 'providers',
      label: t.settings.nav.providers,
      onSelect: () => selectActiveView('providers')
    },
    {
      active: activeView === 'gateway',
      icon: Globe,
      id: 'gateway',
      label: t.settings.nav.gateway,
      onSelect: () => selectActiveView('gateway')
    },
    {
      active: activeView === 'keybinds',
      icon: Keyboard,
      id: 'keybinds',
      label: t.settings.nav.keybinds,
      onSelect: () => selectActiveView('keybinds')
    },
    {
      active: activeView === 'keys',
      children: [
        {
          active: activeView === 'keys' && keysView === 'tools',
          icon: Wrench,
          id: 'kview:tools',
          label: t.settings.nav.keysTools,
          onSelect: () => openKeysView('tools')
        },
        {
          active: activeView === 'keys' && keysView === 'settings',
          icon: Settings2,
          id: 'kview:settings',
          label: t.settings.nav.keysSettings,
          onSelect: () => openKeysView('settings')
        }
      ],
      icon: KeyRound,
      id: 'keys',
      label: t.settings.nav.apiKeys,
      onSelect: () => selectActiveView('keys')
    },
    {
      active: activeView === 'plugins',
      icon: Package,
      id: 'plugins',
      label: t.settings.nav.plugins,
      onSelect: () => selectActiveView('plugins')
    },
    {
      active: activeView === 'sessions',
      icon: Archive,
      id: 'sessions',
      label: t.settings.nav.archivedChats,
      onSelect: () => selectActiveView('sessions')
    },
    {
      active: activeView === 'about',
      gapBefore: true,
      icon: Info,
      id: 'about',
      label: t.settings.nav.about,
      onSelect: () => selectActiveView('about')
    }
  ]

  const settingsSearch = useSettingsSearch({
    groups: navGroups,
    onSelect: () => setSettingsQuery(''),
    query: settingsQuery
  })

  const searchHeader = (
    <ScopedCommandSearch
      busy={settingsSearch.loading}
      emptyMessage={settingsSearch.loading ? null : t.settings.search.noResults}
      itemClassName="grid-cols-[auto_minmax(0,1fr)_auto]"
      items={settingsSearch.entries}
      listClassName="min-h-0 max-h-none flex-1"
      listHeader={
        <>
          {settingsSearch.error && (
            <div
              className="flex items-center justify-between gap-3 px-3 py-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)"
              role="alert"
            >
              <span>{t.settings.search.catalogError}</span>
              <Button onClick={settingsSearch.retry} size="inline" variant="textStrong">
                {t.common.retry}
              </Button>
            </div>
          )}
          {settingsSearch.loading && (
            <div className="flex items-center gap-2 px-3 py-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              <Loader className="size-3.5" />
              <span>{t.settings.search.loading}</span>
            </div>
          )}
        </>
      }
      onSelect={settingsSearch.selectEntry}
      onValueChange={setSettingsQuery}
      placeholder={t.settings.search.placeholder}
      popoverClassName="left-0 flex max-h-[calc(100vh-9rem)] w-[min(32rem,calc(100vw-4rem))] flex-col max-[47.5rem]:inset-x-0 max-[47.5rem]:w-auto"
      renderItem={entry => {
        const Icon = entry.icon

        return (
          <>
            <Icon className="size-4 self-center text-muted-foreground" />
            <span className="min-w-0">
              <span className="flex min-w-0 items-baseline gap-2">
                <span className="truncate font-medium">{entry.label}</span>
                <span className="shrink-0 text-xs text-muted-foreground">{entry.context}</span>
              </span>
              {entry.description && (
                <span className="mt-0.5 block truncate text-xs text-muted-foreground">{entry.description}</span>
              )}
            </span>
            <ChevronRight className="size-4 self-center text-muted-foreground opacity-0 transition-opacity group-data-[selected=true]:opacity-100" />
          </>
        )
      }}
      resultSummary={
        settingsSearch.loading
          ? t.settings.search.loading
          : t.settings.search.resultCount(settingsSearch.entries.length)
      }
      shouldFilter={false}
      value={settingsQuery}
    />
  )

  const navFooter = (
    <>
      <Tip label={t.settings.exportConfig}>
        <OverlayIconButton onClick={() => void exportConfig()}>
          <Download />
        </OverlayIconButton>
      </Tip>
      <Tip label={t.settings.importConfig}>
        <OverlayIconButton
          onClick={() => {
            triggerHaptic('open')
            importInputRef.current?.click()
          }}
        >
          <Upload />
        </OverlayIconButton>
      </Tip>
      <Tip label={t.settings.resetToDefaults}>
        <OverlayIconButton
          className="hover:text-destructive"
          onClick={() => {
            triggerHaptic('warning')
            void resetConfig()
          }}
        >
          <RefreshCw />
        </OverlayIconButton>
      </Tip>
    </>
  )

  const activeSettingsContent =
    activeView === 'config:appearance' ? (
      <AppearanceSettings />
    ) : activeView === 'about' ? (
      <AboutSettings />
    ) : activeView === 'gateway' ? (
      <GatewaySettings />
    ) : activeView === 'keybinds' ? (
      <KeybindSettings />
    ) : activeView.startsWith('config:') ? (
      <ConfigSettings
        activeSectionId={activeView.slice('config:'.length)}
        importInputRef={importInputRef}
        onConfigSaved={onConfigSaved}
        onMainModelChanged={onMainModelChanged}
      />
    ) : activeView === 'providers' ? (
      <ProvidersSettings
        onClose={onClose}
        onConfigSaved={onConfigSaved}
        onMainModelChanged={onMainModelChanged}
        onViewChange={setProviderView}
        view={providerView}
      />
    ) : activeView === 'keys' ? (
      <KeysSettings view={keysView} />
    ) : activeView === 'notifications' ? (
      <NotificationsSettings />
    ) : activeView === 'billing' ? (
      <BillingSettings />
    ) : activeView === 'plugins' ? (
      <PluginsSettings />
    ) : (
      <SessionsSettings />
    )

  return (
    <OverlayView closeLabel={t.settings.closeSettings} onClose={onClose}>
      <OverlaySplitLayout>
        <OverlayNav footer={navFooter} groups={navGroups} header={searchHeader} />

        <OverlayMain className="px-0 pb-0 pt-[calc(var(--titlebar-height)+1rem)]">{activeSettingsContent}</OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
  )
}

export { SettingsView as SettingsPage }
