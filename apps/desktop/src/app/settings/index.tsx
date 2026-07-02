import { IconDownload, IconRefresh, IconUpload } from '@tabler/icons-react'
import { useRef } from 'react'
import { useNavigate } from 'react-router-dom'

import { Tip } from '@/components/ui/tooltip'
import { getHermesConfigDefaults, getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Archive, Bell, Globe, KeyRound, MessageCircle, Settings2, Sparkles, Wrench, Zap } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { OverlayIconButton } from '../overlays/overlay-chrome'
import { OverlayMain, OverlayNavItem, OverlaySidebar, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'
import { MESSAGING_ROUTE } from '../routes'

import { AboutSettings } from './about-settings'
import { AppearanceSettings } from './appearance-settings'
import { ConfigSettings } from './config-settings'
import { isConsumerHiddenSection, SECTIONS } from './constants'
import { GatewaySettings } from './gateway-settings'
import { KEYS_VIEWS, KeysSettings, type KeysView } from './keys-settings'
import { McpSettings } from './mcp-settings'
import { NotificationsSettings } from './notifications-settings'
import { PersonalizationSettings } from './personalization-settings'
import { PROVIDER_VIEWS, ProvidersSettings, type ProviderView } from './providers-settings'
import { SessionsSettings } from './sessions-settings'
import type { SettingsPageProps, SettingsView as SettingsViewId } from './types'

// Every view stays registered — consumer-hidden sections (CONSUMER_HIDDEN_SECTIONS
// in constants.ts) disappear from the nav below but keep working when a `?tab=`
// deep link lands on them.
const SETTINGS_VIEWS: readonly SettingsViewId[] = [
  ...SECTIONS.map(s => `config:${s.id}` as SettingsViewId),
  'providers',
  'gateway',
  'keys',
  'mcp',
  'notifications',
  'sessions',
  'about'
]

export function SettingsView({ gateway, onClose, onConfigSaved, onMainModelChanged }: SettingsPageProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const [activeView, setActiveView] = useRouteEnumParam('tab', SETTINGS_VIEWS, 'config:personalization' as SettingsViewId)
  // Providers subnav (Accounts vs API keys) lives in its own param so each
  // sub-view is deep-linkable and survives a refresh.
  const [providerView, setProviderView] = useRouteEnumParam<ProviderView>('pview', PROVIDER_VIEWS, 'accounts')
  const [keysView, setKeysView] = useRouteEnumParam<KeysView>('kview', KEYS_VIEWS, 'tools')

  const openProviderView = (view: ProviderView) => {
    setActiveView('providers')
    setProviderView(view)
  }

  const openKeysView = (view: KeysView) => {
    setActiveView('keys')
    setKeysView(view)
  }

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

  return (
    <OverlayView closeLabel={t.settings.closeSettings} onClose={onClose}>
      <OverlaySplitLayout>
        <OverlaySidebar>
          {/* Consumer nav: 个性化 / 外观 / 提供方 / 已归档对话. Everything else is
              gated through isConsumerHiddenSection (one set in constants.ts), so
              restoring a section re-lights its nav row, its ⌘K entries and its
              field search hits at once. */}
          {SECTIONS.filter(s => !isConsumerHiddenSection(`config:${s.id}`)).map(s => {
            const view = `config:${s.id}` as SettingsViewId

            return (
              <OverlayNavItem
                active={activeView === view}
                icon={s.icon}
                key={s.id}
                label={t.settings.sections[s.id] ?? s.label}
                onClick={() => setActiveView(view)}
              />
            )
          })}
          {!isConsumerHiddenSection('notifications') && (
            <OverlayNavItem
              active={activeView === 'notifications'}
              icon={Bell}
              label={t.settings.nav.notifications}
              onClick={() => setActiveView('notifications')}
            />
          )}
          <div className="my-2 h-px bg-border/30" />
          <OverlayNavItem
            active={activeView === 'providers'}
            icon={Zap}
            label={t.settings.nav.providers}
            onClick={() => setActiveView('providers')}
          />
          {activeView === 'providers' && (
            <div className="ml-3.5 flex flex-col gap-0.5 pl-1.5">
              <OverlayNavItem
                active={providerView === 'accounts'}
                icon={Sparkles}
                label={t.settings.nav.providerAccounts}
                nested
                onClick={() => openProviderView('accounts')}
              />
              <OverlayNavItem
                active={providerView === 'keys'}
                icon={KeyRound}
                label={t.settings.nav.providerApiKeys}
                nested
                onClick={() => openProviderView('keys')}
              />
            </div>
          )}
          {!isConsumerHiddenSection('gateway') && (
            <OverlayNavItem
              active={activeView === 'gateway'}
              icon={Globe}
              label={t.settings.nav.gateway}
              onClick={() => setActiveView('gateway')}
            />
          )}
          {!isConsumerHiddenSection('keys') && (
            <>
              <OverlayNavItem
                active={activeView === 'keys'}
                icon={KeyRound}
                label={t.settings.nav.apiKeys}
                onClick={() => setActiveView('keys')}
              />
              {activeView === 'keys' && (
                <div className="ml-3.5 flex flex-col gap-0.5 pl-1.5">
                  <OverlayNavItem
                    active={keysView === 'tools'}
                    icon={Wrench}
                    label={t.settings.nav.keysTools}
                    nested
                    onClick={() => openKeysView('tools')}
                  />
                  <OverlayNavItem
                    active={keysView === 'settings'}
                    icon={Settings2}
                    label={t.settings.nav.keysSettings}
                    nested
                    onClick={() => openKeysView('settings')}
                  />
                </div>
              )}
            </>
          )}
          {!isConsumerHiddenSection('mcp') && (
            <OverlayNavItem
              active={activeView === 'mcp'}
              icon={Wrench}
              label={t.settings.nav.mcp}
              onClick={() => setActiveView('mcp')}
            />
          )}
          <OverlayNavItem
            active={activeView === 'sessions'}
            icon={Archive}
            label={t.settings.nav.archivedChats}
            onClick={() => setActiveView('sessions')}
          />
          {/* 消息平台 jump to its full-screen route (navigating closes the
              settings overlay). 产物 moved to the chat sidebar (below 插件);
              关于 lives inside 个性化 now. */}
          {!isConsumerHiddenSection('messaging') && (
            <>
              <div className="my-2 h-px bg-border/30" />
              <OverlayNavItem
                active={false}
                icon={MessageCircle}
                label={t.commandCenter.nav.messaging.title}
                onClick={() => navigate(MESSAGING_ROUTE)}
              />
            </>
          )}
          <div className="mt-auto flex items-center gap-1 pt-2">
            <Tip label={t.settings.exportConfig}>
              <OverlayIconButton onClick={() => void exportConfig()}>
                <IconDownload className="size-3.5" />
              </OverlayIconButton>
            </Tip>
            <Tip label={t.settings.importConfig}>
              <OverlayIconButton
                onClick={() => {
                  triggerHaptic('open')
                  importInputRef.current?.click()
                }}
              >
                <IconUpload className="size-3.5" />
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
                <IconRefresh className="size-3.5" />
              </OverlayIconButton>
            </Tip>
          </div>
        </OverlaySidebar>

        <OverlayMain className="px-0 pb-0 pt-[calc(var(--titlebar-height)+1rem)]">
          {activeView === 'config:personalization' ? (
            <PersonalizationSettings onConfigSaved={onConfigSaved} />
          ) : activeView === 'config:appearance' ? (
            <AppearanceSettings />
          ) : activeView === 'about' ? (
            <AboutSettings />
          ) : activeView === 'gateway' ? (
            <GatewaySettings />
          ) : activeView.startsWith('config:') ? (
            <ConfigSettings
              activeSectionId={activeView.slice('config:'.length)}
              importInputRef={importInputRef}
              onConfigSaved={onConfigSaved}
              onMainModelChanged={onMainModelChanged}
            />
          ) : activeView === 'providers' ? (
            <ProvidersSettings onClose={onClose} onViewChange={setProviderView} view={providerView} />
          ) : activeView === 'keys' ? (
            <KeysSettings view={keysView} />
          ) : activeView === 'mcp' ? (
            <McpSettings gateway={gateway} onConfigSaved={onConfigSaved} />
          ) : activeView === 'notifications' ? (
            <NotificationsSettings />
          ) : (
            <SessionsSettings />
          )}
        </OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
  )
}

export { SettingsView as SettingsPage }
