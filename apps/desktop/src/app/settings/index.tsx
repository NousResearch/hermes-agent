import { IconDownload, IconRefresh, IconUpload } from '@tabler/icons-react'
import { useEffect, useRef, useState } from 'react'

import { getHermesConfigDefaults, getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import { useTranslation } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Archive, Globe, Info, KeyRound, Wrench } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { OverlayIconButton } from '../overlays/overlay-chrome'
import { OverlaySearchInput } from '../overlays/overlay-search-input'
import { OverlayMain, OverlayNavItem, OverlaySidebar, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'

import { AboutSettings } from './about-settings'
import { AppearanceSettings } from './appearance-settings'
import { ConfigSettings } from './config-settings'
import { SEARCH_PLACEHOLDER_KEYS, SECTIONS } from './constants'
import { GatewaySettings } from './gateway-settings'
import { KeysSettings } from './keys-settings'
import { McpSettings } from './mcp-settings'
import { SessionsSettings } from './sessions-settings'
import type { SettingsPageProps, SettingsQueryKey, SettingsView as SettingsViewId } from './types'

const SETTINGS_VIEWS: readonly SettingsViewId[] = [
  ...SECTIONS.map(s => `config:${s.id}` as SettingsViewId),
  'gateway',
  'keys',
  'mcp',
  'sessions',
  'about'
]

export function SettingsView({ gateway, onClose, onConfigSaved, onMainModelChanged }: SettingsPageProps) {
  const [activeView, setActiveView] = useRouteEnumParam('tab', SETTINGS_VIEWS, 'config:model' as SettingsViewId)
  const t = useTranslation()

  const [queries, setQueries] = useState<Record<SettingsQueryKey, string>>({
    about: '',
    config: '',
    gateway: '',
    keys: '',
    mcp: '',
    sessions: ''
  })

  const searchInputRef = useRef<HTMLInputElement>(null)
  const importInputRef = useRef<HTMLInputElement | null>(null)

  const queryKey: SettingsQueryKey = activeView.startsWith('config:') ? 'config' : (activeView as SettingsQueryKey)
  const query = queries[queryKey]
  const setQuery = (next: string) => setQueries(c => ({ ...c, [queryKey]: next }))

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
      notifyError(err, t('settings.actions.exportFailed'))
    }
  }

  const resetConfig = async () => {
    if (!window.confirm(t('settings.actions.resetConfirm'))) {
      return
    }

    try {
      await saveHermesConfig(await getHermesConfigDefaults())
      triggerHaptic('success')
      onConfigSaved?.()
    } catch (err) {
      notifyError(err, t('settings.actions.resetFailed'))
    }
  }

  // OverlayView handles Esc; this just adds Cmd/Ctrl+P → focus search.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'p') {
        e.preventDefault()
        searchInputRef.current?.focus()
        searchInputRef.current?.select()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  return (
    <OverlayView
      closeLabel={t('settings.actions.close')}
      headerContent={
        <OverlaySearchInput
          containerClassName="w-[min(36rem,calc(100vw-32rem))] min-w-80"
          inputRef={searchInputRef}
          onChange={setQuery}
          placeholder={t(SEARCH_PLACEHOLDER_KEYS[queryKey])}
          value={query}
        />
      }
      onClose={onClose}
    >
      <OverlaySplitLayout>
        <OverlaySidebar>
          {SECTIONS.map(s => {
            const view = `config:${s.id}` as SettingsViewId

            return (
              <OverlayNavItem
                active={activeView === view && !queries.config.trim()}
                icon={s.icon}
                key={s.id}
                label={t(s.labelKey)}
                onClick={() => setActiveView(view)}
              />
            )
          })}
          <div className="my-2 h-px bg-border/30" />
          <OverlayNavItem
            active={activeView === 'gateway'}
            icon={Globe}
            label={t('settings.sections.gateway')}
            onClick={() => setActiveView('gateway')}
          />
          <OverlayNavItem
            active={activeView === 'keys'}
            icon={KeyRound}
            label={t('settings.sections.keys')}
            onClick={() => setActiveView('keys')}
          />
          <OverlayNavItem
            active={activeView === 'mcp'}
            icon={Wrench}
                label={t('settings.sections.mcp')}
            onClick={() => setActiveView('mcp')}
          />
          <OverlayNavItem
            active={activeView === 'sessions'}
            icon={Archive}
            label={t('settings.sections.sessions')}
            onClick={() => setActiveView('sessions')}
          />
          <div className="my-2 h-px bg-border/30" />
          <OverlayNavItem
            active={activeView === 'about'}
            icon={Info}
            label={t('settings.sections.about')}
            onClick={() => setActiveView('about')}
          />
          <div className="mt-auto flex items-center gap-1 pt-2">
            <OverlayIconButton onClick={() => void exportConfig()} title={t('settings.actions.export')}>
              <IconDownload className="size-3.5" />
            </OverlayIconButton>
            <OverlayIconButton
              onClick={() => {
                triggerHaptic('open')
                importInputRef.current?.click()
              }}
              title={t('settings.actions.import')}
            >
              <IconUpload className="size-3.5" />
            </OverlayIconButton>
            <OverlayIconButton
              className="hover:text-destructive"
              onClick={() => {
                triggerHaptic('warning')
                void resetConfig()
              }}
              title={t('settings.actions.reset')}
            >
              <IconRefresh className="size-3.5" />
            </OverlayIconButton>
          </div>
        </OverlaySidebar>

        <OverlayMain className="p-0">
          {activeView === 'config:appearance' ? (
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
              query={queries.config}
            />
          ) : activeView === 'keys' ? (
            <KeysSettings query={queries.keys} />
          ) : activeView === 'mcp' ? (
            <McpSettings gateway={gateway} onConfigSaved={onConfigSaved} query={queries.mcp} />
          ) : (
            <SessionsSettings query={queries.sessions} />
          )}
        </OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
  )
}

export { SettingsView as SettingsPage }
