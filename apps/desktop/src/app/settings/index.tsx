import { IconDownload, IconRefresh, IconUpload } from '@tabler/icons-react'
import { useEffect, useRef, useState } from 'react'

import { getHermesConfigDefaults, getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import { triggerHaptic } from '@/lib/haptics'
import { Globe, Info, KeyRound, Package, Wrench } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { OverlayIconButton } from '../overlays/overlay-chrome'
import { OverlaySearchInput } from '../overlays/overlay-search-input'
import { OverlayMain, OverlayNavItem, OverlaySidebar, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'

import { AboutSettings } from './about-settings'
import { AppearanceSettings } from './appearance-settings'
import { ConfigSettings } from './config-settings'
import { SEARCH_PLACEHOLDER, SECTIONS } from './constants'
import { GatewaySettings } from './gateway-settings'
import { KeysSettings } from './keys-settings'
import { McpSettings } from './mcp-settings'
import { ToolsSettings } from './tools-settings'
import type { SettingsPageProps, SettingsQueryKey, SettingsView as SettingsViewId } from './types'

const SETTINGS_VIEWS: readonly SettingsViewId[] = [
  ...SECTIONS.map(s => `config:${s.id}` as SettingsViewId),
  'gateway',
  'keys',
  'mcp',
  'tools',
  'about'
]

export function SettingsView({ gateway, onClose, onConfigSaved }: SettingsPageProps) {
  const [activeView, setActiveView] = useRouteEnumParam('tab', SETTINGS_VIEWS, 'config:model' as SettingsViewId)

  const [queries, setQueries] = useState<Record<SettingsQueryKey, string>>({
    about: '',
    config: '',
    gateway: '',
    keys: '',
    mcp: '',
    tools: ''
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
      notifyError(err, 'Export failed')
    }
  }

  const resetConfig = async () => {
    if (!window.confirm('Reset all settings to Hermes defaults?')) {
      return
    }

    try {
      await saveHermesConfig(await getHermesConfigDefaults())
      triggerHaptic('success')
      onConfigSaved?.()
    } catch (err) {
      notifyError(err, 'Reset failed')
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
      closeLabel="Close settings"
      headerContent={
        <OverlaySearchInput
          containerClassName="w-[min(36rem,calc(100vw-32rem))] min-w-80"
          inputRef={searchInputRef}
          onChange={setQuery}
          placeholder={SEARCH_PLACEHOLDER[queryKey]}
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
                label={s.label}
                onClick={() => setActiveView(view)}
              />
            )
          })}
          <div className="my-2 h-px bg-border/30" />
          <OverlayNavItem
            active={activeView === 'gateway'}
            icon={Globe}
            label="Gateway"
            onClick={() => setActiveView('gateway')}
          />
          <OverlayNavItem
            active={activeView === 'keys'}
            icon={KeyRound}
            label="API Keys"
            onClick={() => setActiveView('keys')}
          />
          <OverlayNavItem
            active={activeView === 'tools'}
            icon={Package}
            label="Skills & Tools"
            onClick={() => setActiveView('tools')}
          />
          <OverlayNavItem
            active={activeView === 'mcp'}
            icon={Wrench}
            label="MCP"
            onClick={() => setActiveView('mcp')}
          />
          <div className="my-2 h-px bg-border/30" />
          <OverlayNavItem
            active={activeView === 'about'}
            icon={Info}
            label="About"
            onClick={() => setActiveView('about')}
          />
          <div className="mt-auto flex items-center gap-1 pt-2">
            <OverlayIconButton onClick={() => void exportConfig()} title="Export config">
              <IconDownload className="size-3.5" />
            </OverlayIconButton>
            <OverlayIconButton
              onClick={() => {
                triggerHaptic('open')
                importInputRef.current?.click()
              }}
              title="Import config"
            >
              <IconUpload className="size-3.5" />
            </OverlayIconButton>
            <OverlayIconButton
              className="hover:text-destructive"
              onClick={() => {
                triggerHaptic('warning')
                void resetConfig()
              }}
              title="Reset to defaults"
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
              query={queries.config}
            />
          ) : activeView === 'keys' ? (
            <KeysSettings query={queries.keys} />
          ) : activeView === 'mcp' ? (
            <McpSettings gateway={gateway} onConfigSaved={onConfigSaved} query={queries.mcp} />
          ) : (
            <ToolsSettings query={queries.tools} />
          )}
        </OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
  )
}

export { SettingsView as SettingsPage }
