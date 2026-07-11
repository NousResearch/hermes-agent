import { useStore } from '@nanostores/react'
import * as React from 'react'

import { onDesktopStateSync, requestActiveDesktopProfile } from '@/lib/desktop-state-sync'
import { $menuBarTransparency, getMenuBarSurfaceAlphas } from '@/store/menu-bar-transparency'
import { $activeGatewayProfile } from '@/store/profile'
import { useTheme } from '@/themes'

import { AppearanceTab } from './tabs/appearance-tab'
import { AtlasTab } from './tabs/atlas-tab'
import { CommandsTab } from './tabs/commands-tab'
import { QuickTab } from './tabs/quick-tab'

export type CompanionTabId = 'appearance' | 'atlas' | 'commands' | 'quick'

const TABS: Array<{ id: CompanionTabId; label: string }> = [
  { id: 'commands', label: 'Commands' },
  { id: 'quick', label: 'Quick' },
  { id: 'appearance', label: 'Appearance' },
  { id: 'atlas', label: 'Atlas' }
]

function prefersReducedMotion(): boolean {
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
}

export function MenuBarCompanionShell() {
  // Desktop theme switcher still works from Appearance tab.
  // Companion chrome itself is always dark (data-chrome="dark").
  const { themeName, mode, resolvedMode } = useTheme()
  const activeProfile = useStore($activeGatewayProfile)
  const menuBarTransparency = useStore($menuBarTransparency)
  const [tab, setTab] = React.useState<CompanionTabId>('commands')
  const [opened, setOpened] = React.useState(false)
  const surfaceAlphas = getMenuBarSurfaceAlphas(menuBarTransparency)

  const surfaceStyle = {
    '--mbc-bg-alpha': surfaceAlphas.background,
    '--mbc-gradient-bottom-alpha': surfaceAlphas.gradientBottom,
    '--mbc-gradient-top-alpha': surfaceAlphas.gradientTop,
    '--mbc-panel-alpha': surfaceAlphas.panel
  } as React.CSSProperties

  React.useEffect(() => {
    const unsubscribe = onDesktopStateSync(message => {
      if (message.type === 'active-profile' && message.profile !== $activeGatewayProfile.get()) {
        $activeGatewayProfile.set(message.profile)
      }
    })

    requestActiveDesktopProfile()

    return unsubscribe
  }, [])

  React.useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        window.close()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  React.useEffect(() => {
    if (prefersReducedMotion()) {
      setOpened(true)

      return
    }

    const frame = window.requestAnimationFrame(() => setOpened(true))

    return () => window.cancelAnimationFrame(frame)
  }, [])

  return (
    <div
      aria-label="Hermes menu bar companion"
      className={opened ? 'hermes-menubar-shell is-open' : 'hermes-menubar-shell'}
      data-chrome="dark"
      data-desktop-mode={mode}
      data-desktop-resolved={resolvedMode}
      data-skin={themeName}
      role="dialog"
      style={surfaceStyle}
    >
      <div aria-hidden="true" className="mbc-seal-burst" />
      <header className="mbc-header">
        <img alt="" className="mbc-brand" src="./nous-girl-template.png" />
        <div className="mbc-header-copy">
          <h1>Hermes</h1>
          <p>
            Dark chrome · skin {themeName} · Desktop {mode}
            {mode === 'system' ? ` → ${resolvedMode}` : ''} · profile {activeProfile}
          </p>
        </div>
      </header>

      <nav aria-label="Companion tabs" className="mbc-tabs">
        {TABS.map(entry => (
          <button
            aria-selected={tab === entry.id}
            className={tab === entry.id ? 'mbc-tab is-active' : 'mbc-tab'}
            key={entry.id}
            onClick={() => setTab(entry.id)}
            role="tab"
            type="button"
          >
            {entry.label}
          </button>
        ))}
      </nav>

      <div className="mbc-body" role="tabpanel">
        {tab === 'commands' ? <CommandsTab /> : null}
        {tab === 'quick' ? <QuickTab /> : null}
        {tab === 'appearance' ? <AppearanceTab /> : null}
        {tab === 'atlas' ? <AtlasTab /> : null}
      </div>
    </div>
  )
}

export function mountMenuBarCompanion() {
  return MenuBarCompanionShell
}
