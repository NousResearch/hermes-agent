import { useStore } from '@nanostores/react'
import { type ReactNode, useLayoutEffect } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { formatModifierToken } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import { openCommandPalette } from '@/store/command-palette'
import { $interfaceMode, setInterfaceMode } from '@/store/interface-mode'
import { $fileBrowserOpen, $sidebarOpen, setFileBrowserOpen, toggleFileBrowserOpen, toggleSidebarOpen } from '@/store/layout'
import { $newChatProfile } from '@/store/profile'
import { openFolderAsProject } from '@/store/projects'
import { isBrowserWindow, isHudWindow } from '@/store/windows'

import { appViewForPath, CAPABILITIES_ROUTE, CRON_ROUTE, isOverlayView, MESSAGING_ROUTE, navigateToWorkspacePage, SETTINGS_ROUTE } from '../routes'

import { $titlebarMenuActive, TITLEBAR_CHROME_CHANGED_EVENT, TITLEBAR_HEIGHT } from './titlebar'

const menuBarGeneration = Date.now()

const menuTriggerClass =
  'h-6 rounded px-2 text-[13px] text-(--ui-text-secondary) outline-none hover:bg-(--ui-control-hover-background) hover:text-foreground data-[state=open]:bg-(--ui-control-hover-background) data-[state=open]:text-foreground'

function editCommand(command: 'copy' | 'cut' | 'paste' | 'selectAll' | 'undo') {
  requestAnimationFrame(() => {
    document.execCommand(command)
  })
}

function Menu({ label, children }: { label: string; children: ReactNode }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className={menuTriggerClass} type="button">
          {label}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="z-[100]" sideOffset={6}>
        {children}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function MenuBar({ startFreshSession }: { startFreshSession: () => void }) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const location = useLocation()
  const sidebarOpen = useStore($sidebarOpen)
  const filesOpen = useStore($fileBrowserOpen)
  const mode = useStore($interfaceMode)
  const view = appViewForPath(location.pathname)
  const hidden = isHudWindow() || isBrowserWindow() || isOverlayView(view)

  useLayoutEffect(() => {
    if (hidden) {
      return
    }

    $titlebarMenuActive.set(true)
    window.dispatchEvent(new Event(TITLEBAR_CHROME_CHANGED_EVENT))

    return () => {
      $titlebarMenuActive.set(false)
      window.dispatchEvent(new Event(TITLEBAR_CHROME_CHANGED_EVENT))
    }
  }, [hidden, menuBarGeneration])

  if (hidden) {
    return null
  }

  const openPage = (path: string) => navigateToWorkspacePage(navigate, path)

  return (
    <>
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-x-0 top-0 z-[60]"
      data-titlebar-menu-bg=""
      style={{
        height: TITLEBAR_HEIGHT,
        background: 'color-mix(in srgb, var(--ui-bg-editor) 45%, var(--ui-bg-chrome))'
      }}
    />
    <div
      className="pointer-events-auto fixed z-[80] flex items-center [-webkit-app-region:drag]"
      data-titlebar-menu=""
      style={{
        top: 0,
        height: TITLEBAR_HEIGHT,
        left: 'calc(var(--titlebar-controls-left) + var(--titlebar-controls-width))',
        right: 'calc(var(--titlebar-tools-right) + var(--titlebar-tools-width))'
      }}
    >
      <nav aria-label={t.menuBar.file} className="relative z-10 flex items-center pl-1 [-webkit-app-region:no-drag]">
        <Menu label={t.menuBar.file}>
          <DropdownMenuItem
            onSelect={() => {
              setWorkspaceScope('sessions')
              $newChatProfile.set(null)
              startFreshSession()
            }}
          >
            {t.sidebar.nav['new-session']}
          </DropdownMenuItem>
          <DropdownMenuItem
            onSelect={() => {
              void openFolderAsProject().then(() => setFileBrowserOpen(true))
            }}
          >
            {t.menuBar.openFolder}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => toggleFileBrowserOpen()}>
            {filesOpen ? t.menuBar.hideFiles : t.menuBar.showFiles}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => navigate(SETTINGS_ROUTE)}>{t.commandCenter.nav.settings.title}</DropdownMenuItem>
        </Menu>
        <Menu label={t.menuBar.edit}>
          <DropdownMenuItem onSelect={() => editCommand('undo')}>{t.menuBar.undo}</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => editCommand('cut')}>{t.contextMenu.edit.cut}</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => editCommand('copy')}>{t.common.copy}</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => editCommand('paste')}>{t.contextMenu.edit.paste}</DropdownMenuItem>
        </Menu>
        <Menu label={t.menuBar.selection}>
          <DropdownMenuItem onSelect={() => editCommand('selectAll')}>{t.contextMenu.edit.selectAll}</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => editCommand('copy')}>{t.common.copy}</DropdownMenuItem>
        </Menu>
        <Menu label={t.menuBar.view}>
          <DropdownMenuItem onSelect={() => toggleSidebarOpen()}>
            {sidebarOpen ? t.titlebar.hideSidebar : t.titlebar.showSidebar}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => setInterfaceMode('simple')}>
            <span className={cn('w-3', mode === 'simple' && 'text-foreground')}>{mode === 'simple' ? '✓' : ''}</span>
            {t.interfaceMode.simple.label}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => setInterfaceMode('advanced')}>
            <span className={cn('w-3', mode === 'advanced' && 'text-foreground')}>{mode === 'advanced' ? '✓' : ''}</span>
            {t.interfaceMode.advanced.label}
          </DropdownMenuItem>
        </Menu>
        <Menu label={t.menuBar.tools}>
          <DropdownMenuItem onSelect={() => openPage(CAPABILITIES_ROUTE)}>{t.sidebar.nav.capabilities}</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => openPage(MESSAGING_ROUTE)}>{t.sidebar.nav.messaging}</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => openPage(CRON_ROUTE)}>{t.sidebar.nav.cron}</DropdownMenuItem>
        </Menu>
      </nav>
      <button
        className="absolute z-[1] flex h-6 w-[min(18rem,32vw)] -translate-x-1/2 items-center gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-editor) px-2 text-[12px] text-(--ui-text-tertiary) [-webkit-app-region:no-drag] hover:text-foreground"
        onClick={() => openCommandPalette()}
        style={{
          left: 'calc(50vw - (var(--titlebar-controls-left) + var(--titlebar-controls-width)))',
          top: (TITLEBAR_HEIGHT - 24) / 2
        }}
        title={t.titlebar.searchTitle}
        type="button"
      >
        <Codicon className="shrink-0" name="search" size="0.8rem" />
        <span className="min-w-0 flex-1 truncate text-left">{t.titlebar.search}</span>
        <span className="shrink-0 text-[10px] tracking-wide opacity-80">{formatModifierToken('mod')} K</span>
      </button>
    </div>
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-x-0 z-[85] h-px"
      data-titlebar-menu-line=""
      style={{
        top: TITLEBAR_HEIGHT - 1,
        background: 'var(--sidebar-edge-border)'
      }}
    />
    </>
  )
}
