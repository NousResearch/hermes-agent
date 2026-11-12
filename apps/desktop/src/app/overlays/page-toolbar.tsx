import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { TitlebarIcon } from '@/app/shell/titlebar-icon'
import { Button } from '@/components/ui/button'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $sidebarOpen, toggleSidebarOpen } from '@/store/layout'

import { $pageHasSidebar } from './page-sidebar'

/** The redesigned page's titlebar band (Figma 856:2713): a 30px Back chrome button where the band's
 *  leading cluster starts (`--portal-band-start`, past macOS traffic lights), the page's own controls
 *  centered, its actions at the end. The toolbar begins at the rail's edge, so it insets by the rest.
 *  Only redesigned pages mount it. */

export function PageToolbar({
  actions,
  children,
  onBack
}: {
  actions?: ReactNode
  children?: ReactNode
  onBack: () => void
}) {
  const { t } = useI18n()
  const hasSidebar = useStore($pageHasSidebar)
  const sidebarOpen = useStore($sidebarOpen)

  return (
    <header className="flex h-full min-h-0 shrink-0 items-center gap-(--titlebar-control-gap) ps-[calc(var(--portal-band-start)-var(--global-nav-width))] pe-5 rtl:ps-5">
      <Tip label={t.common.back} placement="toolbar">
        <Button
          aria-label={t.common.back}
          className="portal-chrome-button [-webkit-app-region:no-drag]"
          onClick={onBack}
          size="icon"
          variant="ghost"
        >
          <TitlebarIcon name="arrow-left" />
        </Button>
      </Tip>
      {/* A folded page panel comes back the way the chat sidebar does: the band's toggle, ⌘B. */}
      {hasSidebar && !sidebarOpen && (
        <Tip
          label={<TipKeybindLabel actionId="view.toggleSidebar" text={t.titlebar.showSidebar} />}
          placement="toolbar"
        >
          <Button
            aria-label={t.titlebar.showSidebar}
            className="portal-chrome-button [-webkit-app-region:no-drag]"
            onClick={toggleSidebarOpen}
            size="icon"
            variant="ghost"
          >
            <TitlebarIcon name="layout-sidebar-left" />
          </Button>
        </Tip>
      )}
      <div className="flex min-w-0 flex-1 items-center justify-center gap-2 [-webkit-app-region:no-drag]">
        {children}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-1.5 [-webkit-app-region:no-drag]">{actions}</div>}
    </header>
  )
}
