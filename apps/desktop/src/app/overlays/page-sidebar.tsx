import { useStore } from '@nanostores/react'
import { IconArrowBarToLeft } from '@tabler/icons-react'
import { atom, computed } from 'nanostores'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $sidebarOpen, toggleSidebarOpen } from '@/store/layout'

/** A page's sidebar panel follows the chat sidebar's open state: one ⌘B, one collapse button, and
 *  switching between Chats, Agents and Settings never re-opens a panel the user folded away. */
export function usePageSidebarOpen() {
  // Mounting a page that owns a panel tells the band (PageToolbar) to offer "show sidebar" while
  // the panel is folded away, as the chat band does.
  useEffect(() => {
    $pagePanels.set($pagePanels.get() + 1)

    return () => $pagePanels.set($pagePanels.get() - 1)
  }, [])

  return useStore($sidebarOpen)
}

const $pagePanels = atom(0)

/** A page with a sidebar panel is showing. */
export const $pageHasSidebar = computed($pagePanels, count => count > 0)

/** The panel's title row (895:8991 "Agents", 856:2741 "Settings"): the display title at the
 *  panel's 20px inset and the chat sidebar's 30px collapse button at its end. */
export function PageSidebarHead({ title }: { title: string }) {
  const { t } = useI18n()

  return (
    <div className="page-sidebar-head">
      <h2 className="page-nav-title">{title}</h2>
      <Tip label={t.titlebar.hideSidebar}>
        <Button
          aria-label={t.titlebar.hideSidebar}
          className="page-sidebar-collapse"
          onClick={toggleSidebarOpen}
          size="icon"
          variant="ghost"
        >
          <IconArrowBarToLeft aria-hidden />
        </Button>
      </Tip>
    </div>
  )
}
