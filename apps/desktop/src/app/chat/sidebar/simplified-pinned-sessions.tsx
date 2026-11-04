import { useStore } from '@nanostores/react'
import { type ComponentProps } from 'react'

import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { Codicon } from '@/components/ui/codicon'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { RowButton } from '@/components/ui/row-button'
import { useI18n } from '@/i18n'
import { recordIaAction } from '@/store/ia-metrics'
import { $sidebarPinsOpen, setSidebarPinsOpen } from '@/store/layout'

import { SimplifiedSessionList } from './simplified-session-list'

export function SimplifiedPinnedSessions(props: ComponentProps<typeof SimplifiedSessionList>) {
  const { t } = useI18n()
  const pinsOpen = useStore($sidebarPinsOpen)

  return (
    <section
      aria-label={t.sidebar.pinned}
      className={pinsOpen ? 'flex min-h-17 flex-[1_1_0%] flex-col' : 'flex shrink-0 flex-col'}
    >
      <RowButton
        aria-expanded={pinsOpen}
        className="flex w-full shrink-0 items-center rounded-sm text-left hover:bg-(--chrome-action-hover) focus-visible:outline-2 focus-visible:outline-ring"
        onClick={() => {
          recordIaAction(pinsOpen ? 'ia.section.pinned_collapsed' : 'ia.section.pinned_expanded')
          setSidebarPinsOpen(!pinsOpen)
        }}
      >
        <SidebarPanelLabel className="flex-1 px-2 py-2 [&>span:first-child]:mx-1">{t.sidebar.pinned}</SidebarPanelLabel>
        <Codicon
          className="mr-2 text-(--ui-text-tertiary)"
          name={pinsOpen ? 'chevron-down' : 'chevron-right'}
          size={12}
        />
      </RowButton>
      {pinsOpen && (
        <FadeScroll className="min-h-0" fade="min(2.5rem, 25%)" maxHeight="none">
          <SimplifiedSessionList {...props} showAgent />
        </FadeScroll>
      )}
    </section>
  )
}
