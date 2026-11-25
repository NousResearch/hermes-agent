import { useStore } from '@nanostores/react'
import type { ComponentProps } from 'react'

import { FadeScroll } from '@/components/ui/fade-scroll'
import { useI18n } from '@/i18n'
import { $portalPinnedSectionOpen } from '@/store/agent-sidebar-preferences'
import { recordIaAction } from '@/store/ia-metrics'

import { PortalSectionHeader } from './portal/portal-section-header'
import { SimplifiedSessionList } from './simplified-session-list'

/** Pinned is drawn only while it holds rows (HADESK-132); an empty one is the sidebar's own drop
 *  shelf (see PortalPinnedShelf). */
export function SimplifiedPinnedSessions(props: ComponentProps<typeof SimplifiedSessionList>) {
  const { t } = useI18n()
  const pinsOpen = useStore($portalPinnedSectionOpen)

  return (
    <section
      aria-label={t.sidebar.pinned}
      className="portal-section"
      data-open={pinsOpen}
      data-session-pin-drop=""
      data-slot="pinned-section"
    >
      <PortalSectionHeader
        label={t.sidebar.pinned}
        onToggle={() => {
          recordIaAction(pinsOpen ? 'ia.section.pinned_collapsed' : 'ia.section.pinned_expanded')
          $portalPinnedSectionOpen.set(!pinsOpen)
        }}
        open={pinsOpen}
      />
      {pinsOpen && (
        <FadeScroll className="portal-section-scroll" maxHeight="none">
          <SimplifiedSessionList {...props} showAgent />
        </FadeScroll>
      )}
    </section>
  )
}
