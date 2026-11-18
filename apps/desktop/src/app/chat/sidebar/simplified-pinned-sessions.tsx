import { useStore } from '@nanostores/react'
import type { ComponentProps, ReactNode } from 'react'

import { FadeScroll } from '@/components/ui/fade-scroll'
import { useI18n } from '@/i18n'
import { $portalPinnedSectionOpen } from '@/store/agent-sidebar-preferences'
import { recordIaAction } from '@/store/ia-metrics'

import { PortalSectionHeader } from './portal/portal-section-header'
import { SimplifiedSessionList } from './simplified-session-list'

interface SimplifiedPinnedSessionsProps extends ComponentProps<typeof SimplifiedSessionList> {
  emptyState?: ReactNode
}

export function SimplifiedPinnedSessions({ emptyState, ...props }: SimplifiedPinnedSessionsProps) {
  const { t } = useI18n()
  const pinsOpen = useStore($portalPinnedSectionOpen)

  return (
    <section aria-label={t.sidebar.pinned} className="portal-section" data-open={pinsOpen} data-slot="pinned-section">
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
          {props.sessions.length === 0 ? emptyState : <SimplifiedSessionList {...props} showAgent />}
        </FadeScroll>
      )}
    </section>
  )
}
