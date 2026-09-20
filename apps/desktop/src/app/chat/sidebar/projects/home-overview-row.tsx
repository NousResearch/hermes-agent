import { type NewSessionSplitHandler, startNewSessionDrag } from '@/app/chat/new-session-drag'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import {
  SIDEBAR_LEAD_ICON_SIZE,
  SidebarGroupRow,
  SidebarRowLead,
  SidebarRowLeadGlyph,
  SidebarRowLink
} from '../chrome'

import type { SidebarProjectTree } from './workspace-groups'
import { WorkspaceAddButton } from './workspace-header'

interface HomeOverviewRowProps {
  home: SidebarProjectTree
  onEnter?: (id: string) => void
  onNewSession?: (path: null | string) => void
  onNewSessionSplit?: NewSessionSplitHandler
  activeProjectId?: null | string
}

export function HomeOverviewRow({
  home,
  onEnter,
  onNewSession,
  onNewSessionSplit,
  activeProjectId
}: HomeOverviewRowProps) {
  const { t } = useI18n()
  const s = t.sidebar

  return (
    <div data-sidebar-home={home.id}>
      <SidebarGroupRow
        actions={
          onNewSession ? (
            <WorkspaceAddButton
              label={s.newSessionIn(home.label)}
              onClick={() => onNewSession(null)}
              onPointerDown={
                onNewSessionSplit
                  ? event => {
                      startNewSessionDrag(
                        placement => {
                          onNewSessionSplit(placement.dir, {
                            anchor: placement.anchor,
                            before: placement.before,
                            cwd: null
                          })
                        },
                        event,
                        { cwd: null, label: s.newSessionIn(home.label) }
                      )
                    }
                  : undefined
              }
            />
          ) : undefined
        }
        label={
          <SidebarRowLink
            aria-label={s.projects.enter(home.label)}
            labelClassName={cn(
              'hover:text-foreground hover:underline',
              home.id === activeProjectId && 'text-foreground'
            )}
            onClick={() => onEnter?.(home.id)}
          >
            {home.label}
          </SidebarRowLink>
        }
        lead={
          <SidebarRowLead>
            <SidebarRowLeadGlyph>
              <Codicon name="home" size={SIDEBAR_LEAD_ICON_SIZE} />
            </SidebarRowLeadGlyph>
          </SidebarRowLead>
        }
      />
    </div>
  )
}
