import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { recordAction } from '@/store/desktop-metrics'
import { requestSimplifiedNewSession } from '@/store/simplified-new-session'

import { $workspaceIsPage, NEW_CHAT_ROUTE } from '../routes'

import { TitlebarIcon } from './titlebar-icon'

interface WorkspaceTabChrome {
  tabTitle?: () => ReactNode
}

/** The redesigned band's session tab (Figma 906:19481 + 906:19483): the open session's title as a
 *  pill at the stage's leading edge, then "+" for a new session. The workspace pane already
 *  publishes its live title (stored title, or the draft's) — the same one its tab would show. */
export function PortalSessionTab() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const panes = useContributions('panes')
  const page = useStore($workspaceIsPage)
  const workspace = panes.find(pane => pane.id === 'workspace')

  if (!workspace || page) {
    return null
  }

  const title = (workspace.data as undefined | WorkspaceTabChrome)?.tabTitle?.() ?? workspace.title
  const label = t.sidebar.nav['new-session']

  return (
    <div className="portal-session-tab" data-slot="portal-session-tab">
      {/* The widest thing mid-band and not interactive, so it drags the window like the band
          around it (its cluster is no-drag for the "+"). No tooltip: a drag region gets no hover. */}
      <span className="portal-session-tab-title [-webkit-app-region:drag]">{title}</span>
      <Tip label={<TipKeybindLabel actionId="session.new" text={label} />} placement="toolbar">
        <Button
          aria-label={label}
          className="portal-chrome-button"
          onClick={() => {
            recordAction('session.new', 'click')

            if (!requestSimplifiedNewSession()) {
              navigate(NEW_CHAT_ROUTE)
            }
          }}
          onPointerDown={event => event.stopPropagation()}
          size="icon"
          variant="ghost"
        >
          <TitlebarIcon name="add" />
        </Button>
      </Tip>
    </div>
  )
}
