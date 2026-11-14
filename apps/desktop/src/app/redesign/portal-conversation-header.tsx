import { useStore } from '@nanostores/react'
import { useLocation } from 'react-router'

import { ChatItemsButton } from '@/app/chat/items/chat-items-button'
import { $workspaceIsPage, appViewForPath, isOverlayView } from '@/app/routes'
import { tabStripVisibleForZone } from '@/components/pane-shell/tree/renderer/strip-visibility'
import type { ZoneHeaderHook } from '@/components/pane-shell/tree/renderer/zone-header-provider'
import { isSessionStripPane } from '@/components/pane-shell/tree/store'
import { usePortalShell } from '@/store/shell-variant'

import { PortalAgentSettingsButton, useConversationAgent } from './portal-agent-settings'

export function usePortalConversation(paneId: string) {
  const portal = usePortalShell()
  const page = useStore($workspaceIsPage)

  return portal && isSessionStripPane(paneId) && (paneId !== 'workspace' || !page)
}

/** The band's title pill stands in for a zone's strip only while that zone holds the lone main chat:
 *  one tab needs no switcher, and the pill already carries its title and "+". Once a second tab
 *  opens, the zone's own strip (in the band, as Figma's header pills) lists them all. */
export function bandPillReplacesStrip(shown: readonly string[]): boolean {
  return shown.length === 1 && shown[0] === 'workspace'
}

/** Every zone keeps the shared strip resolver (so the Tab Strip setting and the stranded-tab rule
 *  hold in the redesign too), except the lone main chat whose tab is the band pill. Installed into
 *  every zone by the contrib controller (provideZoneHeader). */
export const usePortalZoneHeader: ZoneHeaderHook = (node, options) => {
  const portal = usePortalConversation(options.active)

  return {
    stripVisible:
      tabStripVisibleForZone({ ...options, mode: node.tabStrip }) && !(portal && bandPillReplacesStrip(options.shown)),
    conversationHeader: portal && !node.minimized ? <PortalConversationHeader paneId={options.active} /> : null
  }
}

/** Chrome stays outside the chat host, so mode/theme changes cannot discard its draft. The main
 *  chat's Agent settings rides the band (PortalBandAgentSettings); an extra tile keeps its own here. */
export function PortalConversationHeader({ paneId }: { paneId: string }) {
  const { pathname } = useLocation()
  const { chat, label, sessionId } = useConversationAgent(paneId)

  // Route overlays cover the chat but keep its host alive. Covered chrome must
  // leave the tab order without replacing the draft-owning body underneath.
  if (isOverlayView(appViewForPath(pathname))) {
    return null
  }

  // The session title lives in the titlebar pill. What remains overlays the
  // stage's top-right corner out of flow (stage.css), so the stage photo and the
  // transcript start at the panel's top edge; the row itself ignores the pointer
  // and only its controls take it.
  return (
    <header className="@container/conversation" data-slot="portal-conversation-header">
      <span className="portal-conversation-agent hidden min-w-0 truncate @min-[30rem]/conversation:block">{label}</span>
      <ChatItemsButton chat={chat} sessionId={sessionId} />
      {paneId !== 'workspace' && <PortalAgentSettingsButton paneId={paneId} />}
    </header>
  )
}
