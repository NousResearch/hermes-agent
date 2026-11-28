import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ContextMenu, ContextMenuContent, ContextMenuTrigger } from '@/components/ui/context-menu'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Check, Eye, EyeOff, Info, MoreVertical } from '@/lib/icons'
import { setSidebarAgentHidden } from '@/store/agent-sidebar-preferences'
import { openPortalCreateProjectDialog } from '@/store/portal-projects'
import { runExportProfileFlow } from '@/store/profile-share'
import { canOpenNewWindow, openNewWindow } from '@/store/windows'

import { useHiddenAgentCopy } from '../hidden-agent-copy'
import { useBotModeLoaded } from '../simplified-bot-chat-row'
import { agentOwner, type SimplifiedAgent } from '../use-simplified-agents'

import {
  type AgentMenuKit,
  CONTEXT_AGENT_MENU_KIT,
  DROPDOWN_AGENT_MENU_KIT,
  PortalBotAgentItems
} from './portal-bot-agent-items'
function HiddenAgentExplanation() {
  const c = useHiddenAgentCopy()

  return (
    <div className="portal-hidden-explanation">
      <strong>{c.hiddenNotMuted}</strong>
      <p>{c.hideDescription}</p>
      <ul>
        <li>
          <Check aria-hidden />
          <span>{c.notificationsContinue}</span>
        </li>
        <li>
          <Check aria-hidden />
          <span>{c.attentionExplanation}</span>
        </li>
      </ul>
      <p className="portal-hidden-restore">{c.restoreExplanation}</p>
    </div>
  )
}

interface PortalAgentMenuProps {
  agent: SimplifiedAgent
  hidden: boolean
  pending: boolean
  onOpen: () => void
}

function PortalAgentMenuItems({
  agent,
  hidden,
  kit,
  onOpen,
  pending
}: PortalAgentMenuProps & { kit: AgentMenuKit }) {
  const { t } = useI18n()
  const c = useHiddenAgentCopy()
  const owner = agentOwner(agent)
  const botMode = useBotModeLoaded()

  return (
    <>
      {/* No "New chat" row: the + beside this menu already starts one (design review 2026-11-17). */}
      <kit.Item disabled={pending} onSelect={onOpen}>
        <FigmaAgentIcon className="portal-agent-menu-icon" color="transparent" variant="outline" />
        {c.manageAgent}
      </kit.Item>
      <kit.Item onSelect={() => openPortalCreateProjectDialog({ agentLabel: agent.label, owner })}>
        <Codicon aria-hidden name="folder" size="1rem" />
        New project
      </kit.Item>
      {canOpenNewWindow() && (
        <kit.Item onSelect={() => void openNewWindow({ connectionId: owner.connectionId, profile: owner.profile })}>
          <Codicon aria-hidden name="empty-window" size="1rem" />
          {t.sidebar.row.newWindow}
        </kit.Item>
      )}
      <kit.Item onSelect={() => void runExportProfileFlow(owner.profile)}>
        <Codicon aria-hidden name="cloud-upload" size="1rem" />
        {t.profiles.exportProfile}
      </kit.Item>
      {botMode && <PortalBotAgentItems agent={agent} kit={kit} />}
      <kit.Separator />
      {hidden ? (
        <kit.Item onSelect={() => setSidebarAgentHidden(owner, false)}>
          <Eye aria-hidden />
          {c.showInChatList}
        </kit.Item>
      ) : (
        <Tip
          align="start"
          boundary="viewport"
          className="portal-hidden-help"
          label={<HiddenAgentExplanation />}
          side="right"
          sideOffset={12}
        >
          {/* The tooltip trigger would rename the row's slot; it stays a menu row, so it keeps the
              menu's row inset and its icon lines up with the rows above. */}
          <kit.Item data-slot="dropdown-menu-item" onSelect={() => setSidebarAgentHidden(owner, true)}>
            <EyeOff aria-hidden />
            <span className="flex-1">{c.hide}</span>
            <Info aria-hidden className="portal-hide-info" />
          </kit.Item>
        </Tip>
      )}
    </>
  )
}

export function PortalAgentMenu(props: PortalAgentMenuProps) {
  const c = useHiddenAgentCopy()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={`${c.actions}: ${props.agent.label} · ${props.agent.gateway}`}
          className="portal-agent-menu-trigger"
          size="icon-xs"
          variant="ghost"
        >
          <MoreVertical aria-hidden />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="portal-agent-menu">
        <PortalAgentMenuItems {...props} kit={DROPDOWN_AGENT_MENU_KIT} />
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** Right-click context menu on an agent heading row: mirrors the kebab menu verbatim. */
export function PortalAgentContextMenu({ children, ...props }: PortalAgentMenuProps & { children: ReactNode }) {
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
      <ContextMenuContent className="portal-agent-menu">
        <PortalAgentMenuItems {...props} kit={CONTEXT_AGENT_MENU_KIT} />
      </ContextMenuContent>
    </ContextMenu>
  )
}
