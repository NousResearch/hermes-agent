import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { Tip } from '@/components/ui/tooltip'
import { Check, Eye, EyeOff, Info, MoreVertical, Plus } from '@/lib/icons'
import { setSidebarAgentHidden } from '@/store/agent-sidebar-preferences'
import { recordAction } from '@/store/desktop-metrics'
import { recordIaAction } from '@/store/ia-metrics'

import { useHiddenAgentCopy } from '../hidden-agent-copy'
import { startSimplifiedAgentSession } from '../simplified-new-session-model'
import type { SimplifiedAgent } from '../use-simplified-agents'

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

export function PortalAgentMenu({ agent, hidden, pending, onOpen }: PortalAgentMenuProps) {
  const c = useHiddenAgentCopy()
  const owner = { connectionId: agent.connectionId, profile: agent.targetProfile || agent.profile }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={`${c.actions}: ${agent.label} · ${agent.gateway}`}
          className="portal-agent-menu-trigger"
          size="icon-xs"
          variant="ghost"
        >
          <MoreVertical aria-hidden />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="portal-agent-menu">
        <DropdownMenuItem
          onSelect={() => {
            recordAction('session.new', 'click')
            recordIaAction('ia.new_session.agent_row')
            startSimplifiedAgentSession(agent)
          }}
        >
          <Plus aria-hidden />
          {c.newChat}
        </DropdownMenuItem>
        <DropdownMenuItem disabled={pending} onSelect={onOpen}>
          <FigmaAgentIcon className="portal-agent-menu-icon" color="transparent" variant="outline" />
          {c.openAgent}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        {hidden ? (
          <DropdownMenuItem onSelect={() => setSidebarAgentHidden(owner, false)}>
            <Eye aria-hidden />
            {c.showInChatList}
          </DropdownMenuItem>
        ) : (
          <Tip
            align="start"
            boundary="viewport"
            className="portal-hidden-help"
            label={<HiddenAgentExplanation />}
            side="right"
            sideOffset={12}
          >
            <DropdownMenuItem onSelect={() => setSidebarAgentHidden(owner, true)}>
              <EyeOff aria-hidden />
              <span className="flex-1">{c.hide}</span>
              <Info aria-hidden className="portal-hide-info" />
            </DropdownMenuItem>
          </Tip>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
