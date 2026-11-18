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
import { Check, Eye, EyeOff, Info, MoreVertical } from '@/lib/icons'
import { setSidebarAgentHidden } from '@/store/agent-sidebar-preferences'

import { useHiddenAgentCopy } from '../hidden-agent-copy'
import { agentOwner, type SimplifiedAgent } from '../use-simplified-agents'

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
  const owner = agentOwner(agent)

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
        {/* No "New chat" row: the + beside this menu already starts one (design review 2026-11-17). */}
        <DropdownMenuItem disabled={pending} onSelect={onOpen}>
          <FigmaAgentIcon className="portal-agent-menu-icon" color="transparent" variant="outline" />
          {c.manageAgent}
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
            {/* The tooltip trigger would rename the row's slot; it stays a menu row, so it keeps the
                menu's row inset and its icon lines up with the rows above. */}
            <DropdownMenuItem data-slot="dropdown-menu-item" onSelect={() => setSidebarAgentHidden(owner, true)}>
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
