import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { Tip } from '@/components/ui/tooltip'
import { Check, Eye, EyeOff, Info, Monitor, MoreVertical } from '@/lib/icons'
import { $botMeta } from '@/plugins/hermes-bots/data'
import { useBots } from '@/plugins/hermes-bots/i18n'
import { botRosterMeta } from '@/plugins/hermes-bots/routing'
import { toggleBotScreenAutoOpen } from '@/plugins/hermes-bots/screen-open'
import { setSidebarAgentHidden } from '@/store/agent-sidebar-preferences'

import { useHiddenAgentCopy } from '../hidden-agent-copy'
import { useBotModeLoaded } from '../simplified-bot-chat-row'
import { agentOwner, type SimplifiedAgent } from '../use-simplified-agents'

import { openAgentScreen, useScreenBot } from './portal-agent-screen'

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

/** Bot Mode's Screen for this agent, whichever chat is in front. Mounted only while that plugin is
 *  loaded. */
function ScreenItems({ agent }: { agent: SimplifiedAgent }) {
  const b = useBots()
  const meta = useStore($botMeta)
  const bot = useScreenBot(agent.connectionId, agent.profile)

  return (
    <>
      <DropdownMenuItem onSelect={() => void openAgentScreen(agent.connectionId, agent.profile)}>
        <Monitor aria-hidden />
        {b.screen.menu}
      </DropdownMenuItem>
      {/* A setting, not a second way to open: no icon of its own, its text under the action's. */}
      <DropdownMenuCheckboxItem
        checked={Boolean(botRosterMeta(bot, meta)?.screenAutoOpen)}
        className="portal-agent-menu-setting"
        onSelect={() => void toggleBotScreenAutoOpen(bot)}
      >
        {b.screen.autoOpenMenu}
      </DropdownMenuCheckboxItem>
    </>
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
  const botMode = useBotModeLoaded()

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
        {botMode && <ScreenItems agent={agent} />}
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
