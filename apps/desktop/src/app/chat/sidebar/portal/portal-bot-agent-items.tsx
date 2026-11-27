import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { lazy, Suspense } from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenuCheckboxItem,
  ContextMenuItem,
  ContextMenuSeparator
} from '@/components/ui/context-menu'
import {
  DropdownMenuCheckboxItem,
  DropdownMenuItem,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { Monitor } from '@/lib/icons'
import { ensureBotMetadata, ensureBotProfile } from '@/plugins/hermes-bots/canonical-chat'
import { $botMeta, saveBotMeta } from '@/plugins/hermes-bots/data'
import { openGroupChat } from '@/plugins/hermes-bots/group-chat-view'
import { botGroups } from '@/plugins/hermes-bots/group-membership'
import { isBotPinned } from '@/plugins/hermes-bots/hidden-bots'
import { useBots } from '@/plugins/hermes-bots/i18n'
import { displayName } from '@/plugins/hermes-bots/labels'
import { botRecentSession, openBotRecentSession } from '@/plugins/hermes-bots/recent-session'
import { $activityToasts, openRosterBot, setActivityToasts } from '@/plugins/hermes-bots/roster-actions'
import { botRosterMeta } from '@/plugins/hermes-bots/routing'
import { toggleBotScreenAutoOpen } from '@/plugins/hermes-bots/screen-open'
import type { RosterRow } from '@/plugins/hermes-bots/types'
import { refreshFleetRoster } from '@/store/fleet-roster'
import { notify, notifyError } from '@/store/notifications'
import { refreshProfiles } from '@/store/profile'

import { agentOwner, type SimplifiedAgent } from '../use-simplified-agents'

import { openAgentScreen, useScreenBot } from './portal-agent-screen'

const LazyEditProfileDialog = lazy(async () => ({
  default: (await import('@/plugins/hermes-bots/edit-profile-dialog')).EditProfileDialog
}))
const LazyGroupDialog = lazy(async () => ({
  default: (await import('@/plugins/hermes-bots/create-dialog')).GroupDialog
}))

const $portalEditBot = atom<RosterRow | null>(null)
const $portalGroupBot = atom<RosterRow | null>(null)

export interface AgentMenuKit {
  CheckboxItem: typeof DropdownMenuCheckboxItem | typeof ContextMenuCheckboxItem
  Item: typeof DropdownMenuItem | typeof ContextMenuItem
  Separator: typeof DropdownMenuSeparator | typeof ContextMenuSeparator
}

export const DROPDOWN_AGENT_MENU_KIT: AgentMenuKit = {
  CheckboxItem: DropdownMenuCheckboxItem,
  Item: DropdownMenuItem,
  Separator: DropdownMenuSeparator
}

export const CONTEXT_AGENT_MENU_KIT: AgentMenuKit = {
  CheckboxItem: ContextMenuCheckboxItem,
  Item: ContextMenuItem,
  Separator: ContextMenuSeparator
}

/** Bot Mode actions inside the Redesign agent kebab & right-click menu. Mounted only while the
 *  hermes-bots plugin is loaded; keeps every verb functional and styled in the design system. */
export function PortalBotAgentItems({ agent, kit }: { agent: SimplifiedAgent; kit: AgentMenuKit }) {
  const b = useBots()
  const meta = useStore($botMeta)
  const activityToasts = useStore($activityToasts)
  const owner = agentOwner(agent)
  const bot = useScreenBot(owner.connectionId, owner.profile)
  const botMeta = botRosterMeta(bot, meta)
  const pinned = isBotPinned(bot, meta)
  const groups = botGroups(botMeta)
  const hasRecent = Boolean(botRecentSession(bot))

  const togglePin = () => {
    void ensureBotMetadata(bot)
      .then(current => {
        const nextPinned = !current.pinned
        void saveBotMeta(bot, { pinned: nextPinned })
        notify({
          kind: 'info',
          message: nextPinned
            ? b.bot.pinnedToast(displayName(bot, current))
            : b.bot.unpinnedToast(displayName(bot, current))
        })
      })
      .catch(error => notifyError(error, b.bot.metadataLoadFailed))
  }

  return (
    <>
      <kit.Separator />
      <kit.Item onSelect={() => void openRosterBot(bot, { intent: 'tab', label: agent.label })}>
        <Codicon aria-hidden name="go-to-file" size="1rem" />
        <span>Open in new tab</span>
      </kit.Item>
      <kit.Item disabled={!hasRecent} onSelect={() => void openBotRecentSession(bot)}>
        <Codicon aria-hidden name="history" size="1rem" />
        <span>Open recent session</span>
      </kit.Item>
      <kit.Item onSelect={togglePin}>
        <Codicon aria-hidden name="pin" size="1rem" />
        <span>{pinned ? b.bot.unpin : b.bot.pinToTop}</span>
      </kit.Item>
      <kit.Item
        onSelect={() =>
          void ensureBotProfile(bot)
            .then(loaded => $portalEditBot.set(loaded))
            .catch(error => notifyError(error, b.bot.loadFailed))
        }
      >
        <Codicon aria-hidden name="edit" size="1rem" />
        <span>{b.bot.editMenu}</span>
      </kit.Item>
      <kit.Item
        onSelect={() =>
          void ensureBotMetadata(bot)
            .then(() => $portalGroupBot.set(bot))
            .catch(error => notifyError(error, b.bot.groupsLoadFailed))
        }
      >
        <Codicon aria-hidden name="organization" size="1rem" />
        <span>{groups.length ? b.bot.groupsMenu(groups.join(', ')) : b.bot.manageGroups}</span>
      </kit.Item>
      <kit.Separator />
      <kit.Item onSelect={() => void openAgentScreen(owner.connectionId, owner.profile)}>
        <Monitor aria-hidden />
        <span>{b.screen.menu}</span>
      </kit.Item>
      <kit.CheckboxItem
        checked={Boolean(botMeta?.screenAutoOpen)}
        className="portal-agent-menu-setting"
        onSelect={() => void toggleBotScreenAutoOpen(bot)}
      >
        {b.screen.autoOpenMenu}
      </kit.CheckboxItem>
      <kit.CheckboxItem
        checked={activityToasts}
        className="portal-agent-menu-setting"
        onSelect={() => setActivityToasts(!activityToasts)}
      >
        Activity notifications
      </kit.CheckboxItem>
    </>
  )
}

/** Mounted once in PortalSidebar so Edit Bot / Avatar and Manage Groups dialogs stay alive after
 *  the dropdown or context menu closes. */
export function PortalBotAgentDialogs() {
  const editing = useStore($portalEditBot)
  const grouping = useStore($portalGroupBot)

  if (!editing && !grouping) {
    return null
  }

  return (
    <Suspense fallback={null}>
      {editing ? (
        <LazyEditProfileDialog
          bot={editing}
          onClose={() => {
            $portalEditBot.set(null)
            void refreshProfiles()
            void refreshFleetRoster({ force: true })
          }}
          open
        />
      ) : null}
      {grouping ? (
        <LazyGroupDialog
          bot={grouping}
          onClose={() => $portalGroupBot.set(null)}
          onCreated={(group, origin) => openGroupChat(group, origin)}
        />
      ) : null}
    </Suspense>
  )
}
