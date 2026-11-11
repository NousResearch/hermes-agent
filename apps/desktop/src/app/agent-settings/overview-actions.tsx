import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { $apiRequestScope } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { writeClipboardText } from '@/lib/clipboard'
import { $hiddenSidebarAgents, agentSidebarKey, setSidebarAgentHidden } from '@/store/agent-sidebar-preferences'
import { $activeConnectionId } from '@/store/connections'
import { $fleetRoster, refreshFleetRoster } from '@/store/fleet-roster'
import { notify, notifyError } from '@/store/notifications'
import { $profiles, refreshProfiles } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import { useHiddenAgentCopy } from '../chat/sidebar/hidden-agent-copy'
import { CreateProfileDialog } from '../profiles/create-profile-dialog'

import { useAgentSettingsCopy } from './copy'
import type { AgentSettingsOwner } from './data'
import { useDefaultAgent } from './default-agent'
import { useOverviewActionsCopy } from './overview-actions-copy'

interface AgentOverviewActionsProps {
  owner: AgentSettingsOwner
  profile?: ProfileInfo
}

export function AgentOverviewActions({ owner, profile }: AgentOverviewActionsProps) {
  const { t } = useI18n()
  const c = useOverviewActionsCopy()
  const shared = useHiddenAgentCopy()
  const overview = useAgentSettingsCopy()
  const hiddenAgents = useStore($hiddenSidebarAgents)
  const roster = useStore($fleetRoster)
  const [duplicate, setDuplicate] = useState<{ owner: AgentSettingsOwner; profiles: ProfileInfo[] } | null>(null)
  const { canSetDefault, isDefault, makeDefault, saving: savingDefault } = useDefaultAgent(owner, profile)
  const route = { connectionId: owner.connectionId ?? null, profile: owner.profile }

  const sidebarOwner = {
    ...route,
    profile:
      roster?.agents.find(agent => agent.connectionId === route.connectionId && agent.profile === owner.profile)
        ?.targetProfile || owner.profile
  }

  const hidden = hiddenAgents.includes(agentSidebarKey(sidebarOwner))

  async function copyPath() {
    if (!profile?.path) {
      return
    }

    try {
      await writeClipboardText(profile.path)
      notify({ kind: 'success', title: t.common.copied, message: profile.path })
    } catch (error) {
      notifyError(error, t.common.copyFailed)
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button aria-label={shared.actions} size="icon" variant="secondary">
            <Codicon name="ellipsis" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          aria-label={shared.actions}
          className="agent-overview-menu w-66 max-w-[calc(100vw-2rem)] p-2"
        >
          <DropdownMenuItem
            disabled={!profile}
            onSelect={() => setDuplicate({ owner: { ...owner }, profiles: $profiles.get() })}
            title={!profile ? overview.unavailable : undefined}
          >
            <Codicon name="copy" />
            <span>{c.duplicate}</span>
          </DropdownMenuItem>
          <DropdownMenuItem disabled={isDefault || !canSetDefault || savingDefault} onSelect={() => void makeDefault()}>
            <Codicon name={isDefault ? 'check' : 'star-empty'} />
            <span>{isDefault ? t.profiles.defaultProfile : c.setDefault}</span>
          </DropdownMenuItem>
          {!canSetDefault && <p className="px-2 py-1 text-xs text-(--ui-text-tertiary)">{c.defaultUnavailable}</p>}
          <DropdownMenuItem onSelect={() => setSidebarAgentHidden(sidebarOwner, !hidden)}>
            <Codicon name="eye-closed" />
            <span>{hidden ? shared.showInChatList : shared.hide}</span>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem className="items-start" disabled={!profile?.path} onSelect={() => void copyPath()}>
            <Codicon className="mt-0.5" name="copy" />
            <span className="min-w-0">
              <span className="block">{c.copyPath}</span>
              <span className="mt-1 block break-all font-mono text-xs text-(--ui-text-tertiary)">
                {profile?.path || overview.unavailable}
              </span>
            </span>
          </DropdownMenuItem>
          <p className="px-2 pb-1 pt-3 text-xs leading-relaxed text-(--ui-text-tertiary)">{c.deleteHint}</p>
        </DropdownMenuContent>
      </DropdownMenu>
      {duplicate && (
        <CreateProfileDialog
          initialCloneFrom={duplicate.owner.profile}
          onClose={() => setDuplicate(null)}
          onCreated={async () => {
            if (
              ($apiRequestScope.get().connectionId ?? $activeConnectionId.get() ?? LOCAL_CONNECTION_ID) ===
              duplicate.owner.connectionId
            ) {
              await refreshProfiles().catch(error => notifyError(error, t.profiles.failedLoad))
            }

            await refreshFleetRoster({ force: true })
          }}
          open
          profiles={duplicate.profiles}
          scope={duplicate.owner}
        />
      )}
    </>
  )
}
