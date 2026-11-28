import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { lazy, Suspense, useLayoutEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { $apiRequestScope } from '@/api/client'
import { getProfiles } from '@/hermes'
import { useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { $lastRoster } from '@/plugins/hermes-bots/data'
import type { RosterRow } from '@/plugins/hermes-bots/types'
import { $activeConnectionId } from '@/store/connections'
import { refreshFleetRoster } from '@/store/fleet-roster'
import { notifyError } from '@/store/notifications'
import { profileLabel, refreshProfiles } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import { useBotModeLoaded } from '../chat/sidebar/simplified-bot-chat-row'
import { CreateProfileDialog } from '../profiles/create-profile-dialog'
import { DeleteProfileDialog } from '../profiles/delete-profile-dialog'
import { RenameProfileDialog } from '../profiles/rename-profile-dialog'
import { PROFILES_ROUTE } from '../routes'

import { type AgentRouteOwner, agentSettingsKey } from './data'
import { agentMachine, forgetViewedAgent } from './landing'
import { useOverviewActionsCopy } from './overview-actions-copy'
const LazyCreateBotDialog = lazy(async () => ({
  default: (await import('@/plugins/hermes-bots/create-dialog')).CreateAgentDialog
}))
const LazyEditBotDialog = lazy(async () => ({
  default: (await import('@/plugins/hermes-bots/edit-profile-dialog')).EditProfileDialog
}))
/** A create names only the machine (and, duplicating, the agent copied); a rename or delete names
 *  the agent it acts on. */
export type AgentDialogTarget =
  | { kind: 'create' | 'duplicate'; owner: AgentRouteOwner }
  | { kind: 'delete' | 'rename'; owner: AgentRouteOwner; profile: ProfileInfo }

/** The owner's profile list, as its page's General settings reads it (one cache row). */
export const agentProfilesQuery = (owner: AgentRouteOwner) => ({
  queryKey: [...agentSettingsKey(owner), 'profiles'],
  queryFn: () => getProfiles(owner)
})

/** After an agent is created, renamed or deleted, everything that lists agents re-reads: the window's
 *  own list (`$profiles`, which the agent list and an agent page's header read) when the agent lives
 *  on this window's machine, the roster for every machine, and the owner's page reads. */
async function refreshAgentLists(owner: AgentRouteOwner, failedLoad: string): Promise<void> {
  const foreground =
    agentMachine($apiRequestScope.get().connectionId ?? $activeConnectionId.get()) === owner.connectionId

  await Promise.all([
    foreground ? refreshProfiles().catch(error => notifyError(error, failedLoad)) : null,
    refreshFleetRoster({ force: true }),
    queryClient.invalidateQueries({
      predicate: ({ queryKey }) =>
        queryKey[0] === 'agent-settings' && queryKey[1] === owner.connectionId && queryKey[3] === 'profiles'
    })
  ])
}

interface AgentDialogsOptions {
  /** The page follows its renamed agent to the new name (a display-name change keeps the name). */
  onRenamed?: (name: string) => Promise<void>
}

/** New agent, Duplicate, Rename and Delete for every agent surface: one set of dialogs, one refresh
 *  after each, and a delete leaves the deleted agent's page. */
export function useAgentDialogs({ onRenamed }: AgentDialogsOptions = {}) {
  const [target, setTarget] = useState<AgentDialogTarget | null>(null)

  return {
    // The owner is captured as opened: a later window or page change must not retarget the dialog.
    open: (next: AgentDialogTarget) => setTarget({ ...next, owner: { ...next.owner } }),
    dialogs: target && <AgentDialog onClose={() => setTarget(null)} onRenamed={onRenamed} target={target} />
  }
}

interface AgentDialogProps extends AgentDialogsOptions {
  target: AgentDialogTarget
  onClose: () => void
}

export function AgentDialog({ target, onClose, onRenamed }: AgentDialogProps) {
  const { t } = useI18n()
  const c = useOverviewActionsCopy()
  const navigate = useNavigate()
  const { owner } = target
  const profiles = useQuery({ ...agentProfilesQuery(owner), enabled: !('profile' in target) })
  const refresh = () => refreshAgentLists(owner, t.profiles.failedLoad)

  // A delete that finishes after the user moved on must not pull them back.
  const { key } = useLocation()
  const here = useRef<null | string>(key)
  useLayoutEffect(() => {
    here.current = key

    return () => {
      here.current = null
    }
  }, [key])

  async function leaveDeleted() {
    forgetViewedAgent(owner)

    if (here.current === key) {
      navigate(PROFILES_ROUTE, { replace: true })
    }

    await refresh()
  }

  const botMode = useBotModeLoaded()
  const roster = useStore($lastRoster)
  const [editingBot, setEditingBot] = useState<RosterRow | null>(null)
  const activeMachine = agentMachine($apiRequestScope.get().connectionId ?? $activeConnectionId.get())

  if (!('profile' in target)) {
    if (target.kind === 'create' && botMode && owner.connectionId === activeMachine) {
      return (
        <Suspense fallback={null}>
          <LazyCreateBotDialog
            onClose={() => {
              onClose()
              void refresh()
            }}
            onConfigureModel={bot => setEditingBot(bot)}
            open={!editingBot}
            roster={roster}
          />
          {editingBot ? (
            <LazyEditBotDialog
              bot={editingBot}
              onClose={() => {
                setEditingBot(null)
                onClose()
                void refresh()
              }}
              open
            />
          ) : null}
        </Suspense>
      )
    }

    return (
      <CreateProfileDialog
        initialCloneFrom={target.kind === 'duplicate' ? owner.profile : undefined}
        onClose={onClose}
        onCreated={refresh}
        open
        profiles={profiles.data?.profiles}
        scope={owner}
        title={target.kind === 'duplicate' ? c.duplicate : undefined}
      />
    )
  }

  const { profile } = target

  return target.kind === 'rename' ? (
    <RenameProfileDialog
      currentName={owner.profile}
      displayName={profileLabel(profile)}
      isDefault={profile.is_default}
      onClose={onClose}
      onRenamed={async name => {
        // Following the agent first: the page under the old name is already gone.
        if (!profile.is_default) {
          await onRenamed?.(name)
        }

        await refresh()
      }}
      open
      owner={owner}
    />
  ) : (
    <DeleteProfileDialog onClose={onClose} onDeleted={leaveDeleted} open owner={owner} profile={profile} />
  )
}
