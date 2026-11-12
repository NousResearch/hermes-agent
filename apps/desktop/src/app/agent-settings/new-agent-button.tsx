import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { $apiRequestScope } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { KbdCombo } from '@/components/ui/kbd'
import { useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { refreshFleetRoster } from '@/store/fleet-roster'
import { $bindings, bindingsFor } from '@/store/keybinds'
import { $newAgentRequest, hostNewAgentRequests } from '@/store/new-agent-request'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profiles, refreshProfiles } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import { CreateProfileDialog } from '../profiles/create-profile-dialog'

import type { AgentSettingsOwner } from './data'

interface NewAgentButtonProps {
  owner?: AgentSettingsOwner
  /** The Agents panel draws it as a 40px sidebar row (895:9447), not a 36px page button. */
  row?: boolean
}

export function NewAgentButton({ owner, row = false }: NewAgentButtonProps) {
  const { t } = useI18n()
  const [target, setTarget] = useState<{ owner: AgentSettingsOwner; profiles: ProfileInfo[] } | null>(null)
  const [host] = useState(() => Symbol('new-agent'))
  const request = useStore($newAgentRequest)
  const shortcut = bindingsFor('agents.new', useStore($bindings))[0]

  const open = () =>
    setTarget({
      owner: {
        connectionId:
          owner?.connectionId ?? $apiRequestScope.get().connectionId ?? $activeConnectionId.get() ?? 'local',
        profile: owner?.profile ?? $activeGatewayProfile.get()
      },
      profiles: $profiles.get()
    })

  useEffect(() => hostNewAgentRequests(host), [host])

  // Only a request made while this button hosts it opens the dialog, never one replayed on mount.
  const [seen, setSeen] = useState(request?.seq)

  if (request?.seq !== seen) {
    setSeen(request?.seq)

    if (request?.host === host) {
      open()
    }
  }

  return (
    <>
      <Button
        className={row ? 'new-agent-row' : undefined}
        onClick={open}
        size={row ? 'sm' : undefined}
        variant="secondary"
      >
        <Codicon name="add" />
        {row ? <span className="new-agent-row-label">{t.profiles.newProfile}</span> : t.profiles.newProfile}
        {row && shortcut && <KbdCombo aria-hidden className="portal-kbd" combo={shortcut} size="sm" />}
      </Button>
      {target && (
        <CreateProfileDialog
          onClose={() => setTarget(null)}
          onCreated={async () => {
            if (
              ($apiRequestScope.get().connectionId ?? $activeConnectionId.get() ?? 'local') ===
              target.owner.connectionId
            ) {
              await refreshProfiles().catch(error => notifyError(error, t.profiles.failedLoad))
            }

            await refreshFleetRoster({ force: true })
          }}
          open
          profiles={target.profiles}
          scope={target.owner}
        />
      )}
    </>
  )
}
