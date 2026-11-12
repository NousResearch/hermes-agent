import { useState } from 'react'

import { $apiRequestScope } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { refreshFleetRoster } from '@/store/fleet-roster'
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

  return (
    <>
      <Button
        onClick={() =>
          setTarget({
            owner: {
              connectionId:
                owner?.connectionId ?? $apiRequestScope.get().connectionId ?? $activeConnectionId.get() ?? 'local',
              profile: owner?.profile ?? $activeGatewayProfile.get()
            },
            profiles: $profiles.get()
          })
        }
        size={row ? 'sm' : undefined}
        variant="secondary"
      >
        <Codicon name="add" />
        {t.profiles.newProfile}
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
