import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { $apiRequestScope } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { KbdCombo } from '@/components/ui/kbd'
import { useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { $bindings, bindingsFor } from '@/store/keybinds'
import { $newAgentRequest, hostNewAgentRequests } from '@/store/new-agent-request'
import { $activeGatewayProfile } from '@/store/profile'

import { useAgentDialogs } from './agent-dialogs'
import type { AgentSettingsOwner } from './data'
import { agentMachine } from './landing'

interface NewAgentButtonProps {
  owner?: AgentSettingsOwner
  /** The Agents panel draws it as a 40px sidebar row (895:9447), not a 36px page button. */
  row?: boolean
}

/** A new agent's home: the given owner's machine and profile, else the window's own. */
export const newAgentTarget = (owner?: AgentSettingsOwner) => ({
  kind: 'create' as const,
  owner: {
    connectionId: agentMachine(owner?.connectionId ?? $apiRequestScope.get().connectionId ?? $activeConnectionId.get()),
    profile: owner?.profile ?? $activeGatewayProfile.get()
  }
})

export function NewAgentButton({ owner, row = false }: NewAgentButtonProps) {
  const { t } = useI18n()
  const { open: openDialog, dialogs } = useAgentDialogs()
  const [host] = useState(() => Symbol('new-agent'))
  const request = useStore($newAgentRequest)
  const shortcut = bindingsFor('agents.new', useStore($bindings))[0]

  const open = () => openDialog(newAgentTarget(owner))

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
      {dialogs}
    </>
  )
}
