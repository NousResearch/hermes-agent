import { useState } from 'react'

import { AgentDialog } from './agent-dialogs'
import type { AgentSettingsOwner } from './data'
import { newAgentTarget } from './new-agent-button'

interface NewAgentDialogProps {
  /** An agent of the machine the new one is made on. */
  owner: AgentSettingsOwner
  onClose: () => void
}

/** The New agent dialog for a surface outside the Agents page (a machine's "+" in the chat sidebar):
 *  the same dialog and refresh, for that machine. A default export so that surface loads it only
 *  when it is first opened. */
export default function NewAgentDialog({ owner, onClose }: NewAgentDialogProps) {
  // Captured as opened: a window or profile change while it is up must not retarget it.
  const [target] = useState(() => newAgentTarget(owner))

  return <AgentDialog onClose={onClose} target={target} />
}
