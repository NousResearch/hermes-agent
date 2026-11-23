import { useState } from 'react'

import { AgentDialog } from './agent-dialogs'
import { newAgentTarget } from './new-agent-button'

interface NewAgentDialogProps {
  onClose: () => void
}

/** The New agent dialog for a surface outside the Agents page (the chat sidebar's "+"): the same
 *  dialog and refresh, for the window's own machine. A default export so that surface loads it only
 *  when it is first opened. */
export default function NewAgentDialog({ onClose }: NewAgentDialogProps) {
  // Captured as opened: a window or profile change while it is up must not retarget it.
  const [target] = useState(() => newAgentTarget())

  return <AgentDialog onClose={onClose} target={target} />
}
