import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $backgroundStatusBySession, refreshBackgroundProcesses } from '@/store/composer-status'
import { $activeSessionId } from '@/store/session'

import { setActiveTerminalId } from './buffer'
import { AgentTerminalInstance, TerminalInstance } from './instance'
import { $activeTerminalId, $terminals, ensureAgentTerminal } from './terminals'

interface TerminalWorkspaceProps {
  onAddSelectionToChat: (text: string, label?: string) => void
}

// Faster than the 5s status-stack poll so an open agent tab tails near-live.
const AGENT_POLL_MS = 1500

/** The persistent-overlay layer: the stack of live xterm instances (only these
 *  must stay in the fixed overlay, for the WebGL host). Mount/visibility is owned
 *  by PersistentTerminal (latched so shells survive hiding); the tab rail and
 *  new-terminal control live in the pane DOM — see TerminalPaneChrome. */
export function TerminalWorkspace({ onAddSelectionToChat }: TerminalWorkspaceProps) {
  const terminals = useStore($terminals)
  const activeId = useStore($activeTerminalId)
  const activeSession = useStore($activeSessionId)
  const background = useStore($backgroundStatusBySession)

  // Mirror the tab selection into the agent reader (read_terminal reads it).
  useEffect(() => {
    const unsubscribe = $activeTerminalId.subscribe(setActiveTerminalId)

    return () => {
      unsubscribe()
      setActiveTerminalId(null)
    }
  }, [])

  // Surface the agent's background processes as read-only tabs (once each).
  useEffect(() => {
    for (const item of (activeSession && background[activeSession]) || []) {
      ensureAgentTerminal(item.id, item.title)
    }
  }, [background, activeSession])

  // While an agent tab exists, tail its process faster than the status stack.
  const hasAgent = terminals.some(term => term.kind === 'agent')

  useEffect(() => {
    if (!hasAgent || !activeSession) {
      return
    }

    const interval = setInterval(() => void refreshBackgroundProcesses(activeSession), AGENT_POLL_MS)

    return () => clearInterval(interval)
  }, [hasAgent, activeSession])

  return (
    <>
      {terminals.map(term =>
        term.kind === 'agent' ? (
          <AgentTerminalInstance active={term.id === activeId} key={term.id} procId={term.procId!} />
        ) : (
          <TerminalInstance
            active={term.id === activeId}
            cwd={term.cwd}
            id={term.id}
            key={term.id}
            onAddSelectionToChat={onAddSelectionToChat}
          />
        )
      )}
    </>
  )
}
