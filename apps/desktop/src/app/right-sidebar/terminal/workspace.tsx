import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $backgroundStatusBySession } from '@/store/composer-status'

import { setActiveTerminalId } from './buffer'
import { AgentTerminalInstance, TerminalInstance } from './instance'
import { $activeTerminalId, $terminals, ensureAgentTerminal } from './terminals'

interface TerminalWorkspaceProps {
  onAddSelectionToChat: (text: string, label?: string) => void
}

/** The persistent-overlay layer: the stack of live xterm instances (only these
 *  must stay in the fixed overlay, for the WebGL host). Mount/visibility is owned
 *  by PersistentTerminal (latched so shells survive hiding); the tab rail and
 *  new-terminal control live in the pane DOM — see TerminalPaneChrome. */
export function TerminalWorkspace({ onAddSelectionToChat }: TerminalWorkspaceProps) {
  const terminals = useStore($terminals)
  const activeId = useStore($activeTerminalId)
  const background = useStore($backgroundStatusBySession)

  // Mirror the tab selection into the agent reader (read_terminal reads it).
  useEffect(() => {
    const unsubscribe = $activeTerminalId.subscribe(setActiveTerminalId)

    return () => {
      unsubscribe()
      setActiveTerminalId(null)
    }
  }, [])

  // Surface the agent's background processes as read-only tabs (once each); their
  // output streams in live via agent.terminal.output, no polling needed.
  useEffect(() => {
    for (const list of Object.values(background)) {
      for (const item of list) {
        ensureAgentTerminal(item.id, item.title)
      }
    }
  }, [background])

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
