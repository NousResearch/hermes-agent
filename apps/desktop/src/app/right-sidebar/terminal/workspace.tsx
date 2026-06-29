import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { setActiveTerminalId } from './buffer'
import { TerminalInstance } from './instance'
import { $activeTerminalId, $terminals } from './terminals'

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

  // Mirror the tab selection into the agent reader (read_terminal reads it).
  useEffect(() => {
    const unsubscribe = $activeTerminalId.subscribe(setActiveTerminalId)

    return () => {
      unsubscribe()
      setActiveTerminalId(null)
    }
  }, [])

  return (
    <>
      {terminals.map(term => (
        <TerminalInstance
          active={term.id === activeId}
          cwd={term.cwd}
          id={term.id}
          key={term.id}
          onAddSelectionToChat={onAddSelectionToChat}
        />
      ))}
    </>
  )
}
