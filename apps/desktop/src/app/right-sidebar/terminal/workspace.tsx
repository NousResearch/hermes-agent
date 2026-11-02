import { useStore } from '@nanostores/react'
import { type CSSProperties, useEffect } from 'react'

import { $backgroundStatusBySession } from '@/store/composer-status'

import { seedAgentTerminalCommand, syncAgentTerminalSnapshot } from './agent-terminal-stream'
import { setActiveTerminalId } from './buffer'
import { AgentTerminalInstance, TerminalInstance } from './instance'
import { $activeTerminalId, $splitTerminalId, $terminals, ensureAgentTerminal } from './terminals'

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
  const splitId = useStore($splitTerminalId)
  const background = useStore($backgroundStatusBySession)

  const isSplit = Boolean(
    splitId &&
    activeId &&
    splitId !== activeId &&
    terminals.some(t => t.id === splitId) &&
    terminals.some(t => t.id === activeId)
  )

  const splitTerminals = isSplit ? terminals.filter(t => t.id === activeId || t.id === splitId).slice(0, 2) : []

  const leftId = splitTerminals[0]?.id
  const rightId = splitTerminals[1]?.id

  // Mirror the tab selection into the agent reader (read_terminal reads it).
  useEffect(() => {
    const unsubscribe = $activeTerminalId.subscribe(setActiveTerminalId)

    return () => {
      unsubscribe()
      setActiveTerminalId(null)
    }
  }, [])

  // Surface the agent's background processes as read-only tabs (once each).
  // Live chunks stream via agent.terminal.output; the process-list snapshot also
  // seeds/falls back so the tab never stays blank if the stream races startup.
  useEffect(() => {
    for (const list of Object.values(background)) {
      for (const item of list) {
        ensureAgentTerminal(item.id, item.title)
        seedAgentTerminalCommand(item.id, item.title)
        syncAgentTerminalSnapshot(item.id, item.output ?? '')
      }
    }
  }, [background])

  return (
    <>
      {terminals.map(term => {
        const isActive = term.id === activeId
        let visible = isActive
        let style: CSSProperties | undefined

        if (isSplit) {
          const topBorder = isActive ? '2px solid var(--ui-stroke-primary)' : '2px solid transparent'

          if (term.id === leftId) {
            visible = true
            style = {
              left: 0,
              width: 'calc(50% - 1px)',
              right: 'auto',
              borderRight: '1px solid var(--ui-stroke-secondary)',
              borderTop: topBorder
            }
          } else if (term.id === rightId) {
            visible = true
            style = {
              left: '50%',
              width: '50%',
              right: 'auto',
              borderTop: topBorder
            }
          }
        }

        return term.kind === 'agent' ? (
          <AgentTerminalInstance
            active={isActive}
            id={term.id}
            key={term.id}
            procId={term.procId!}
            style={style}
            visible={visible}
          />
        ) : (
          <TerminalInstance
            active={isActive}
            cwd={term.cwd}
            id={term.id}
            key={term.id}
            onAddSelectionToChat={onAddSelectionToChat}
            restoreCwd={term.restoreCwd}
            reviveBuffer={term.reviveBuffer}
            style={style}
            visible={visible}
          />
        )
      })}
    </>
  )
}
