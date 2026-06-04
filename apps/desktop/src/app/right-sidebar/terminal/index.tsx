import '@xterm/xterm/css/xterm.css'

import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Loader } from '@/components/ui/loader'

import { SidebarPanelLabel } from '../../shell/sidebar-label'
import { $terminalTakeover, setRightSidebarTab, setTerminalTakeover } from '../store'

import { addSelectionShortcutLabel } from './selection'
import { useTerminalSession } from './use-terminal-session'

interface TerminalTabProps {
  cwd: string
  onAddSelectionToChat: (text: string, label?: string) => void
}

export function TerminalTab({ cwd, onAddSelectionToChat }: TerminalTabProps) {
  const { addSelectionToChat, hostRef, selection, selectionStyle, shellName, status } = useTerminalSession({
    cwd,
    onAddSelectionToChat
  })

  const takeover = useStore($terminalTakeover)
  const label = takeover ? 'Return to split view' : 'Focus terminal view'

  const toggleTakeover = () => {
    // Pre-select the Terminal tab so the slot is ready to host us on return.
    if (takeover) {
      setRightSidebarTab('terminal')
    }
    setTerminalTakeover(!takeover)
  }

  return (
    <div className="relative flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex h-8 shrink-0 items-center gap-2 px-2.5">
        <SidebarPanelLabel className="text-white!">{shellName}</SidebarPanelLabel>
        <Button
          aria-label={label}
          className="ml-auto size-6 rounded-md text-white!"
          onClick={toggleTakeover}
          size="icon"
          title={label}
          type="button"
          variant="ghost"
        >
          <Codicon name={takeover ? 'screen-normal' : 'screen-full'} size="0.875rem" />
        </Button>
      </div>
      <div className="relative min-h-0 flex-1 bg-[#002b36] p-2">
        {status === 'starting' && (
          <div className="pointer-events-none absolute inset-0 z-10 grid place-items-center">
            <Loader
              className="size-8 text-(--ui-text-tertiary)"
              pathSteps={180}
              strokeScale={0.68}
              type="spiral-search"
            />
          </div>
        )}
        {selection.trim() && (
          <div className="absolute z-50 flex items-center gap-1" style={selectionStyle ?? { right: 12, top: 8 }}>
            <Button
              className="h-6 rounded-md px-2 text-[0.68rem] shadow-md backdrop-blur-md"
              onClick={event => event.preventDefault()}
              onMouseDown={event => {
                event.preventDefault()
                event.stopPropagation()
                addSelectionToChat()
              }}
              type="button"
              variant="secondary"
            >
              Add to chat
              <span className="ml-1 text-[0.6rem] text-(--ui-text-tertiary)">{addSelectionShortcutLabel()}</span>
            </Button>
          </div>
        )}
        {/* Outer div paints the dark inset; inner div is the xterm host so the
            canvas sizes to the *content* area and p-2 shows as terminal padding.
            Forcing screen/viewport bg avoids xterm's default black peeking
            through the unused pixels below the last full row. */}
        <div
          className="h-full min-h-0 overflow-hidden text-(--ui-text-secondary) [&_.xterm]:h-full [&_.xterm-screen]:bg-[#002b36]! [&_.xterm-viewport]:bg-[#002b36]!"
          ref={hostRef}
        />
      </div>
    </div>
  )
}
