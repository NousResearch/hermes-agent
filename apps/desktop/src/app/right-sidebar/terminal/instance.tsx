import '@xterm/xterm/css/xterm.css'

import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { KbdCombo } from '@/components/ui/kbd'
import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $backgroundOutputByProc } from '@/store/composer-status'

import { reportTerminalShell } from './terminals'
import { useAgentTerminal } from './use-agent-terminal'
import { useTerminalSession } from './use-terminal-session'

// Absolute-stacked so inactive tabs keep layout size (a display:none host goes
// 0×0 and renders garbled on re-show); visibility toggles which one is seen.
const INSTANCE_CLASS = 'absolute inset-0 flex flex-col bg-(--ui-editor-surface-background) px-2 pb-2 pt-0'

interface TerminalInstanceProps {
  id: string
  cwd: string
  active: boolean
  onAddSelectionToChat: (text: string, label?: string) => void
}

/** One persistent xterm+PTY. Every open tab stays mounted (so its shell and
 *  scrollback survive tab switches); only the active one is shown. */
export function TerminalInstance({ id, active, cwd, onAddSelectionToChat }: TerminalInstanceProps) {
  const { t } = useI18n()

  const { addSelectionToChat, hostRef, selection, selectionStyle, status } = useTerminalSession({
    id,
    cwd,
    active,
    onAddSelectionToChat,
    onShell: shell => reportTerminalShell(id, shell)
  })

  return (
    <div
      className={cn(INSTANCE_CLASS, active ? 'visible' : 'invisible pointer-events-none')}
      // Focus-scope marker so isFocusWithin('[data-terminal]') can route ⌘W here.
      data-terminal=""
    >
      {status === 'starting' && (
        <div className="pointer-events-none absolute inset-0 z-10 grid place-items-center">
          <Loader className="size-8 text-(--ui-text-tertiary)" pathSteps={180} strokeScale={0.68} type="spiral-search" />
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
            {t.rightSidebar.addToChat}
            <KbdCombo className="ml-1 opacity-70" combo="mod+l" size="sm" />
          </Button>
        </div>
      )}
      {/* Outer div paints the terminal inset; inner div is the xterm host so the
          canvas sizes to the content area and p-2 stays as terminal padding. */}
      <div
        className="h-full min-h-0 overflow-hidden text-(--ui-text-secondary) [&_.xterm]:h-full [&_.xterm-screen]:bg-(--ui-editor-surface-background)! [&_.xterm-viewport]:bg-(--ui-editor-surface-background)!"
        ref={hostRef}
      />
    </div>
  )
}

interface AgentTerminalInstanceProps {
  active: boolean
  procId: string
}

/** Read-only mirror of an agent background process — a write-only xterm fed by
 *  the process's output tail (no PTY, no input). */
export function AgentTerminalInstance({ active, procId }: AgentTerminalInstanceProps) {
  const output = useStore($backgroundOutputByProc)[procId] ?? ''
  const { hostRef } = useAgentTerminal({ active, output })

  return (
    <div className={cn(INSTANCE_CLASS, active ? 'visible' : 'invisible pointer-events-none')}>
      <div
        className="h-full min-h-0 overflow-hidden text-(--ui-text-secondary) [&_.xterm]:h-full [&_.xterm-screen]:bg-(--ui-editor-surface-background)! [&_.xterm-viewport]:bg-(--ui-editor-surface-background)!"
        ref={hostRef}
      />
    </div>
  )
}
