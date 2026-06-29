import '@xterm/xterm/css/xterm.css'

import { Button } from '@/components/ui/button'
import { KbdCombo } from '@/components/ui/kbd'
import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { reportTerminalShell } from './terminals'
import { useTerminalSession } from './use-terminal-session'

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
      className={cn(
        // Stack every terminal absolutely and toggle visibility (NOT display) so
        // inactive tabs keep their layout size and track pane resizes — a
        // display:none host goes 0×0, skips fit, and renders garbled when shown
        // again at a changed size. No top padding so the prompt hugs the
        // titlebar-clearance line (the rest of the gap is required clearance).
        'absolute inset-0 flex flex-col bg-(--ui-editor-surface-background) px-2 pb-2 pt-0',
        active ? 'visible' : 'invisible pointer-events-none'
      )}
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
