import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { $focusedLeafId } from '@/lib/terminal-store'
import { cn } from '@/lib/utils'

import { addSelectionShortcutLabel } from './selection'
import { useTerminalSession } from './use-terminal-session'

interface TerminalPaneProps {
  cwd: string
  leafId: string
  onClose: () => void
  onFocus: () => void
  onAddSelectionToChat: (text: string, label?: string) => void
}

export function TerminalPane({ cwd, leafId, onClose, onFocus, onAddSelectionToChat }: TerminalPaneProps) {
  const focusedLeafId = useStore($focusedLeafId)
  const focused = focusedLeafId === leafId

  const { addSelectionToChat, focus, hostRef, selection, shellName, status } = useTerminalSession({
    cwd,
    onAddSelectionToChat
  })

  return (
    <div
      className={cn('relative flex min-h-0 min-w-0 flex-1 flex-col', focused && 'ring-1 ring-(--ui-accent)/40')}
      onClick={() => {
        onFocus()
        focus()
      }}
    >
      <div className="flex h-6 shrink-0 items-center gap-1.5 px-2 bg-[#002736]">
        <span className="text-[0.6rem] font-medium text-white/60 truncate">{shellName}</span>
        <div className="ml-auto flex items-center gap-0.5">
          {selection.trim() && (
            <Button
              className="h-4 rounded px-1 text-[0.55rem] shadow-sm backdrop-blur-md"
              onClick={event => event.preventDefault()}
              onMouseDown={event => {
                event.preventDefault()
                event.stopPropagation()
                addSelectionToChat()
              }}
              size="icon-xs"
              type="button"
              variant="secondary"
            >
              {addSelectionShortcutLabel()}
            </Button>
          )}
          <button
            className="rounded p-0.5 text-white/40 transition-colors hover:bg-white/10 hover:text-white/80"
            onClick={event => {
              event.stopPropagation()
              onClose()
            }}
            type="button"
          >
            <Codicon name="close" size="0.625rem" />
          </button>
        </div>
      </div>
      <div className="relative min-h-0 flex-1 flex flex-col bg-[#002b36]">
        {status === 'starting' && (
          <div className="pointer-events-none absolute inset-0 z-10 grid place-items-center">
            <div className="size-4 animate-spin rounded-full border-2 border-white/20 border-t-white/60" />
          </div>
        )}
        <div
          className="absolute inset-0 overflow-hidden text-(--ui-text-secondary) [&_.xterm]:h-full [&_.xterm]:w-full [&_.xterm-screen]:bg-[#002b36]! [&_.xterm-viewport]:bg-[#002b36]!"
          ref={hostRef}
        />
      </div>
    </div>
  )
}
