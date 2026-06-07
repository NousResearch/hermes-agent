import { Clock, Copy } from '@/lib/icons'
import { notify } from '@/store/notifications'

import { CheckCircle2, AlertCircle } from '@/lib/icons'

interface TerminalHeaderProps {
  command?: string
  duration?: string
  exitCode?: number
  lineCount?: number
}

export function TerminalHeader({ command, duration, exitCode, lineCount }: TerminalHeaderProps) {
  const handleCopyCommand = () => {
    if (command) {
      void navigator.clipboard.writeText(command)
      notify({ kind: 'success', title: 'Copied', message: 'Command copied to clipboard.' })
    }
  }

  return (
    <div className="flex items-center gap-2 px-2 py-1 text-[0.65rem] text-muted-foreground">
      {command && (
        <span className="truncate font-mono text-foreground">{command}</span>
      )}
      {duration && (
        <span className="flex shrink-0 items-center gap-0.5">
          <Clock size={10} />
          {duration}
        </span>
      )}
      {exitCode !== undefined && (
        <span className="flex shrink-0 items-center gap-0.5">
          {exitCode === 0 ? (
            <CheckCircle2 size={10} className="text-emerald-500" />
          ) : (
            <AlertCircle size={10} className="text-rose-500" />
          )}
          <span className={exitCode === 0 ? 'text-emerald-500' : 'text-rose-500'}>
            {exitCode}
          </span>
        </span>
      )}
      {lineCount !== undefined && (
        <span className="shrink-0">{lineCount} lines</span>
      )}
      {command && (
        <button
          className="ml-auto shrink-0 rounded p-0.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          onClick={handleCopyCommand}
          title="Copy command"
          type="button"
        >
          <Copy size={10} />
        </button>
      )}
    </div>
  )
}
