import { Tip } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

interface McpToolChipProps {
  action: string
  details: string
  enabled: boolean
  onToggle: () => void
  saved: boolean
  toolName: string
}

export function McpToolChip({ action, details, enabled, onToggle, saved, toolName }: McpToolChipProps) {
  const tooltip = details ? `${action}\n\n${details}` : action

  return (
    <Tip label={<span className="whitespace-pre-line">{tooltip}</span>}>
      <button
        aria-label={action}
        aria-pressed={enabled}
        className={cn(
          'rounded-md px-1.5 py-0.5 font-mono text-[0.65rem] text-(--ui-text-tertiary) hover:text-foreground',
          saved ? 'cursor-pointer' : 'cursor-default',
          enabled ? 'bg-(--ui-bg-quinary)' : 'line-through opacity-70'
        )}
        disabled={!saved}
        onClick={onToggle}
        type="button"
      >
        {toolName}
      </button>
    </Tip>
  )
}
