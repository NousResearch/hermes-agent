import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { FolderOpen } from '@/lib/icons'
import { cn } from '@/lib/utils'

const PILL = cn(
  'h-(--composer-control-size) max-w-40 shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground',
  // Divider separating the folder read-out from the action buttons to its left.
  'border-l border-(--ui-stroke-tertiary) pl-(--composer-control-gap)'
)

/** Basename of a path — the last segment after the final `/` (or `\` on
 *  Windows). Returns the original string when it has no separator so a bare
 *  folder name is shown verbatim. */
export function folderName(cwd: string): string {
  const trimmed = cwd.trim()

  if (!trimmed) {
    return ''
  }

  // Handle both POSIX and Windows separators so the pill reads correctly
  // regardless of the gateway's OS.
  const segments = trimmed.split(/[/\\]/).filter(Boolean)

  return segments[segments.length - 1] ?? trimmed
}

/**
 * Composer folder indicator — sits at the far right of the controls row,
 * separated from the action buttons by a divider. Shows the working
 * directory's basename (the folder name) with the full path in a tooltip.
 * Hidden entirely when no cwd is set.
 */
export function CwdPill({ cwd }: { cwd: null | string | undefined }) {
  const name = folderName(cwd ?? '')

  if (!name) {
    return null
  }

  return (
    <Tip label={cwd ?? name}>
      <Button className={PILL} disabled type="button" variant="ghost">
        <FolderOpen className="size-3 shrink-0" />
        <span className="truncate">{name}</span>
      </Button>
    </Tip>
  )
}
