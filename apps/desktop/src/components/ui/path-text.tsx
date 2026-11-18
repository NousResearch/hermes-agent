import { Tip } from '@/components/ui/tooltip'
import { displayProfilePath } from '@/lib/display-path'
import { cn } from '@/lib/utils'

interface PathTextProps {
  path: string
  className?: string
}

/** A filesystem path on ONE line: home-relative, the middle elided so the folder name at the end
 *  stays readable, the full path in a tooltip. Mono, because body faces kern `/.` into one glyph. */
export function PathText({ path, className }: PathTextProps) {
  const shown = displayProfilePath(path)
  const cut = shown.lastIndexOf('/')
  const head = cut > 0 ? shown.slice(0, cut) : ''
  const tail = cut > 0 ? shown.slice(cut) : shown

  return (
    <Tip label={path}>
      <span className={cn('inline-flex min-w-0 font-mono whitespace-nowrap', className)} data-slot="path-text">
        {head && <span className="min-w-0 truncate">{head}</span>}
        <span className="shrink-0">{tail}</span>
      </span>
    </Tip>
  )
}
