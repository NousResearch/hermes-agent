import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

/** Titlebar-cluster glyph, sized by `--titlebar-icon-size` (13.9px; the redesign's chrome uses 16px). */
export function TitlebarIcon({
  className,
  name,
  size = 'var(--titlebar-icon-size)',
  spinning
}: {
  className?: string
  name: string
  size?: number | string
  spinning?: boolean
}) {
  return <Codicon className={cn('leading-none', className)} name={name} size={size} spinning={spinning} />
}
