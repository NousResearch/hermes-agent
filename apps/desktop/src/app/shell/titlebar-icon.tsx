import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

import { titlebarIconSizeCss } from './titlebar'

/** Titlebar-cluster glyph — 13.9px inline (see titlebarIconSizeCss). */
export function TitlebarIcon({
  className,
  name,
  size = titlebarIconSizeCss(),
  spinning
}: {
  className?: string
  name: string
  size?: number | string
  spinning?: boolean
}) {
  return <Codicon className={cn('leading-none', className)} name={name} size={size} spinning={spinning} />
}
