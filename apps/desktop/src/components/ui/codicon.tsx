import type { Icon } from '@tabler/icons-react'
import { type HTMLAttributes, memo } from 'react'

import { cn } from '@/lib/utils'

export interface CodiconProps extends HTMLAttributes<HTMLElement> {
  name: string
  size?: number | string
  spinning?: boolean
}

// Memo: assistant footers / tooltips re-render on stream commits; the glyph
// props almost never change, so skipping the host element cuts wasted work
// measured under render-churn (Codicon showed up on the wasted list).
export const Codicon = memo(function Codicon({ className, name, size, spinning, style, ...props }: CodiconProps) {
  return (
    <i
      aria-hidden="true"
      className={cn('codicon', `codicon-${name}`, spinning && 'codicon-modifier-spin', className)}
      style={{ ...(size != null && size !== '' ? { fontSize: size } : {}), ...style }}
      {...props}
    />
  )
})

/** Wrap a codicon as a Tabler-shaped icon for nav rows that expect `IconComponent`. */
export function codiconIcon(name: string): Icon {
  function CodiconIcon({ className }: { className?: string }) {
    return <Codicon aria-hidden className={cn('leading-none', className)} name={name} size="1em" />
  }

  CodiconIcon.displayName = `Codicon(${name})`

  return CodiconIcon as Icon
}
