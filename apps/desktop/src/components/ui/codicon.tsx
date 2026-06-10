import type * as React from 'react'

import { cn } from '@/lib/utils'

export interface CodiconProps extends React.HTMLAttributes<HTMLElement> {
  /** Thickens outline glyphs so they read as filled at small sizes (tool rows). */
  filled?: boolean
  name: string
  size?: number | string
  spinning?: boolean
}

export function Codicon({ className, filled, name, size, spinning, style, ...props }: CodiconProps) {
  return (
    <i
      aria-hidden="true"
      className={cn(
        'codicon',
        `codicon-${name}`,
        spinning && 'codicon-modifier-spin',
        filled && 'codicon-modifier-filled',
        className
      )}
      style={{ fontSize: size, ...style }}
      {...props}
    />
  )
}
