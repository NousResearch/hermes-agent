import * as React from 'react'

import { cn } from '@/lib/utils'

function Kbd({ className, ...props }: React.ComponentProps<'kbd'>) {
  return (
    <kbd
      className={cn(
        'inline-grid h-4 min-w-4 place-items-center rounded-sm border border-border/70 bg-muted/45 px-1 font-mono text-[0.5625rem] font-medium leading-none text-muted-foreground shadow-xs',
        className
      )}
      data-slot="kbd"
      {...props}
    />
  )
}

interface KbdGroupProps extends Omit<React.ComponentProps<'span'>, 'children'> {
  keys: string[]
  size?: 'sm'
}

function KbdGroup({ className, keys, size, ...props }: KbdGroupProps) {
  const keyClassName = size === 'sm' ? 'h-3.5 min-w-3.5 px-0.5 text-[0.5rem]' : undefined

  return (
    <span
      aria-label={keys.join(' ')}
      className={cn('inline-flex shrink-0 items-center gap-0.5 opacity-55', className)}
      data-slot="kbd-group"
      {...props}
    >
      {keys.map(key => (
        <Kbd className={keyClassName} key={key}>
          {key}
        </Kbd>
      ))}
    </span>
  )
}

export { Kbd, KbdGroup }
