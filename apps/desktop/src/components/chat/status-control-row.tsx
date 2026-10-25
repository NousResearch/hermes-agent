import type { ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { cn } from '@/lib/utils'

import { StatusRow } from './status-row'

interface StatusControlRowProps {
  children: ReactNode
  className?: string
  copyText?: string
  icon?: string
}

/** Text/detail rows share the task icon column without decorating supporting copy. */
export function StatusControlRow({ children, className, copyText, icon }: StatusControlRowProps) {
  return (
    <StatusRow
      className={cn('status-control-row text-[0.7rem] text-muted-foreground/80', className)}
      leading={icon ? <Codicon className="text-muted-foreground/70" name={icon} size="0.8rem" /> : undefined}
      trailing={copyText ? <CopyButton appearance="icon" buttonSize="icon-xs" text={copyText} /> : undefined}
    >
      {children}
    </StatusRow>
  )
}
