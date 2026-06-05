import type { ReactNode } from 'react'

import { Check, Loader2 } from '@/lib/icons'

// idle → saving → done label+icon for action buttons (create / rename / delete…).
export function ActionStatus({
  state,
  idle,
  busy,
  done,
  idleIcon = null
}: {
  state: 'done' | 'idle' | 'saving'
  idle: string
  busy: string
  done: string
  idleIcon?: ReactNode
}) {
  return (
    <>
      {state === 'saving' ? <Loader2 className="size-4 animate-spin" /> : state === 'done' ? <Check /> : idleIcon}
      {state === 'saving' ? busy : state === 'done' ? done : idle}
    </>
  )
}
