import type { RefObject } from 'react'

import { Button } from '@/components/ui/button'
import { Activity, CheckCircle2, Clock, Wrench, X } from '@/lib/icons'
import { cn } from '@/lib/utils'

import type { JarvisEvent } from './types'

interface ActivityPanelProps {
  className?: string
  closeButtonRef?: RefObject<HTMLButtonElement | null>
  copy: {
    empty: string
    close: string
    title: string
    types: Record<string, string>
  }
  events: readonly JarvisEvent[]
  id?: string
  labelledBy?: string
  onClose?: () => void
  surface?: 'bottom-sheet' | 'drawer' | 'panel'
}

function activityLabel(event: JarvisEvent, copy: ActivityPanelProps['copy']): string {
  return event.label?.trim() || copy.types[event.type] || event.type
}

function activityDetail(event: JarvisEvent): string | null {
  return event.detail?.trim() || null
}

function ActivityIcon({ type }: { type: string }) {
  if (type === 'tool.completed' || type === 'task.verified') {
    return <CheckCircle2 className="size-4 text-(--ui-accent)" />
  }

  if (type.startsWith('tool.')) {
    return <Wrench className="size-4 text-(--ui-text-tertiary)" />
  }

  return <Clock className="size-4 text-(--ui-text-tertiary)" />
}

export function JarvisActivityPanel({
  className,
  closeButtonRef,
  copy,
  events,
  id,
  labelledBy,
  onClose,
  surface = 'panel'
}: ActivityPanelProps) {
  const visibleEvents = events.slice(-50).reverse()
  const overlay = surface !== 'panel'

  return (
    <aside
      aria-label={copy.title}
      aria-labelledby={labelledBy}
      className={cn(
        'flex min-h-0 flex-col bg-(--ui-chat-surface-background) text-(--ui-text-primary)',
        surface === 'panel' && 'border-l border-(--ui-stroke-tertiary)',
        surface !== 'panel' && 'shadow-nous border border-(--stroke-nous)',
        className
      )}
      data-activity-surface={surface}
      id={id}
      role={overlay ? 'dialog' : 'complementary'}
    >
      <div className="flex min-h-11 items-center gap-2 px-4 py-3">
        <Activity className="size-4 text-(--ui-accent)" />
        <h2 className="min-w-0 flex-1 text-sm font-semibold" id={labelledBy}>
          {copy.title}
        </h2>
        {onClose && (
          <Button
            aria-label={copy.close}
            className="min-h-11 min-w-11"
            onClick={onClose}
            ref={closeButtonRef}
            size="icon"
            type="button"
            variant="ghost"
          >
            <X />
          </Button>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-auto px-4 pb-4">
        {visibleEvents.length === 0 ? (
          <p className="py-4 text-sm text-(--ui-text-secondary)">{copy.empty}</p>
        ) : (
          <ol className="space-y-3">
            {visibleEvents.map((event, index) => {
              const detail = activityDetail(event)

              return (
                <li className="grid grid-cols-[1rem_1fr] gap-3 text-sm" key={`${event.at}:${event.type}:${index}`}>
                  <ActivityIcon type={event.type} />
                  <div className="min-w-0">
                    <div className="truncate font-medium text-(--ui-text-primary)">{activityLabel(event, copy)}</div>
                    {detail && <div className="truncate text-xs text-(--ui-text-secondary)">{detail}</div>}
                  </div>
                </li>
              )
            })}
          </ol>
        )}
      </div>
    </aside>
  )
}
