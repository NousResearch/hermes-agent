import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

import { $multitaskLayout, setMultitaskLayout, type MultitaskLayout } from '@/store/multitask'

const LAYOUTS: { id: MultitaskLayout; label: string; icon: string }[] = [
  { id: 'grid-2', label: '2×2', icon: 'layout-sidebar-left' },
  { id: 'grid-3', label: '3×3', icon: 'layout-sidebar-left' },
  { id: 'horizontal', label: 'Horizontal', icon: 'split-horizontal' },
  { id: 'vertical', label: 'Vertical', icon: 'layout-sidebar-right' }
]

export function LayoutSwitcher({ className }: { className?: string }) {
  const currentLayout = useStore($multitaskLayout)

  return (
    <div className={cn('flex items-center gap-1', className)}>
      {LAYOUTS.map(layout => (
        <Button
          key={layout.id}
          aria-pressed={currentLayout === layout.id}
          className={cn(
            'flex h-7 items-center gap-1 rounded-md px-2 text-[0.75rem] font-medium transition-colors',
            currentLayout === layout.id
              ? 'bg-(--ui-control-active-background) text-foreground'
              : 'text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground'
          )}
          onClick={() => setMultitaskLayout(layout.id)}
          title={layout.label}
          type="button"
          variant="ghost"
        >
          <Codicon name={layout.icon} className="size-3.5" />
          <span className="hidden sm:inline">{layout.label}</span>
        </Button>
      ))}
    </div>
  )
}
