import './drawer.css'

import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'

interface StatusDrawerContentProps {
  children: ReactNode
  collapsed: boolean
  id: string
}

/** Keep the live sections mounted: hiding the drawer must not reset their disclosures. */
export function StatusDrawerContent({ children, collapsed, id }: StatusDrawerContentProps) {
  return (
    <div
      aria-hidden={collapsed || undefined}
      className="status-drawer"
      data-collapsed={collapsed ? '' : undefined}
      id={id}
      inert={collapsed}
    >
      <div className="status-drawer-clip">
        <div className="status-drawer-content">{children}</div>
      </div>
    </div>
  )
}

interface StatusDrawerToggleProps {
  collapsed: boolean
  controls: string
  onToggle: () => void
}

export function StatusDrawerToggle({ collapsed, controls, onToggle }: StatusDrawerToggleProps) {
  const { t } = useI18n()
  const label = collapsed ? t.statusStack.showStack : t.statusStack.hideStack

  return (
    <Tip label={label}>
      <Button
        aria-controls={controls}
        aria-expanded={!collapsed}
        aria-label={label}
        className="status-drawer-toggle absolute -top-2 left-1/2 z-10 -translate-x-1/2"
        data-slot="status-drawer-toggle"
        onClick={onToggle}
        size="grip"
        type="button"
        variant="grip"
      >
        <span aria-hidden className="h-0.75 w-7 rounded-full bg-current" />
      </Button>
    </Tip>
  )
}
