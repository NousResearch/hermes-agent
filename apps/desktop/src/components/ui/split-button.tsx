import type { VariantProps } from 'class-variance-authority'
import type { ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

import type { buttonVariants } from './button';
import { Button } from './button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from './dropdown-menu'

export interface SplitButtonAction {
  id: string
  label: string
  icon?: ReactNode
}

interface SplitButtonProps {
  actions: SplitButtonAction[]
  /** The id of the action the primary button runs (the user's current default). */
  value: string
  /** Picking from the menu changes the default (so the next primary click repeats it). */
  onValueChange: (id: string) => void
  /** Run an action by id (primary click or menu pick both call this). */
  onTrigger: (id: string) => void
  disabled?: boolean
  className?: string
  /** Icon shown on the primary button only (e.g. a ✓ for Commit). */
  primaryIcon?: ReactNode
  variant?: VariantProps<typeof buttonVariants>['variant']
  size?: VariantProps<typeof buttonVariants>['size']
}

/**
 * A primary action fused to a caret that opens alternates — VS Code's
 * Commit / Commit & Push pattern. The primary button runs `value`; picking a
 * menu item runs it AND makes it the new default, so the control adapts to how
 * the user works without a separate settings toggle.
 */
export function SplitButton({
  actions,
  value,
  onValueChange,
  onTrigger,
  disabled,
  className,
  primaryIcon,
  variant = 'secondary',
  size = 'sm'
}: SplitButtonProps) {
  const active = actions.find(action => action.id === value) ?? actions[0]

  if (!active) {
    return null
  }

  return (
    <div className={cn('inline-flex min-w-0', className)}>
      <Button
        className="min-w-0 flex-1 rounded-r-none"
        disabled={disabled}
        onClick={() => onTrigger(active.id)}
        size={size}
        variant={variant}
      >
        {primaryIcon ?? active.icon}
        <span className="truncate">{active.label}</span>
      </Button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label="More actions"
            className="rounded-l-none border-l border-current/25 px-2"
            disabled={disabled}
            size={size}
            variant={variant}
          >
            <Codicon name="chevron-down" size="0.8rem" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-44">
          {actions.map(action => (
            <DropdownMenuItem
              key={action.id}
              onSelect={() => {
                onValueChange(action.id)
                onTrigger(action.id)
              }}
            >
              {action.icon}
              <span className="flex-1 truncate">{action.label}</span>
              {action.id === value && <Codicon className="text-(--ui-text-tertiary)" name="check" size="0.75rem" />}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
