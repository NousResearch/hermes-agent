import type * as React from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { ExternalLink, Eye, EyeOff, KeyRound, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

interface EnvVarActionsMenuProps extends Pick<
  React.ComponentProps<typeof DropdownMenuContent>,
  'align' | 'sideOffset'
> {
  children: React.ReactNode
  clearDisabled?: boolean
  docsUrl?: string | null
  isRevealed?: boolean
  isSet: boolean
  label: string
  onClear?: () => void
  onEdit: () => void
  /** Internal navigation to Settings → API Keys with this key highlighted.
   *  Rendered only when provided AND the key is set (an unset key is managed
   *  right here via Set). */
  onManageKeys?: () => void
  onReveal?: () => void
  showReveal?: boolean
}

export function EnvVarActionsMenu({
  align = 'end',
  children,
  clearDisabled = false,
  docsUrl,
  isRevealed = false,
  isSet,
  label,
  onClear,
  onEdit,
  onManageKeys,
  onReveal,
  showReveal = true,
  sideOffset = 6
}: EnvVarActionsMenuProps) {
  const { t } = useI18n()
  const copy = t.settings.envActions
  const hasClear = isSet && onClear
  const hasReveal = isSet && showReveal && onReveal
  const hasManageKeys = isSet && onManageKeys
  const hasDocs = Boolean(docsUrl?.trim())

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
      <DropdownMenuContent align={align} aria-label={copy.actionsFor(label)} className="w-44" sideOffset={sideOffset}>
        {hasDocs && (
          <DropdownMenuItem
            onSelect={event => {
              event.preventDefault()
              triggerHaptic('selection')
              window.open(docsUrl!, '_blank', 'noopener,noreferrer')
            }}
          >
            <ExternalLink className="size-3.5" />
            <span>{copy.docs}</span>
          </DropdownMenuItem>
        )}

        {hasReveal && (
          <DropdownMenuItem
            onSelect={() => {
              triggerHaptic('selection')
              onReveal()
            }}
          >
            {isRevealed ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
            <span>{isRevealed ? copy.hideValue : copy.revealValue}</span>
          </DropdownMenuItem>
        )}

        <DropdownMenuItem
          onSelect={() => {
            triggerHaptic('selection')
            onEdit()
          }}
        >
          <Codicon name="edit" size="0.875rem" />
          <span>{isSet ? copy.replace : copy.set}</span>
        </DropdownMenuItem>

        {hasManageKeys && (
          <DropdownMenuItem
            onSelect={() => {
              triggerHaptic('selection')
              onManageKeys()
            }}
          >
            <KeyRound className="size-3.5" />
            <span>{copy.manageInKeys}</span>
          </DropdownMenuItem>
        )}

        {hasClear && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              disabled={clearDisabled}
              onSelect={() => {
                triggerHaptic('warning')
                onClear()
              }}
              variant="destructive"
            >
              <Trash2 className="size-3.5" />
              <span>{copy.clear}</span>
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

interface EnvVarActionsTriggerProps extends Omit<React.ComponentProps<typeof Button>, 'size' | 'variant'> {
  label: string
}

export function EnvVarActionsTrigger({ className, label, ...props }: EnvVarActionsTriggerProps) {
  const { t } = useI18n()
  const copy = t.settings.envActions

  return (
    <Tip label={copy.credentialActions}>
      <Button
        aria-label={copy.actionsFor(label)}
        className={cn('text-muted-foreground hover:text-foreground', className)}
        size="icon-sm"
        variant="ghost"
        {...props}
      >
        <Codicon name="ellipsis" size="0.875rem" />
      </Button>
    </Tip>
  )
}
