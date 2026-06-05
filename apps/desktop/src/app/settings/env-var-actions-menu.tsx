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
import { useTranslation } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { ExternalLink, Eye, EyeOff, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

interface EnvVarActionsMenuProps
  extends Pick<React.ComponentProps<typeof DropdownMenuContent>, 'align' | 'sideOffset'> {
  children: React.ReactNode
  clearDisabled?: boolean
  docsUrl?: string | null
  isRevealed?: boolean
  isSet: boolean
  label: string
  onClear?: () => void
  onEdit: () => void
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
  onReveal,
  showReveal = true,
  sideOffset = 6
}: EnvVarActionsMenuProps) {
  const t = useTranslation()
  const hasClear = isSet && onClear
  const hasReveal = isSet && showReveal && onReveal
  const hasDocs = Boolean(docsUrl?.trim())

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
      <DropdownMenuContent
        align={align}
        aria-label={t('settings.keys.actions.actionsFor', { label })}
        className="w-44"
        sideOffset={sideOffset}
      >
        {hasDocs && (
          <DropdownMenuItem
            onSelect={event => {
              event.preventDefault()
              triggerHaptic('selection')
              window.open(docsUrl!, '_blank', 'noopener,noreferrer')
            }}
          >
            <ExternalLink className="size-3.5" />
            <span>{t('settings.keys.actions.docs')}</span>
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
            <span>
              {isRevealed ? t('settings.keys.actions.hideValue') : t('settings.keys.actions.revealValue')}
            </span>
          </DropdownMenuItem>
        )}

        <DropdownMenuItem
          onSelect={() => {
            triggerHaptic('selection')
            onEdit()
          }}
        >
          <Codicon name="edit" size="0.875rem" />
          <span>{isSet ? t('settings.keys.actions.replace') : t('settings.keys.actions.set')}</span>
        </DropdownMenuItem>

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
              <span>{t('settings.keys.actions.clear')}</span>
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
  const t = useTranslation()

  return (
    <Button
      aria-label={t('settings.keys.actions.actionsFor', { label })}
      className={cn('text-muted-foreground hover:text-foreground', className)}
      size="icon-sm"
      title={t('settings.keys.actions.credentialActions')}
      variant="ghost"
      {...props}
    >
      <Codicon name="ellipsis" size="0.875rem" />
    </Button>
  )
}
