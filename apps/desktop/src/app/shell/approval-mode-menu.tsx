import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo } from 'react'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import {
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { AlertCircle, Lock, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $approvalModes,
  type ApprovalMode,
  type ApprovalModeRequester,
  setApprovalModeForProfile,
  syncApprovalModeForProfile
} from '@/store/approval-mode'

const APPROVAL_MODE_VALUES: readonly ApprovalMode[] = ['manual', 'smart', 'off']

/** Approval modes carry a small, consistent visual signal everywhere they
 * appear. The warm full-access treatment deliberately reads as elevated access
 * rather than an error, while manual and smart modes remain easy to scan. */
export const approvalModeToneClass: Record<ApprovalMode, string> = {
  manual: 'text-sky-600 dark:text-sky-400',
  smart: 'text-emerald-600 dark:text-emerald-400',
  off: 'text-amber-600 dark:text-amber-400'
}

const approvalModeInteractiveToneClass: Record<ApprovalMode, string> = {
  manual: 'text-sky-600 hover:text-sky-600 dark:text-sky-400 dark:hover:text-sky-400',
  smart: 'text-emerald-600 hover:text-emerald-600 dark:text-emerald-400 dark:hover:text-emerald-400',
  off: 'text-amber-600 hover:text-amber-600 dark:text-amber-400 dark:hover:text-amber-400'
}

export interface ApprovalModeMenuState {
  ariaLabel: string
  descriptions: Record<ApprovalMode, string>
  labels: Record<ApprovalMode, string>
  mode: ApprovalMode
  setMode: (mode: ApprovalMode) => void
  title: string
}

export function ApprovalModeGlyph({ className, mode }: { className?: string; mode: ApprovalMode }) {
  const glyphClassName = cn('shrink-0', approvalModeToneClass[mode], className)

  if (mode === 'manual') {
    return <Lock aria-hidden className={glyphClassName} />
  }

  if (mode === 'off') {
    return <AlertCircle aria-hidden className={glyphClassName} />
  }

  return <Zap aria-hidden className={glyphClassName} />
}

/** Shared approval-mode state and copy for any menu trigger. Keeping it here
 * means the compact composer control and the status-bar control cannot drift
 * in their labels, selected state, or profile-scoped gateway writes. */
export function useApprovalModeMenu(
  profile: string,
  requestGateway: ApprovalModeRequester,
  { sync = true }: { sync?: boolean } = {}
): ApprovalModeMenuState {
  const { t } = useI18n()
  const copy = t.shell.approvalMode
  const modes = useStore($approvalModes)
  const mode = modes[profile.trim() || 'default'] ?? 'smart'

  const labels = useMemo<Record<ApprovalMode, string>>(
    () => ({ manual: copy.manual, smart: copy.smart, off: copy.off }),
    [copy.manual, copy.off, copy.smart]
  )

  const descriptions = useMemo<Record<ApprovalMode, string>>(
    () => ({
      manual: copy.manualDescription,
      smart: copy.smartDescription,
      off: copy.offDescription
    }),
    [copy.manualDescription, copy.offDescription, copy.smartDescription]
  )

  const setMode = useCallback(
    (value: ApprovalMode) => {
      void setApprovalModeForProfile(requestGateway, profile, value).catch(() => undefined)
    },
    [profile, requestGateway]
  )

  useEffect(() => {
    if (!sync) {
      return
    }

    void syncApprovalModeForProfile(requestGateway, profile).catch(() => undefined)
  }, [profile, requestGateway, sync])

  return {
    ariaLabel: copy.ariaLabel(labels[mode]),
    descriptions,
    labels,
    mode,
    setMode,
    title: copy.title
  }
}

export function ApprovalModeMenuContent({ menu }: { menu: ApprovalModeMenuState }) {
  return (
    <>
      <DropdownMenuLabel>{menu.title}</DropdownMenuLabel>
      <DropdownMenuSeparator />
      <DropdownMenuRadioGroup onValueChange={value => menu.setMode(value as ApprovalMode)} value={menu.mode}>
        {APPROVAL_MODE_VALUES.map(value => (
          <DropdownMenuRadioItem className="items-start gap-2.5" data-approval-mode={value} key={value} value={value}>
            <ApprovalModeGlyph className="mt-0.5 size-3.5" mode={value} />
            <span className="flex min-w-0 flex-col gap-0.5">
              <span className={cn('text-xs font-medium', approvalModeToneClass[value])}>{menu.labels[value]}</span>
              <span className="text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">
                {menu.descriptions[value]}
              </span>
            </span>
          </DropdownMenuRadioItem>
        ))}
      </DropdownMenuRadioGroup>
    </>
  )
}

export function useApprovalModeStatusbarItem(profile: string, requestGateway: ApprovalModeRequester): StatusbarItem {
  const menu = useApprovalModeMenu(profile, requestGateway)
  const fullAccess = menu.mode === 'off'

  return {
    className: cn(approvalModeInteractiveToneClass[menu.mode], fullAccess && 'bg-amber-500/10 hover:bg-amber-500/15'),
    icon: <ApprovalModeGlyph className="size-3.5" mode={menu.mode} />,
    id: 'approval-mode',
    label: menu.labels[menu.mode],
    menuAlign: 'end',
    menuClassName: 'w-72 p-1',
    menuContent: <ApprovalModeMenuContent menu={menu} />,
    title: menu.ariaLabel,
    variant: 'menu'
  }
}
