import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import {
  DropdownMenuCheckboxItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { Zap, ZapFilled } from '@/lib/icons'
import {
  $approvalModes,
  type ApprovalMode,
  type ApprovalModeRequester,
  setApprovalModeForProfile,
  syncApprovalModeForProfile
} from '@/store/approval-mode'

export interface SessionYoloOptions {
  /** Per-session approval bypass (`/yolo`, ⌘K "Toggle yolo") for the FOCUSED
   *  chat. Independent of the profile-wide approval mode: the zap lights up
   *  for either, so a bypass is never invisible in the status bar. */
  active: boolean
  onToggle: (enabled: boolean) => Promise<void> | void
}

const LIT_CLASS = 'bg-(--chrome-action-hover) text-foreground'

export function useApprovalModeStatusbarItem(
  profile: string,
  requestGateway: ApprovalModeRequester,
  sessionYolo?: SessionYoloOptions
): StatusbarItem {
  const { t } = useI18n()
  const copy = t.shell.approvalMode
  const modes = useStore($approvalModes)
  const mode = modes[profile.trim() || 'default'] ?? 'smart'
  const yoloActive = sessionYolo?.active === true
  const onToggleYolo = sessionYolo?.onToggle

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

  useEffect(() => {
    void syncApprovalModeForProfile(requestGateway, profile).catch(() => undefined)
  }, [profile, requestGateway])

  // Global "off" already bypasses everything; the per-chat flag adds nothing
  // on top of it, so the trigger reads the mode and the row is inert.
  const bypass = mode === 'off' || yoloActive
  const label = mode === 'off' ? labels.off : yoloActive ? copy.sessionYolo : labels[mode]
  const title = mode !== 'off' && yoloActive ? copy.sessionYoloAriaLabel(labels[mode]) : copy.ariaLabel(labels[mode])

  return {
    className: bypass ? LIT_CLASS : undefined,
    icon: bypass ? <ZapFilled className="size-3.5" /> : <Zap className="size-3.5 opacity-70" />,
    id: 'approval-mode',
    label,
    // The pill is hideable from the bar's context menu — but an active bypass
    // IS status: dangerous commands run unasked. Pin it on screen for as long
    // as that holds so a toggled /yolo is never invisible.
    lockedVisible: bypass,
    menuAlign: 'end',
    menuClassName: 'w-72 p-1',
    menuContent: (
      <>
        <DropdownMenuLabel>{copy.title}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuRadioGroup
          onValueChange={value => {
            void setApprovalModeForProfile(requestGateway, profile, value as ApprovalMode).catch(() => undefined)
          }}
          value={mode}
        >
          {(['manual', 'smart', 'off'] as const).map(value => (
            <DropdownMenuRadioItem className="items-start gap-2" key={value} value={value}>
              <span className="flex min-w-0 flex-col gap-0.5">
                <span className="text-xs text-foreground">{labels[value]}</span>
                <span className="text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">{descriptions[value]}</span>
              </span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
        {sessionYolo ? (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuCheckboxItem
              checked={yoloActive}
              className="items-start gap-2"
              disabled={mode === 'off'}
              onCheckedChange={checked => {
                void Promise.resolve(onToggleYolo?.(checked === true)).catch(() => undefined)
              }}
            >
              <span className="flex min-w-0 flex-col gap-0.5">
                <span className="text-xs text-foreground">{copy.sessionYoloRow}</span>
                <span className="text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">
                  {copy.sessionYoloDescription}
                </span>
              </span>
            </DropdownMenuCheckboxItem>
          </>
        ) : null}
      </>
    ),
    title,
    variant: 'menu'
  }
}
