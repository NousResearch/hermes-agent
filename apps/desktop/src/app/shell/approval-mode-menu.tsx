import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'
import { useEffect, useMemo } from 'react'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import { findGroup, findGroupOfPane } from '@/components/pane-shell/tree/model'
import {
  $activeTreeGroup,
  $collapsedTreeSides,
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  isPaneVisible
} from '@/components/pane-shell/tree/store'
import {
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { $registryVersion, registry } from '@/contrib/registry'
import type { PluginWorkspaceOptions } from '@/contrib/types'
import { useI18n } from '@/i18n'
import { Zap, ZapFilled } from '@/lib/icons'
import {
  $approvalModes,
  type ApprovalMode,
  type ApprovalModeRequester,
  setApprovalModeForProfile,
  syncApprovalModeForProfile
} from '@/store/approval-mode'
import { $lastApprovalContentPane, isApprovalContentPane } from '@/store/session-focus'

const declaresPerProfileApprovals = (paneId: string): boolean =>
  (registry.getArea('panes').find(c => c.id === paneId)?.data as Partial<PluginWorkspaceOptions> | undefined)
    ?.approvalScope === 'per-profile'

// A workspace that declares `approvalScope: 'per-profile'` (a group room) runs several
// independently gated profiles, so the ambient profile's mode neither describes nor controls it.
const $showsPerProfileWorkspace = computed(
  [
    $activeTreeGroup,
    $layoutTree,
    $lastApprovalContentPane,
    $dismissedPanes,
    $hiddenTreePanes,
    $collapsedTreeSides,
    $registryVersion
  ],
  (groupId, tree, remembered) => {
    const active = tree && groupId ? findGroup(tree, groupId)?.active : undefined
    const visible = (paneId?: null | string) => (paneId && isPaneVisible(paneId) ? paneId : undefined)

    // A preview fronting the remembered pane's group covers it without leaving it.
    const covered = (paneId: null | string) => {
      if (!paneId || !tree || $hiddenTreePanes.get().has(paneId) || $dismissedPanes.get().has(paneId)) {
        return undefined
      }

      const front = findGroupOfPane(tree, paneId)?.active

      return front && !isApprovalContentPane(front) && visible(front) ? paneId : undefined
    }

    const content =
      (isApprovalContentPane(active) && visible(active)) ||
      visible(remembered) ||
      covered(remembered) ||
      visible(tree ? findGroupOfPane(tree, 'workspace')?.active : undefined)

    return Boolean(content && declaresPerProfileApprovals(content))
  }
)

export function useApprovalModeStatusbarItem(
  profile: string,
  requestGateway: ApprovalModeRequester,
  gatewayState: string,
  // Identity of the route the profile is served on (the active socket): a
  // move between gateways that serve the same profile name re-reads the mode.
  routeIdentity?: unknown
): StatusbarItem {
  const { t } = useI18n()
  const copy = t.shell.approvalMode
  const modes = useStore($approvalModes)
  const mode = modes[profile.trim() || 'default'] ?? 'smart'
  const hasProfileTarget = !useStore($showsPerProfileWorkspace)

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
    // The shell (and this chip) mount before the gateway socket opens at
    // boot, and a sync issued that early rejects with "Hermes gateway
    // unavailable" — the swallow below then leaves the chip on its
    // optimistic 'smart' default until some later session event reconciles
    // it. Gate on the open socket and re-run when it opens (or a reconnect
    // lands), the same way useStatusSnapshot does.
    if (gatewayState !== 'open') {
      return
    }

    void syncApprovalModeForProfile(requestGateway, profile).catch(() => undefined)
  }, [gatewayState, profile, requestGateway, routeIdentity])

  if (!hasProfileTarget) {
    return {
      icon: <Zap className="size-3.5 opacity-70" />,
      id: 'approval-mode',
      label: copy.perProfile,
      menuAlign: 'end',
      menuClassName: 'w-72 p-1',
      menuContent: <DropdownMenuLabel className="whitespace-normal">{copy.perProfileDescription}</DropdownMenuLabel>,
      title: copy.perProfileDescription,
      variant: 'menu'
    }
  }

  return {
    className: mode === 'off' ? 'bg-(--chrome-action-hover) text-foreground' : undefined,
    icon: mode === 'off' ? <ZapFilled className="size-3.5" /> : <Zap className="size-3.5 opacity-70" />,
    id: 'approval-mode',
    label: labels[mode],
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
      </>
    ),
    title: copy.ariaLabel(labels[mode]),
    variant: 'menu'
  }
}
