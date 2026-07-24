import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenuItem,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger
} from '@/components/ui/context-menu'
import {
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import type { SplitDir } from '@/store/session-states'

/** The leaf + submenu components for one menu flavour, so the split submenu
 *  renders in either the `…` dropdown or a right-click context menu. */
export interface SplitMenuKit {
  Item: typeof DropdownMenuItem | typeof ContextMenuItem
  Sub: typeof DropdownMenuSub | typeof ContextMenuSub
  SubContent: typeof DropdownMenuSubContent | typeof ContextMenuSubContent
  SubTrigger: typeof DropdownMenuSubTrigger | typeof ContextMenuSubTrigger
}

export const DROPDOWN_SPLIT_KIT: SplitMenuKit = {
  Item: DropdownMenuItem,
  Sub: DropdownMenuSub,
  SubContent: DropdownMenuSubContent,
  SubTrigger: DropdownMenuSubTrigger
}

export const CONTEXT_SPLIT_KIT: SplitMenuKit = {
  Item: ContextMenuItem,
  Sub: ContextMenuSub,
  SubContent: ContextMenuSubContent,
  SubTrigger: ContextMenuSubTrigger
}

// Ordered so the default (right) sits first, one hop away.
const SPLIT_DIRS: { dir: SplitDir; icon: string }[] = [
  { dir: 'right', icon: 'arrow-right' },
  { dir: 'bottom', icon: 'arrow-down' },
  { dir: 'left', icon: 'arrow-left' },
  { dir: 'top', icon: 'arrow-up' }
]

interface SplitSubmenuProps {
  kit: SplitMenuKit
  label: string
  onSplit: (dir: SplitDir) => void
  disabled?: boolean
  /** Dismiss the owning menu after the row's default (right) split — the
   *  dropdown is controlled and can; a context menu can't, so it's a no-op. */
  close?: () => void
}

/**
 * "Open in split ▸": clicking the row splits right (the common case), and the
 * submenu picks any edge. Shared by session rows and page nav rows.
 */
export function SplitSubmenu({ close, disabled, kit, label, onSplit }: SplitSubmenuProps) {
  const { Item, Sub, SubContent, SubTrigger } = kit
  const { t } = useI18n()

  const directionLabels: Record<SplitDir, string> = {
    top: t.zones.dirUp,
    bottom: t.zones.dirDown,
    left: t.zones.dirLeft,
    right: t.zones.dirRight
  }

  const split = (dir: SplitDir) => {
    triggerHaptic('selection')
    onSplit(dir)
  }

  return (
    <Sub>
      <SubTrigger
        disabled={disabled}
        onClick={() => {
          split('right')
          close?.()
        }}
      >
        <Codicon name="split-horizontal" size="0.875rem" />
        <span>{label}</span>
      </SubTrigger>
      <SubContent>
        {SPLIT_DIRS.map(({ dir, icon }) => (
          <Item key={dir} onSelect={() => split(dir)}>
            <Codicon name={icon} size="0.875rem" />
            <span>{directionLabels[dir]}</span>
          </Item>
        ))}
      </SubContent>
    </Sub>
  )
}
