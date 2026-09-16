import { useStore } from '@nanostores/react'
import { type ReactNode } from 'react'

import { $registryVersion } from '@/contrib/registry'
import { $comboIndex } from '@/store/keybinds'

import { $activeTreeGroup, $hiddenTreePanes, $hoveredTreeGroup, $layoutTree, treeTabSlotTarget } from './store'
import { $heldTabModifier } from './tab-key-hint-state'

/** Subscribe in the tiny lead, not the pane/transcript subtree. */
export function TabKeyHint({ children, groupId, slot }: { children: ReactNode; groupId: string; slot: number }) {
  const held = useStore($heldTabModifier)

  return (
    <span className="relative flex shrink-0 items-center [&:has([data-tab-key-hint])>span:first-child]:invisible">
      <span className="flex items-center">{children}</span>
      {held && <HeldTabKeyHint groupId={groupId} slot={slot} />}
    </span>
  )
}

function HeldTabKeyHint({ groupId, slot }: { groupId: string; slot: number }) {
  useStore($activeTreeGroup)
  useStore($hoveredTreeGroup)
  useStore($layoutTree)
  useStore($hiddenTreePanes)
  useStore($registryVersion)
  const bindings = useStore($comboIndex)

  if (slot > 9 || bindings.get(`mod+${slot}`) !== `profile.switch.${slot}` || treeTabSlotTarget()?.id !== groupId) {
    return null
  }

  return (
    <span
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 flex items-center justify-center text-[0.625rem] font-semibold leading-none tabular-nums text-(--ui-text-secondary)"
      data-tab-key-hint={slot}
    >
      {slot}
    </span>
  )
}
