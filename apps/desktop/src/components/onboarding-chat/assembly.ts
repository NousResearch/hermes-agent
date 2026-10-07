import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'

import { allPaneIds, group, type LayoutNode } from '@/components/pane-shell/tree/model'
import { applyLayoutPreset } from '@/components/pane-shell/tree/presets'
import {
  $activePresetId,
  $layoutTree,
  $userPlacedPanes,
  adoptContributedPanes,
  dismissTreePane,
  markActivePreset,
  persistTree,
  resetEnforcedDocks,
  undismissTreePanes
} from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { DOCKED_SIDEBAR_MIN_PX } from '@/hooks/use-mobile'
import { runtimeTranslations } from '@/i18n/runtime'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { $interfaceMode, type InterfaceMode, setInterfaceMode } from '@/store/interface-mode'
import { setSidebarOpen } from '@/store/layout'
import { loadMachineProfile, machineUserName } from '@/store/machine'
import { skipGuide } from '@/store/onboarding-gate'
import { setOnboardingSurfaceActive } from '@/store/onboarding-presence'
import { $paneStates, type PaneStateSnapshot } from '@/store/panes'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

export const $chatOnboardingSolo = atom(false)

$chatOnboardingSolo.subscribe(solo => setOnboardingSurfaceActive('solo-chat', solo))

export const $chatOnboardingThreadIds = atom<readonly string[]>([])

export const $onboardingGreeting = atom('')

export function pickOnboardingGreeting(): string {
  const existing = $onboardingGreeting.get()

  if (existing) {
    return existing
  }

  const copy = runtimeTranslations().guidedGreeting
  const suggested = machineUserName()

  const greeting = suggested ? `${copy.line}\n\n${copy.nameSuggestion(suggested)}` : copy.line
  $onboardingGreeting.set(greeting)

  return greeting
}

export const $chatLayoutPicked = atom(false)

let previousLayout: {
  id: string
  tree: LayoutNode | null
  panes: Record<string, PaneStateSnapshot>
  placed: ReadonlySet<string>
} | null = null

export function takeGuideShape(): void {
  if ($chatOnboardingSolo.get()) {
    return
  }

  startChatOnboardingSolo()

  if ($chatOnboardingSolo.get()) {
    window.hermesDesktop?.chatOnboarding?.soloBoot?.()
  }
}

export function startChatOnboardingSolo(): void {
  if (!isOnboardingEnabled() || $chatOnboardingSolo.get()) {
    return
  }

  previousLayout = {
    id: $activePresetId.get(),
    tree: $layoutTree.get(),
    panes: $paneStates.get(),
    placed: $userPlacedPanes.get()
  }
  $chatOnboardingSolo.set(true)
  $chatLayoutPicked.set(false)
  void loadMachineProfile()
  applyLayoutPreset('chat-solo', group(['workspace'], { tabStrip: 'never' }))
}

export function endChatOnboardingSolo(): void {
  $chatOnboardingSolo.set(false)
  $onboardingGreeting.set('')
  restorePreviousLayout()
}

function restorePreviousLayout() {
  const previous = previousLayout
  previousLayout = null

  if (previous) {
    const tree = previous.tree ?? registry.getArea('layouts').find(preset => preset.id === 'default')?.data

    if (tree) {
      $layoutTree.set(tree as LayoutNode)
      $paneStates.set(previous.panes)
      $userPlacedPanes.set(previous.placed)
      markActivePreset(previous.tree ? previous.id : 'default')
      persistTree()
    }
  }
}

interface LayoutGrowth {
  bottom?: number
  left?: number
  right?: number
  top?: number
}

const LAYOUT_GROWTH = new Map<string, LayoutGrowth>([
  ['basic', { left: 220 }],
  ['terminal-deck', { bottom: 200, left: 220, right: 240 }]
])

function reconcileLayout(id: string, tree: LayoutNode): void {
  applyLayoutPreset(id, tree)

  const declared = new Set(allPaneIds(tree))

  undismissTreePanes(declared)

  const dismissUndeclared = () => {
    for (const paneId of new Set([
      ...allPaneIds($layoutTree.get() ?? tree),
      ...registry.getArea('panes').map(pane => pane.id)
    ])) {
      if (!declared.has(paneId)) {
        dismissTreePane(paneId)
      }
    }
  }

  setSidebarOpen(true)

  resetEnforcedDocks()
  adoptContributedPanes()

  dismissUndeclared()
}

export function assembleChatOnboarding(id: string, tree: LayoutNode, mode?: InterfaceMode): void {
  const firstPick = $chatOnboardingSolo.get()

  if (mode && mode !== $interfaceMode.get()) {
    restorePreviousLayout()
    setInterfaceMode(mode)
  }

  previousLayout = null

  if (firstPick) {
    const growth = LAYOUT_GROWTH.get(id) ?? { left: 220 }

    window.hermesDesktop?.chatOnboarding?.grow({
      bottom: growth.bottom ?? 0,
      left: growth.left ?? 0,
      right: growth.right ?? 0,
      minWidth: DOCKED_SIDEBAR_MIN_PX,
      top: growth.top ?? 0
    })
  }

  reconcileLayout(id, tree)

  $chatOnboardingSolo.set(false)
}

export function skipChatOnboarding(): void {
  const preset = registry.getArea('layouts').find(contribution => contribution.id === 'basic')

  if (preset?.data) {
    assembleChatOnboarding(preset.id, preset.data as LayoutNode)
  } else {
    $chatOnboardingSolo.set(false)
  }

  skipGuide()
}

export function useOnboardingChatActive(): boolean {
  const solo = useStore($chatOnboardingSolo)
  const threadIds = useStore($chatOnboardingThreadIds)
  const runtimeId = useStore($activeSessionId)
  const storedId = useStore($selectedStoredSessionId)

  return (
    solo || (runtimeId != null && threadIds.includes(runtimeId)) || (storedId != null && threadIds.includes(storedId))
  )
}
