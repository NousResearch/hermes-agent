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
    window.hermesDesktop?.chatOnboarding?.size('onboarding')
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
  if (mode && mode !== $interfaceMode.get()) {
    restorePreviousLayout()
    setInterfaceMode(mode)
  }

  previousLayout = null

  window.hermesDesktop?.chatOnboarding?.size('normal')

  reconcileLayout(id, tree)

  $chatOnboardingSolo.set(false)
}

export function snapshotChatLayout(): () => void {
  const snapshot = {
    id: $activePresetId.get(),
    mode: $interfaceMode.get(),
    panes: $paneStates.get(),
    picked: $chatLayoutPicked.get(),
    placed: $userPlacedPanes.get(),
    previous: previousLayout,
    solo: $chatOnboardingSolo.get(),
    tree: $layoutTree.get()
  }

  return () => {
    if (snapshot.mode !== $interfaceMode.get()) {
      setInterfaceMode(snapshot.mode)
    }

    if (snapshot.tree) {
      $layoutTree.set(snapshot.tree)
      $paneStates.set(snapshot.panes)
      $userPlacedPanes.set(snapshot.placed)
      markActivePreset(snapshot.id)
      persistTree()
    }

    previousLayout = snapshot.previous
    $chatLayoutPicked.set(snapshot.picked)

    if (snapshot.solo && !$chatOnboardingSolo.get()) {
      $chatOnboardingSolo.set(true)
      window.hermesDesktop?.chatOnboarding?.size('onboarding')
    }
  }
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
