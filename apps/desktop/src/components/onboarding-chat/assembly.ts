import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'

import { useSessionView } from '@/app/chat/session-view'
import { DEMO_LAYOUT_ID, DEMO_TREE } from '@/app/contrib/layout-presets'
import { allPaneIds, type LayoutNode } from '@/components/pane-shell/tree/model'
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
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $interfaceMode, type InterfaceMode, modeLayout, setInterfaceMode } from '@/store/interface-mode'
import { setSidebarOpen } from '@/store/layout'
import { $chatOnboardingSolo, $introView } from '@/store/onboarding-intro'
import { $paneStates, type PaneStateSnapshot } from '@/store/panes'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

// The demo is borrowed: nothing it changes is persisted as the user's layout.
$chatOnboardingSolo.subscribe(solo => {
  modeLayout.hold(solo)
  document.documentElement.toggleAttribute('data-onboarding-demo', solo)
})

export const $chatOnboardingThreadIds = atom<readonly string[]>([])

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

function startChatOnboardingSolo(): void {
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
  applyLayoutPreset(DEMO_LAYOUT_ID, DEMO_TREE)
}

/** Leave the demo for the user's own layout at the normal window size. */
export function endChatOnboardingSolo(): void {
  if ($chatOnboardingSolo.get()) {
    window.hermesDesktop?.chatOnboarding?.size('normal')
  }

  $chatOnboardingSolo.set(false)
  restorePreviousLayout()
}

/** Leave the demo on the layout just picked in it: drop the snapshot, release the hold, save the pick. */
export function keepChatOnboardingLayout(): void {
  previousLayout = null

  if ($chatOnboardingSolo.get()) {
    window.hermesDesktop?.chatOnboarding?.size('normal')
  }

  $chatOnboardingSolo.set(false)
  persistTree()
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
  $chatOnboardingSolo.set(false)

  if (mode && mode !== $interfaceMode.get()) {
    restorePreviousLayout()
    setInterfaceMode(mode)
  }

  previousLayout = null

  window.hermesDesktop?.chatOnboarding?.size('normal')

  reconcileLayout(id, tree)
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

    // A pick made in the demo ended the intro, so undoing it lands on the user's own layout.
    if (snapshot.solo) {
      previousLayout = snapshot.previous
      restorePreviousLayout()
      $chatLayoutPicked.set(snapshot.picked)

      return
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
  }
}

/** The setup chat is guided only while the intro runs; in `ended` or `off` it is a normal chat. The
 *  thread ids outlive the intro so a later `start_chat` from that chat is still recognized. */
export function useOnboardingChatActive(): boolean {
  const solo = useStore($chatOnboardingSolo)
  const intro = useStore($introView) === 'intro'
  const threadIds = useStore($chatOnboardingThreadIds)
  const runtimeId = useStore($activeSessionId)
  const storedId = useStore($selectedStoredSessionId)

  return (
    solo ||
    (intro &&
      ((runtimeId != null && threadIds.includes(runtimeId)) || (storedId != null && threadIds.includes(storedId))))
  )
}

/** Whether the chat view this renders in (primary or a tile) shows the setup
 *  chat while the intro runs. Solo covers the primary view before the guide's
 *  session ids are known; in `ended` or `off` the setup chat is a normal chat. */
export function useSetupChatView(): boolean {
  const view = useSessionView()
  const solo = useStore($chatOnboardingSolo)
  const intro = useStore($introView) === 'intro'
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)

  const inThread = useStoreSelector($chatOnboardingThreadIds, ids =>
    [runtimeId, storedId].some(id => id != null && ids.includes(id))
  )

  return (view.kind === 'primary' && solo) || (intro && inThread)
}
