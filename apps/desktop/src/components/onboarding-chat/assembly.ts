/**
 * The guided chat runs alone in a small window. Picking a layout assembles the app around the conversation.
 *
 * The window grows by the minimum the new panes need, with a viewport floor that keeps the sidebar docked. The main
 * process animates the growth with setBounds (electron/chat-onboarding-window.ts), so no CSS transition is involved.
 */

import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'

import { allPaneIds, group, type LayoutNode } from '@/components/pane-shell/tree/model'
import { applyLayoutPreset } from '@/components/pane-shell/tree/presets'
import {
  $activePresetId,
  $layoutTree,
  adoptContributedPanes,
  dismissTreePane,
  resetEnforcedDocks,
  undismissTreePanes
} from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { DOCKED_SIDEBAR_MIN_PX } from '@/hooks/use-mobile'
import { TRANSLATIONS } from '@/i18n/catalog'
import { getRuntimeI18nLocale } from '@/i18n/runtime'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { setSidebarOpen } from '@/store/layout'
import { loadMachineProfile, machineUserName } from '@/store/machine'
import { skipGuide } from '@/store/onboarding-gate'
import { setOnboardingSurfaceActive } from '@/store/onboarding-presence'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

/** True from guide kickoff until assembly places the picked layout. Skip and a failed kickoff also clear it. */
export const $chatOnboardingSolo = atom(false)

// Mirrors solo mode into the presence set, which hides ambient UI such as the update toast (onboarding-presence.ts).
$chatOnboardingSolo.subscribe(solo => setOnboardingSurfaceActive('solo-chat', solo))

/** The thread list keys by stored id; the composer keys by runtime id. Both
 *  identify the conversation that gets onboarding transcript treatment. */
export const $chatOnboardingThreadIds = atom<readonly string[]>([])

/** Holds the localized opener so it is ready before inference: cold first turns took 10 s. The typed reveal and the
 *  seed rows read this same string, so the model receives the text the user saw. */
export const $onboardingGreeting = atom('')

/** First-write-wins keeps the opener stable through profile and backend boot. */
export function pickOnboardingGreeting(): string {
  const existing = $onboardingGreeting.get()

  if (existing) {
    return existing
  }

  const copy = TRANSLATIONS[getRuntimeI18nLocale()].guidedGreeting
  const suggested = machineUserName()

  $onboardingGreeting.set(suggested ? `${copy.line}\n\n${copy.nameSuggestion(suggested)}` : copy.line)

  return $onboardingGreeting.get()
}

/** Applying a layout remounts the card, so its selection must outlive the component. */
export const $chatLayoutPicked = atom(false)

let previousLayout: { id: string; tree: LayoutNode | null } | null = null

/** The guide's shape, all at once: the solo layout and the small centred
 *  window. Called on the tick the guide is owed (film ended, or a boot that
 *  finds the guide queued) so no full-size frame paints in between. */
export function takeGuideShape(): void {
  if ($chatOnboardingSolo.get()) {
    return
  }

  startChatOnboardingSolo()

  // startChatOnboardingSolo declines when the guide is off; shrink only when it took.
  if ($chatOnboardingSolo.get()) {
    window.hermesDesktop?.chatOnboarding?.soloBoot?.()
  }
}

export function startChatOnboardingSolo(): void {
  if (!isOnboardingEnabled() || $chatOnboardingSolo.get()) {
    return
  }

  previousLayout = { id: $activePresetId.get(), tree: $layoutTree.get() }
  $chatOnboardingSolo.set(true)
  $chatLayoutPicked.set(false)
  // The local machine probe finishes before the backend boots, letting the
  // greeting type while kickoff is still waiting for a session.
  void loadMachineProfile().then(() => {
    if ($chatOnboardingSolo.get()) {
      pickOnboardingGreeting()
    }
  })
  // Adoption puts other panes in this same group. Hiding its strip keeps them
  // invisible, including reactive arrivals, until a layout places them.
  applyLayoutPreset('chat-solo', group(['workspace'], { tabStrip: 'never' }))
}

/** Called when the guide kickoff fails, so classic onboarding can resume. */
export function endChatOnboardingSolo(): void {
  $chatOnboardingSolo.set(false)
  $onboardingGreeting.set('')

  const previous = previousLayout
  previousLayout = null

  if (previous) {
    const tree = previous.tree ?? registry.getArea('layouts').find(preset => preset.id === 'default')?.data

    if (tree) {
      // SAFETY: layout contributions declare LayoutNode data, like the saved tree.
      applyLayoutPreset(previous.tree ? previous.id : 'default', tree as LayoutNode)
    }
  }
}

/** Per-preset growth in pixels, sized to what the new panes need. Deriving the growth from the chat's own size made
 *  the window much too large. */
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

/** Re-picks must reset persisted dismissals, docks, and sidebar visibility:
 *  swapping only the tree left Elite's terminal dismissed after Basic. */
function reconcileLayout(id: string, tree: LayoutNode): void {
  applyLayoutPreset(id, tree)

  const declared = new Set(allPaneIds(tree))

  undismissTreePanes(declared)

  // plugins/hermes-bots/plugin.tsx enforces a dock onto Sessions; adoption
  // otherwise adds its roster and a tab strip to the sidebar. Dismiss every
  // undeclared pane, including registry entries not placed yet, so subsequent
  // adoption cannot bring them back. Their own toggles still can.
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

  // A persisted closed sidebar would hide the column this pick just requested.
  setSidebarOpen(true)

  // Solo boot consumed dock enforcement before a sidebar existed. Reset that record so adoption can dock against the
  // newly placed Sessions column.
  resetEnforcedDocks()
  adoptContributedPanes()

  // Showing the sidebar can register more plugin panes synchronously. Dismiss last so those panes are dismissed as
  // well; Basic otherwise gained an empty Cronjobs column.
  dismissUndeclared()
}

/** Grow only when leaving solo mode: repeating the delta would make the window larger on every re-pick.
 *  Reconcile panes on every pick. */
export function assembleChatOnboarding(id: string, tree: LayoutNode): void {
  const firstPick = $chatOnboardingSolo.get()

  if (firstPick) {
    const growth = LAYOUT_GROWTH.get(id) ?? { left: 220 }

    window.hermesDesktop?.chatOnboarding?.grow({
      bottom: growth.bottom ?? 0,
      left: growth.left ?? 0,
      right: growth.right ?? 0,
      // Pane deltas can leave Basic below the sidebar's docking breakpoint at
      // the user's zoom. Main applies this viewport floor, then the display clamp.
      minWidth: DOCKED_SIDEBAR_MIN_PX,
      top: growth.top ?? 0
    })
  }

  reconcileLayout(id, tree)

  $chatOnboardingSolo.set(false)
}

/** Skip ends setup without deleting its conversation; the phase write prevents resuming it. */
export function skipChatOnboarding(): void {
  const preset = registry.getArea('layouts').find(contribution => contribution.id === 'basic')

  if (preset?.data) {
    // SAFETY: Layout presets declare data: LayoutNode (pane-shell/tree/presets.ts).
    assembleChatOnboarding(preset.id, preset.data as LayoutNode)
  } else {
    $chatOnboardingSolo.set(false)
  }

  skipGuide()
}

/** Suppress floating panels and swap chrome throughout the flow's conversations. */
export function useOnboardingChatActive(): boolean {
  const solo = useStore($chatOnboardingSolo)
  const threadIds = useStore($chatOnboardingThreadIds)
  const runtimeId = useStore($activeSessionId)
  const storedId = useStore($selectedStoredSessionId)

  return (
    solo || (runtimeId != null && threadIds.includes(runtimeId)) || (storedId != null && threadIds.includes(storedId))
  )
}
