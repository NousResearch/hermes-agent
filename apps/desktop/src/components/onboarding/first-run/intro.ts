import { atom, computed } from 'nanostores'

import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { TIP_CATALOG } from '@/lib/tips/catalog'
import { onLayoutCommit } from '@/store/appearance-apply'
import { $sidebarOpen, CHAT_SIDEBAR_PANE_ID } from '@/store/layout'
import { $introView } from '@/store/onboarding-intro'
import { $activeGatewayProfile } from '@/store/profile'
import { $selectedStoredSessionId } from '@/store/session'
import { retireTips } from '@/store/tips'

import { $chatOnboardingThreadIds, endChatOnboardingSolo, keepChatOnboardingLayout } from './assembly'

/**
 * The intro copy over an empty onboarding chat: it types centred (`playing`), springs up and waits at
 * the top (`landed`) until the backend's first assistant row, which carries the same words,
 * takes its place.
 */
type IntroCopyStage = 'hidden' | 'landed' | 'playing'

export const $introCopy = atom<IntroCopyStage>('hidden')

/** While the copy types and springs up, the thread under it stays hidden: the backend's first row and the
 *  card after it can land early, and a see-through chat surface (glass) would show them mid-motion. */
export const $introHoldsThread = computed($introCopy, stage => stage === 'playing')

/** The opening turn has been handed to the backend (`onboarding.start` submits it), or failed to be. */
export const $introTurnSent = atom(false)

/** The boot overlay holds while the kickoff opens the onboarding chat; the window takes the demo shape
 *  once `onboarding.start` has answered. */
export function startIntro(): void {
  $introView.set('starting')
}

/** The onboarding chat is open. A chat with no assistant words yet plays the intro copy. */
export function openIntro(blank: boolean): void {
  $introView.set('intro')
  $introCopy.set(blank ? 'playing' : 'hidden')
}

export function failIntro(): void {
  $introView.set('off')
  $introCopy.set('hidden')
  endChatOnboardingSolo()
}

// `keepLayout`: a layout pick ends the intro on the picked layout, not on the one from before the intro.
function endIntroView(keepLayout = false): boolean {
  if ($introView.get() !== 'intro') {
    return false
  }

  $introView.set('ended')
  $introCopy.set('hidden')

  if (keepLayout) {
    keepChatOnboardingLayout()
  } else {
    endChatOnboardingSolo()
  }

  return true
}

// A committed layout (the layout card's Continue, a set_theme layout) ends the intro on the user's own
// layout before it applies, so the pick lands at normal size over the user's layout, never the demo's.
onLayoutCommit(() => void endIntroView())

/** Any leave action (sidebar, another chat, a profile, a Cmd-K command) ends the intro. */
export function leaveIntro(): void {
  endIntroView()
}

/** A committed layout (the layout card's Continue, a `set_theme` layout) ends the intro and keeps the pick. */
export function leaveIntroOnLayout(): void {
  endIntroView(true)
}

/** Skip setup: the intro ends on the previous layout at normal size, and the tutorial tips stop. */
export function skipIntro(): void {
  endIntroView()
  retireTips(TIP_CATALOG.map(tip => tip.id))
}

function watchLeaveActions(): () => void {
  const introProfile = $activeGatewayProfile.get()

  // Below the collapse breakpoint ⌘B and the titlebar toggle only send the reveal event.
  const onReveal = (event: Event) => {
    // SAFETY: PANE_TOGGLE_REVEAL_EVENT is only dispatched as a CustomEvent with this detail (store/layout.ts).
    const detail = (event as CustomEvent<{ id?: string; mode?: string }>).detail

    if (detail?.id === CHAT_SIDEBAR_PANE_ID && detail.mode !== 'close') {
      leaveIntro()
    }
  }

  window.addEventListener(PANE_TOGGLE_REVEAL_EVENT, onReveal)

  const stops = [
    () => window.removeEventListener(PANE_TOGGLE_REVEAL_EVENT, onReveal),
    $sidebarOpen.listen(() => leaveIntro()),
    $activeGatewayProfile.listen(profile => profile !== introProfile && leaveIntro()),
    $selectedStoredSessionId.listen(id => (!id || !$chatOnboardingThreadIds.get().includes(id)) && leaveIntro())
  ]

  return () => stops.forEach(stop => stop())
}

let stopWatching: (() => void) | null = null

$introView.listen(view => {
  stopWatching?.()
  stopWatching = view === 'intro' ? watchLeaveActions() : null
})
