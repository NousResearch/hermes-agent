import { atom } from 'nanostores'

import { DEMO_LAYOUT_ID } from '@/app/contrib/layout-presets'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { $activePresetId } from '@/components/pane-shell/tree/store'
import { TIP_CATALOG } from '@/lib/tips/catalog'
import { LOCAL_SETUP_TIP_ID } from '@/lib/tips/local-cta'
import { $sidebarOpen, CHAT_SIDEBAR_PANE_ID } from '@/store/layout'
import { notify } from '@/store/notifications'
import { $onboardingGate, $setupProfileName, completeGuide, leaveGuide, skipGuide } from '@/store/onboarding-gate'
import { $introView } from '@/store/onboarding-intro'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $newChatRoute,
  type AgentProfileRoute,
  selectProfile
} from '@/store/profile'
import { $selectedStoredSessionId } from '@/store/session'
import { storedSessionIdForRuntimeId } from '@/store/session-states'
import { retireTips } from '@/store/tips'
import { $toursEnabled } from '@/store/tours'

import { $chatOnboardingThreadIds, endChatOnboardingSolo, takeGuideShape } from './assembly'
import { showHandoffTour } from './signpost'

/**
 * The intro copy over an empty setup chat: it types centred (`playing`), springs up and waits at
 * the top (`landed`) until the backend's first assistant row, which carries the same words,
 * takes its place.
 */
export type IntroCopyStage = 'hidden' | 'landed' | 'playing'

export const $introCopy = atom<IntroCopyStage>('hidden')

/** The hidden `/initiate-setup` turn has been handed to the backend (or failed to be). */
export const $introTurnSent = atom(false)

interface LaunchSource {
  newChatProfile: null | string
  newChatRoute: AgentProfileRoute | null
  profile: string
}

// Where new chats went before the intro moved them into the setup profile.
let launch: LaunchSource | null = null

export function startIntro(): void {
  $introView.set('starting')
  takeGuideShape()
}

export function rememberLaunchSource(): void {
  launch = {
    newChatProfile: $newChatProfile.get(),
    newChatRoute: $newChatRoute.get(),
    profile: $activeGatewayProfile.get()
  }
}

/** The setup chat is open. A fresh (empty) chat plays the intro copy. */
export function openIntro(fresh: boolean): void {
  $introView.set('intro')
  $introTurnSent.set(!fresh)
  $introCopy.set(fresh ? 'playing' : 'hidden')
}

export function failIntro(): void {
  $introView.set('off')
  $introCopy.set('hidden')
  endChatOnboardingSolo()
}

function endIntroView(): boolean {
  if ($introView.get() !== 'intro') {
    return false
  }

  $introView.set('ended')
  $introCopy.set('hidden')
  endChatOnboardingSolo()

  // A leave action that picked a profile itself (the rail, a new chat elsewhere) keeps its pick.
  if (launch && $newChatProfile.get() === $setupProfileName.get()) {
    $newChatProfile.set(launch.newChatProfile)
    $newChatRoute.set(launch.newChatRoute)
  }

  return true
}

/** Any leave action (sidebar, another chat, a profile, a layout, a Cmd-K command) ends the intro. */
export function leaveIntro(): void {
  if (endIntroView()) {
    leaveGuide()
  }
}

export function skipIntro(): void {
  if (!endIntroView()) {
    return
  }

  notify({ kind: 'info', message: 'Switching you over to your default profile' })
  skipGuide()
  // Skip stops the rest of the first run: the tutorial tips and the local-model tip.
  retireTips([...TIP_CATALOG.map(tip => tip.id), LOCAL_SETUP_TIP_ID])
  selectProfile(launch?.profile ?? 'default')
}

/** A setup chat's `start_chat` started the task chat: the guided first run is complete. */
export function finishGuidedOnboarding(runtimeId: string): void {
  const threads = $chatOnboardingThreadIds.get()
  const storedId = storedSessionIdForRuntimeId(runtimeId)

  if (!threads.includes(runtimeId) && !(storedId && threads.includes(storedId))) {
    return
  }

  const { phase } = $onboardingGate.get()

  endIntroView()
  completeGuide()

  if ((phase === 'guided' || phase === 'left') && $toursEnabled.get()) {
    void showHandoffTour()
  }
}

function watchLeaveActions(): () => void {
  const setupProfile = $activeGatewayProfile.get()

  // Below the collapse breakpoint ⌘B and the titlebar toggle only send the reveal event.
  const onReveal = (event: Event) => {
    const detail = (event as CustomEvent<{ id?: string; mode?: string }>).detail

    if (detail?.id === CHAT_SIDEBAR_PANE_ID && detail.mode !== 'close') {
      leaveIntro()
    }
  }

  window.addEventListener(PANE_TOGGLE_REVEAL_EVENT, onReveal)

  const stops = [
    () => window.removeEventListener(PANE_TOGGLE_REVEAL_EVENT, onReveal),
    $sidebarOpen.listen(leaveIntro),
    $activePresetId.listen(id => id !== DEMO_LAYOUT_ID && leaveIntro()),
    $activeGatewayProfile.listen(profile => profile !== setupProfile && leaveIntro()),
    $selectedStoredSessionId.listen(id => (!id || !$chatOnboardingThreadIds.get().includes(id)) && leaveIntro())
  ]

  return () => stops.forEach(stop => stop())
}

let stopWatching: (() => void) | null = null

$introView.listen(view => {
  stopWatching?.()
  stopWatching = view === 'intro' ? watchLeaveActions() : null
})
