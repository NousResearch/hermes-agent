import { atom, computed } from 'nanostores'

import { DEMO_LAYOUT_ID } from '@/app/contrib/layout-presets'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { $activePresetId } from '@/components/pane-shell/tree/store'
import { TIP_CATALOG } from '@/lib/tips/catalog'
import { $sidebarOpen, CHAT_SIDEBAR_PANE_ID } from '@/store/layout'
import { notify } from '@/store/notifications'
import { $onboardingGate, $setupProfileName, completeGuide, leaveGuide, skipGuide } from '@/store/onboarding-gate'
import { $introView } from '@/store/onboarding-intro'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $newChatRoute,
  type AgentProfileRoute,
  normalizeProfileKey,
  selectProfile
} from '@/store/profile'
import { $selectedStoredSessionId } from '@/store/session'
import { storedSessionIdForRuntimeId } from '@/store/session-states'
import { isStartChatCallerWatched } from '@/store/start-chat'
import { retireTips } from '@/store/tips'
import { $toursEnabled } from '@/store/tours'

import { $chatOnboardingThreadIds, endChatOnboardingSolo, keepChatOnboardingLayout } from './assembly'
import { showHandoffTour } from './signpost'

/**
 * The intro copy over an empty setup chat: it types centred (`playing`), springs up and waits at
 * the top (`landed`) until the backend's first assistant row, which carries the same words,
 * takes its place.
 */
type IntroCopyStage = 'hidden' | 'landed' | 'playing'

export const $introCopy = atom<IntroCopyStage>('hidden')

/** While the copy types and springs up, the thread under it stays hidden: the backend's first row and the
 *  card after it can land early, and a see-through chat surface (glass) would show them mid-motion. */
export const $introHoldsThread = computed($introCopy, stage => stage === 'playing')

/** The hidden `/initiate-setup` turn has been handed to the backend (or failed to be). */
export const $introTurnSent = atom(false)

interface LaunchSource {
  newChatProfile: null | string
  newChatRoute: AgentProfileRoute | null
  profile: string
}

// Where new chats went before the intro moved them into the setup profile.
let launch: LaunchSource | null = null

/** The boot overlay holds while the kickoff checks inference and opens the setup chat; the window takes
 *  the demo shape only once inference is known to be there. */
export function startIntro(): void {
  $introView.set('starting')
}

export function rememberLaunchSource(): void {
  launch = {
    newChatProfile: $newChatProfile.get(),
    newChatRoute: $newChatRoute.get(),
    profile: $activeGatewayProfile.get()
  }
}

/** The setup chat is open. A chat with no assistant words yet plays the intro copy. */
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

  // A leave action that picked a profile itself (the rail, a new chat elsewhere) keeps its pick. Otherwise
  // new chats go to the launch profile by name: with no intent they would follow the active gateway,
  // which is still the setup profile.
  if (launch && $newChatProfile.get() === $setupProfileName.get()) {
    $newChatProfile.set(launch.newChatProfile ?? normalizeProfileKey(launch.profile))
    $newChatRoute.set(launch.newChatRoute)
  }

  return true
}

/** Any leave action (sidebar, another chat, a profile, a Cmd-K command) ends the intro. */
export function leaveIntro(): void {
  if (endIntroView()) {
    leaveGuide()
  }
}

/** A layout pick (the setup card, `apply_layout`, the layout picker) ends the intro and keeps the pick. */
function leaveIntroOnLayout(): void {
  if (endIntroView(true)) {
    leaveGuide()
  }
}

export function skipIntro(): void {
  if (!endIntroView()) {
    return
  }

  notify({ kind: 'info', message: 'Switching you over to your default profile' })
  skipGuide()
  // Skip stops the tutorial tips. The local-setup offer still arms: its card waits for a finished task.
  retireTips(TIP_CATALOG.map(tip => tip.id))
  selectProfile(launch?.profile ?? 'default')
}

/** A setup chat's `start_chat` started the task chat (`startedId`): the guided first run is complete. */
export function finishGuidedOnboarding(runtimeId: string, startedId: string): void {
  const threads = $chatOnboardingThreadIds.get()
  const storedId = storedSessionIdForRuntimeId(runtimeId)

  if (!threads.includes(runtimeId) && !(storedId && threads.includes(storedId))) {
    return
  }

  const { phase } = $onboardingGate.get()
  // The task chat opens itself only for a watched caller (start-chat-tool.tsx). A setup chat that
  // finished in the background opens nothing, so it must not open a tour either.
  const handoffOpens = isStartChatCallerWatched(storedId ?? runtimeId)

  endIntroView()
  completeGuide()

  if (handoffOpens && (phase === 'guided' || phase === 'left') && $toursEnabled.get()) {
    void showHandoffTour(() => $selectedStoredSessionId.get() === startedId)
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
    $sidebarOpen.listen(() => leaveIntro()),
    $activePresetId.listen(id => id !== DEMO_LAYOUT_ID && leaveIntroOnLayout()),
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
