/**
 * TESTING ONLY: every onboarding and discovery state the lab can put the app in, as data. Each state
 * drives the real stores (the guidance ledger, the splashes, the held message) and, where a test
 * machine cannot produce the fact on demand, one of the stand-ins in store/discovery-lab.ts; then it
 * takes the tester to where the real UI shows it. Nothing here renders discovery UI of its own.
 */

import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import type { ExistingUserVerdict } from '@/lib/existing-user'
import { OPEN_ACCESS, SIGN_IN_OUTCOME_PARAM, signUpHref } from '@/lib/visitor-access'
import {
  $discoveryLabUsageStatsConsent,
  discoveryLabSignUpUrl,
  discoveryLabVisitorAccess,
  setDiscoveryLabSignUp,
  SIGN_UP_RETURN_PARAM
} from '@/store/discovery-lab'
import { $firstTimeSplash, reopenFirstTimeSplash } from '@/store/first-time-splash'
import { $guidanceLedger, type GuidanceMarkOf } from '@/store/guidance-ledger'
import { holdPrompt, takeHeldPrompt } from '@/store/held-prompt'
import { CHAT_SIDEBAR_PANE_ID } from '@/store/layout'
import { $newVersionReady } from '@/store/new-version'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $renameSplash, reopenRenameSplash } from '@/store/rename-splash'
import { setRememberedRoute, setRememberedSessionId } from '@/store/session'
import { $sharedMetricsDetailsOpen } from '@/store/shared-metrics'
import { openUsageStatsDetails } from '@/store/usage-stats-offer'
import { $visitorAccess, dismissSignInRecovery } from '@/store/visitor-access'

import { considerSplashes } from '../discovery/splash-layer'
import { $widthClass } from '../redesign/breakpoints'
import { NEW_CHAT_ROUTE, routePathname, sessionRoute } from '../routes'

export type LabGroup = 'new' | 'offers' | 'returning' | 'signUp' | 'tray' | 'updates'

export const LAB_GROUPS: readonly { id: LabGroup; title: string; description: string }[] = [
  {
    id: 'new',
    title: 'New person',
    description: 'Someone whose machine had no chats, agents or routines on their first visit.'
  },
  {
    id: 'returning',
    title: 'Returning user',
    description: 'Someone who used Hermes before the redesign.'
  },
  {
    id: 'signUp',
    title: 'Sign up',
    description:
      'A visitor to the hosted Webapp who must sign up before anything runs. There is no sign-up service here: ' +
      'a stand-in page in this app takes its place, and nothing leaves the app. The stand-in lasts for this tab ' +
      'until you turn it off or close the tab; each of these reloads the page.'
  },
  {
    id: 'offers',
    title: 'Offers & check-in',
    description: 'What the app asks once someone has settled in.'
  },
  {
    id: 'updates',
    title: 'Updates',
    description: 'A newer build of the Webapp waiting while a tab is open.'
  },
  {
    id: 'tray',
    title: 'Tray',
    description: 'The notifications tray behind the bell in the rail, as each kind of person sees it.'
  }
]

/** What the page knows about this window when a row is shown. */
export interface LabFacts {
  /** The newest stored chat the sidebar lists, for states shown over a stored chat. */
  newestChatId: null | string
  /** A chat whose answer held code this window has seen (the developer-view offer needs one). */
  codeChatId: null | string
  /** The interface mode is Simple (the developer-view offer is for someone in Simple). */
  simpleMode: boolean
  /** The sign-up stand-in is on in this tab. */
  signUpOn: boolean
}

/** How a state reaches the real UI. */
export interface LabHost {
  /** Go to an app route. */
  navigate: (to: string) => void
  /** Load the app afresh at this URL: what sign up settles is read once per page load. */
  load: (url: string) => void
  /** Press a control of the real UI once it is on screen (the bell, the splash's Next), as the
   *  tester would; `prepare` first brings it on screen. */
  press: (selector: string, prepare?: () => void) => void
}

export interface LabState {
  id: string
  group: LabGroup
  title: string
  description: string
  /** What the state writes that outlives a reload (the row says so); absent: memory only. */
  stores?: string
  /** Why it cannot open now, or null. */
  unavailable?: (facts: LabFacts) => null | string
  open: (host: LabHost, facts: LabFacts) => void
}

type Ledger = ReturnType<typeof $guidanceLedger.get>

/** Ids standing for chats in which a task finished: the ledger counts chats, never reads them. */
const FINISHED = ['discovery-lab-chat-1', 'discovery-lab-chat-2']

const HELD_TEXT = 'Plan a three-day trip to Lisbon in May, with a day for museums.'

const BELL = '[data-nav-spot="notifications"]'

/** Open the notifications tray with the bell, as the tester would. The bell is in the rail, which a
 *  compact window shows in the chat sidebar's sheet: that opens first, as the band's menu opens it. */
function openTray(host: LabHost): void {
  const showRail = () =>
    window.dispatchEvent(
      new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: CHAT_SIDEBAR_PANE_ID, mode: 'open' } })
    )

  host.press(BELL, $widthClass.get() === 'compact' ? showRail : undefined)
}

const FIRST_TIME_NEXT = '[data-splash="first-time"] .discovery-welcome-actions button'

const STAND_IN_CONSENT = { decided: false, enabled: false, send: false }

const NEWCOMER: Ledger = { renameSplash: 'not_applicable', firstTimeSplash: 'seen' }

const RETURNING: Ledger = { renameSplash: 'seen', firstTimeSplash: 'not_applicable' }

const MARKS = 'guidance marks (this Hermes instance)'

/** What every sign-up state changes besides the marks. */
const SIGN_UP_CHANGES = `${MARKS}; the sign-up stand-in (this tab); where the app reopens (a new chat)`

const HELD_CHANGES = `${SIGN_UP_CHANGES}; the held message (this browser)`

/** Whatever discovery has on screen goes away without being answered and the in-memory stand-ins
 *  are dropped, so the next state starts clean. Splashes closed here are not marked seen. */
function clearScreen(): void {
  // A newer build flagged (by the lab, or for real) loads on the new chat most states open; a real
  // one is found again at the next check.
  $newVersionReady.set(false)
  $discoveryLabUsageStatsConsent.set(null)
  $renameSplash.set(null)
  $firstTimeSplash.set(null)
  $sharedMetricsDetailsOpen.set(false)
  dismissSignInRecovery()
}

function setLedger(ledger: Ledger): void {
  clearScreen()
  $guidanceLedger.set(ledger)
}

/** The first look at the machine, replayed with `verdict`: the splash that person gets opens. */
function firstLook(host: LabHost, ledger: Ledger, verdict: ExistingUserVerdict): void {
  setLedger(ledger)
  host.navigate(NEW_CHAT_ROUTE)
  considerSplashes(verdict)
}

/** The usage-stats question unanswered (a stand-in, in memory), with its card's place `mark`. */
function usageStatsStandIn(mark?: GuidanceMarkOf<'usageStatsOffer'>): void {
  const { usageStatsOffer: _, ...ledger } = $guidanceLedger.get()
  setLedger(mark ? { ...ledger, usageStatsOffer: mark } : ledger)
  $discoveryLabUsageStatsConsent.set(STAND_IN_CONSENT)
}

/** This page's URL with `hash` as its route and nothing sign up left on it. */
function pageAt(hash: string, outcome?: 'failed'): string {
  const url = new URL(window.location.href)
  url.searchParams.delete(SIGN_UP_RETURN_PARAM)
  url.searchParams.delete(SIGN_IN_OUTCOME_PARAM)

  if (outcome) {
    url.searchParams.set(SIGN_IN_OUTCOME_PARAM, outcome)
  }

  url.hash = hash

  return url.href
}

/**
 * Load the app afresh at `url`. A load at the new chat route reopens the remembered chat or page
 * (use-desktop-integrations.ts), so one bound there goes to the new chat first, where nothing can
 * remember another place before the load, and forgets the remembered chat: the tester's next
 * launch opens on a new chat too, as if they had last been there.
 */
export function loadAt(host: LabHost, url: string): void {
  if (routePathname(new URL(url).hash.slice(1)) === NEW_CHAT_ROUTE) {
    const profile = normalizeProfileKey($activeGatewayProfile.get())
    host.navigate(NEW_CHAT_ROUTE)
    setRememberedRoute(NEW_CHAT_ROUTE, profile)
    setRememberedSessionId(null, profile)
  }

  host.load(url)
}

/** Turn the sign-up stand-in on with `ledger` (and a held message if asked) and load `url`. */
function signUpLoad(host: LabHost, ledger: Ledger, url: string, held = false): void {
  setLedger(ledger)
  setDiscoveryLabSignUp(true)
  // This page too, at once, as the next load will: a message held while sign up is not yet required
  // here would run before the load (held-prompt-run.ts).
  $visitorAccess.set(discoveryLabVisitorAccess(window.location.href) ?? OPEN_ACCESS)
  takeHeldPrompt()

  if (held) {
    holdPrompt(HELD_TEXT)
  }

  loadAt(host, url)
}

const needsStoredChat = (facts: LabFacts) => (facts.newestChatId ? null : 'Needs a stored chat on this machine.')

const signUpWaits = (facts: LabFacts) =>
  facts.signUpOn ? 'Waits while sign up is required: turn the sign-up stand-in off first.' : null

export const LAB_STATES: readonly LabState[] = [
  {
    id: 'welcome',
    group: 'new',
    title: 'Welcome to Hermes',
    description: 'The first-time splash, step 1: what each place in the nav is for. Opens on a new chat.',
    stores: MARKS,
    open: host => firstLook(host, { renameSplash: 'not_applicable' }, 'fresh')
  },
  {
    id: 'make-it-yours',
    group: 'new',
    title: 'Make it yours',
    description: 'The first-time splash, step 2: colour, light or dark, and artwork, then the way in.',
    stores: MARKS,
    open: host => {
      firstLook(host, { renameSplash: 'not_applicable' }, 'fresh')
      host.press(FIRST_TIME_NEXT)
    }
  },
  {
    id: 'welcome-reopened',
    group: 'new',
    title: 'Welcome splash, reopened',
    description: 'The first-time splash as "Meet Hermes" in the tray brings it back after it was seen.',
    stores: MARKS,
    open: host => {
      setLedger(NEWCOMER)
      host.navigate(NEW_CHAT_ROUTE)
      reopenFirstTimeSplash()
    }
  },
  {
    id: 'landing-first',
    group: 'new',
    title: 'Newcomer landing',
    description: 'A new chat before any task finished: "Welcome to Hermes" and the four starter ideas.',
    stores: MARKS,
    open: host => {
      setLedger(NEWCOMER)
      host.navigate(NEW_CHAT_ROUTE)
    }
  },
  {
    id: 'landing-back',
    group: 'new',
    title: 'Landing after a first task',
    description: '"Welcome back." with the ideas still out, once one chat had a task finish.',
    stores: MARKS,
    open: host => {
      setLedger({ ...NEWCOMER, finishedTaskChats: FINISHED.slice(0, 1) })
      host.navigate(NEW_CHAT_ROUTE)
    }
  },
  {
    id: 'ideas-hidden',
    group: 'new',
    title: 'Ideas hidden',
    description: 'The person pressed Hide ideas: "Welcome back." asks for a task, with Show ideas.',
    stores: MARKS,
    open: host => {
      setLedger({ ...NEWCOMER, starterIdeas: 'hidden', finishedTaskChats: FINISHED.slice(0, 1) })
      host.navigate(NEW_CHAT_ROUTE)
    }
  },
  {
    id: 'landing-settled',
    group: 'new',
    title: 'Settled landing',
    description: 'Two chats had a task finish and the check-in was answered: the ideas left on their own.',
    stores: MARKS,
    open: host => {
      setLedger({ ...NEWCOMER, checkIn: 'answered', finishedTaskChats: FINISHED })
      host.navigate(NEW_CHAT_ROUTE)
    }
  },
  {
    id: 'rename',
    group: 'returning',
    title: 'Hermes got a new look',
    description: 'The rename splash: the old names and where each one lives now. Opens on a new chat.',
    stores: MARKS,
    open: host => firstLook(host, { firstTimeSplash: 'not_applicable' }, 'existing')
  },
  {
    id: 'rename-reopened',
    group: 'returning',
    title: 'Rename splash, reopened',
    description: 'The rename splash as the tray brings it back after it was seen.',
    stores: MARKS,
    open: host => {
      setLedger(RETURNING)
      host.navigate(NEW_CHAT_ROUTE)
      reopenRenameSplash()
    }
  },
  {
    id: 'returning-landing',
    group: 'returning',
    title: 'Returning landing',
    description: 'A returning user\'s new chat once the splash is seen: no greeting and no ideas.',
    stores: MARKS,
    open: host => {
      setLedger(RETURNING)
      host.navigate(NEW_CHAT_ROUTE)
    }
  },
  {
    id: 'sign-up-required',
    group: 'signUp',
    title: 'Sign up required',
    description:
      'The sidebar\'s "Create an account" card, "Sign up to start" in the composer, the landing\'s Portal eyebrow ' +
      'and Sign up in the tray. Anything sent is held, never run.',
    stores: SIGN_UP_CHANGES,
    open: host => signUpLoad(host, NEWCOMER, pageAt(`#${NEW_CHAT_ROUTE}`))
  },
  {
    id: 'sign-up-splash',
    group: 'signUp',
    title: 'Welcome splash, sign up required',
    description: 'The first-time splash for a visitor: its second step offers Sign up and Look around (press Next).',
    stores: SIGN_UP_CHANGES,
    open: host => signUpLoad(host, { renameSplash: 'not_applicable' }, pageAt(`#${NEW_CHAT_ROUTE}`))
  },
  {
    id: 'held',
    group: 'signUp',
    title: 'A message held until sign up',
    description: '"I can do that. Sign up and I\'ll start on it right away." with a sample message held in its place.',
    stores: HELD_CHANGES,
    open: host => signUpLoad(host, NEWCOMER, pageAt(`#${NEW_CHAT_ROUTE}`), true)
  },
  {
    id: 'sign-up-page',
    group: 'signUp',
    title: 'Sign up, both ways',
    description:
      'Goes straight to the stand-in sign-up page with a message held, as Sign up would: finish it to come back ' +
      'signed in (the held message runs as a new chat), or come back without signing in.',
    stores: HELD_CHANGES,
    open: host => {
      const back = pageAt(`#${NEW_CHAT_ROUTE}`)
      signUpLoad(host, NEWCOMER, signUpHref(discoveryLabSignUpUrl(back), back), true)
    }
  },
  {
    id: 'recovery',
    group: 'signUp',
    title: 'Let\'s try signing in again',
    description: 'Sign up came back without signing the visitor in: the recovery screen, with the held message saved.',
    stores: HELD_CHANGES,
    open: host => signUpLoad(host, NEWCOMER, pageAt(`#${NEW_CHAT_ROUTE}`, 'failed'), true)
  },
  {
    id: 'usage-card',
    group: 'offers',
    title: 'Usage stats: sidebar card',
    description: 'The unanswered usage-stats question at the foot of the chat sidebar.',
    stores: `${MARKS} (the card's place); its "not answered yet" is a stand-in in memory, and an answer goes nowhere`,
    unavailable: signUpWaits,
    open: host => {
      usageStatsStandIn()
      host.navigate(NEW_CHAT_ROUTE)
    }
  },
  {
    id: 'usage-tray',
    group: 'offers',
    title: 'Usage stats: put away to the tray',
    description: 'The card\'s × moved the question to the notifications tray, unanswered. Opens the tray.',
    stores: `${MARKS}; the stand-in answer is in memory`,
    open: host => {
      usageStatsStandIn('deferred')
      host.navigate(NEW_CHAT_ROUTE)
      openTray(host)
    }
  },
  {
    id: 'usage-dialog',
    group: 'offers',
    title: 'Usage stats: details dialog',
    description: 'The short explainer Details opens, which can answer the question.',
    stores: `${MARKS}; the stand-in answer is in memory`,
    open: host => {
      usageStatsStandIn()
      host.navigate(NEW_CHAT_ROUTE)
      openUsageStatsDetails()
    }
  },
  {
    id: 'check-in',
    group: 'offers',
    title: 'How\'s Hermes going so far?',
    description: 'The check-in on a newcomer\'s new chat, due once two chats had a task finish.',
    stores: MARKS,
    open: host => {
      setLedger({ ...NEWCOMER, finishedTaskChats: FINISHED })
      host.navigate(NEW_CHAT_ROUTE)
    }
  },
  {
    id: 'check-in-tray',
    group: 'offers',
    title: 'Check-in, put away',
    description: 'Dismiss put the check-in in the tray, where Open brings it back. Opens the tray.',
    stores: MARKS,
    open: host => {
      setLedger({ ...NEWCOMER, checkIn: 'dismissed', finishedTaskChats: FINISHED })
      host.navigate(NEW_CHAT_ROUTE)
      openTray(host)
    }
  },
  {
    id: 'developer-view',
    group: 'offers',
    title: 'Want the developer view?',
    description: 'Over the composer of a chat whose answer held code, for someone in Simple.',
    stores: MARKS,
    unavailable: facts =>
      !facts.simpleMode
        ? 'Shown only in Simple: set Interface mode to Simple in Settings › Appearance first.'
        : facts.codeChatId
          ? null
          : 'Open a chat with a code answer in this window first; then it can open here.',
    open: (host, facts) => {
      const { developerViewOffer: _, ...ledger } = $guidanceLedger.get()
      setLedger(ledger)

      if (facts.codeChatId) {
        host.navigate(sessionRoute(facts.codeChatId))
      }
    }
  },
  {
    id: 'new-version-notice',
    group: 'updates',
    title: 'A newer build is waiting',
    description:
      'The notice over a stored chat\'s composer. Refresh now reloads; so does opening a new chat while it waits, ' +
      'as it would for a real build.',
    unavailable: needsStoredChat,
    open: (host, facts) => {
      clearScreen()
      $newVersionReady.set(true)

      if (facts.newestChatId) {
        host.navigate(sessionRoute(facts.newestChatId))
      }
    }
  },
  {
    id: 'tray-newcomer',
    group: 'tray',
    title: 'A newcomer\'s tray',
    description: '"Meet Hermes", the check-in put away and the usage-stats question put away (and Sign up while required).',
    stores: `${MARKS}; the stand-in answer is in memory`,
    open: host => {
      setLedger({ ...NEWCOMER, checkIn: 'dismissed', finishedTaskChats: FINISHED, usageStatsOffer: 'deferred' })
      $discoveryLabUsageStatsConsent.set(STAND_IN_CONSENT)
      host.navigate(NEW_CHAT_ROUTE)
      openTray(host)
    }
  },
  {
    id: 'tray-returning',
    group: 'tray',
    title: 'A returning user\'s tray',
    description: 'The rename splash\'s row with Open introduction, the usage-stats question and a newer build waiting.',
    stores: `${MARKS}; the stand-in answer and the newer build are in memory`,
    unavailable: needsStoredChat,
    open: (host, facts) => {
      setLedger({ ...RETURNING, usageStatsOffer: 'deferred' })
      $discoveryLabUsageStatsConsent.set(STAND_IN_CONSENT)
      $newVersionReady.set(true)

      // Over a stored chat: a new chat would load the waiting build.
      if (facts.newestChatId) {
        host.navigate(sessionRoute(facts.newestChatId))
      }

      openTray(host)
    }
  }
]

/** Start over as someone brand new or as someone who used Hermes before: every mark cleared, the held
 *  message and the in-memory stand-ins dropped, and their first visit replayed on a new chat. */
export function resetPerson(host: LabHost, who: 'new' | 'returning'): void {
  takeHeldPrompt()
  firstLook(host, {}, who === 'new' ? 'fresh' : 'existing')
}

/** Turn the sign-up stand-in off: the next load is signed in again, as it is without the lab. */
export function endSignUpStandIn(host: LabHost): void {
  setDiscoveryLabSignUp(false)
  loadAt(host, pageAt(`#${NEW_CHAT_ROUTE}`))
}
