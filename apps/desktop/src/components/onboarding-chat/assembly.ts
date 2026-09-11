/**
 * The layout assembly for in-chat onboarding.
 *
 * The guided chat starts SOLO: just the chat pane in a small window — no
 * sidebar, nothing to explain. When the user picks a layout in the
 * ::onboarding card, the app assembles around the conversation.
 *
 * The OS window grows OUTWARD by the MINIMUM each layout needs — the
 * sidebar's width to the left, the terminal/rail minimums where a layout
 * has them — animated (macOS setBounds animate), so the chat stays roughly
 * where it was and the window ends as small as the layout allows, but never
 * so small that the sidebar it just docked pops back out as a floating
 * Sheet.
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

/** True from guide kickoff until the layout pick assembles the app. */
export const $chatOnboardingSolo = atom(false)

// Presence mirror — see onboarding-presence.ts (update toast stands down).
$chatOnboardingSolo.subscribe(solo => setOnboardingSurfaceActive('solo-chat', solo))

/** The guided-setup session's ids — stored AND runtime, because consumers key
 *  sessions differently (the thread list by stored id, the composer by runtime
 *  id). That one thread gets the onboarding transcript treatment and drops the
 *  composer's git strip; every other session is untouched. */
export const $chatOnboardingThreadIds = atom<readonly string[]>([])

/** The opening line of the guided chat — PRE-BANKED, never generated. The
 *  first thing a new user sees must be instant; the model's cold-stack first
 *  turn took up to 10 seconds in live runs. The transcript renders this
 *  client-side the moment the chat opens; the model is told what was said
 *  and picks up from the user's answer. The lines themselves live in the i18n
 *  catalog (`guidedGreeting`) so they arrive in the user's language. */

export const $onboardingGreeting = atom('')

/** Pick (and remember) the canned opening line for this run. When the host
 *  reports a suggestable account name (machineUserName), the greeting ends by
 *  offering it as a default — \"or I can just call you akp\". The suggestion
 *  rides the SAME word the seed rows bank, so the canonical row, the typed
 *  reveal, and what the runbook says was said can never disagree.
 *
 *  Read from the i18n catalog rather than a local constant: the runbook tells
 *  the model to answer in the user's own language from its first real turn, so
 *  an English opener on a Japanese machine would be two different agents in
 *  two consecutive messages. Locales without their own copy fall back to
 *  English per-key through defineLocale, which is the same trade every other
 *  string in the app makes. */
export function pickOnboardingGreeting(): string {
  const existing = $onboardingGreeting.get()

  if (existing) {
    return existing
  }

  const copy = TRANSLATIONS[getRuntimeI18nLocale()].guidedGreeting
  const lines = copy.lines.length > 0 ? copy.lines : TRANSLATIONS.en.guidedGreeting.lines
  const line = lines[Math.floor(Math.random() * lines.length)] ?? lines[0] ?? ''
  const suggested = machineUserName()

  $onboardingGreeting.set(suggested ? `${line}\n\n${copy.nameSuggestion(suggested)}` : line)

  return $onboardingGreeting.get()
}

/** Whether the layout card's pick happened. A STORE, not card-local state:
 *  applying the layout replaces the pane tree, which remounts the chat pane
 *  and the card with it — component state would forget the selection the
 *  moment it takes effect. */
export const $chatLayoutPicked = atom(false)

let previousLayout: { id: string; tree: LayoutNode | null } | null = null

export function startChatOnboardingSolo(): void {
  if (!isOnboardingEnabled() || $chatOnboardingSolo.get()) {
    return
  }

  previousLayout = { id: $activePresetId.get(), tree: $layoutTree.get() }
  $chatOnboardingSolo.set(true)
  $chatLayoutPicked.set(false)
  // Bank the opening line the moment the solo chat owns the screen. The
  // machine profile is one local IPC; everything after it in the kickoff —
  // the setup profile, its backend boot, the session create — takes seconds
  // to minutes, and the greeting must be typing in the whole time instead of
  // the empty-draft wordmark. The kickoff picks again and gets this same
  // banked line (pick is first-write-wins).
  void loadMachineProfile().then(() => {
    if ($chatOnboardingSolo.get()) {
      pickOnboardingGreeting()
    }
  })
  // One zone, strip pinned off. applyTree ADOPTS panes the preset doesn't
  // declare (sessions, terminal, …) into this group as tabs — with the strip
  // never shown and workspace active, they're simply invisible until the
  // assembled layout re-places them. That adoption is also why reactive
  // unhides (files on cwd-arrival) can't pop a zone open mid-flow: there is
  // no other zone to open.
  applyLayoutPreset('chat-solo', group(['workspace'], { tabStrip: 'never' }))
}

/** A failed kickoff releases the screen so classic onboarding can resume. */
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

/** Minimal per-edge growth per layout — the least the window must gain for
 *  the new panes to be usable, NOT a chat-size-preserving projection (which
 *  balloons the window). Left = sessions sidebar; Elite adds its right rail
 *  and terminal row. Tune by feel. */
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

/**
 * Put the tree in the state this layout describes — on the first pick AND on
 * every re-pick.
 *
 * All of it has to re-run, because all of it persists: dismissals, dock
 * enforcement, the sidebar's open state. A re-pick that only swapped the
 * preset tree inherited the previous layout's records and came up as a mix of
 * the two (Elite after Basic kept Basic's terminal dismissal, so Elite's
 * terminal was placed and invisible).
 */
function reconcileLayout(id: string, tree: LayoutNode): void {
  applyLayoutPreset(id, tree)

  const declared = new Set(allPaneIds(tree))

  // Everything this layout asks for is wanted, whatever the last one decided.
  undismissTreePanes(declared)

  // The preset IS the layout. Adoption otherwise keeps every pane the preset
  // doesn't declare, and during a first run each of them is a surface the
  // user has no idea exists: a Terminal tab beside the chat on Basic, an
  // empty Cronjobs column, a Bots roster tabbed onto Sessions (its dock is
  // `enforce: true`, so it re-homes there on every pick). That last one also
  // costs the sidebar its plain face — two panes in the left zone is what
  // conjures a tab strip over what should just be the sessions list.
  //
  // So: anything the picked layout didn't ask for is dismissed. The pane
  // isn't gone, only unplaced — its own toggle (⌃` for the terminal, the
  // Layout menu, `revealTreePane`) brings it back the moment the user wants
  // it.
  //
  // Candidates come from the REGISTRY, not just the tree: a pane that isn't
  // placed yet still gets its dismissal recorded, and adoption skips dismissed
  // panes — so this holds whether the pane arrives before or after the sweep.
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

  // The tree now HAS a sessions column, but the renderer drops the whole left
  // column when the persisted ⌘B state says closed ($sidebarOpen →
  // $collapsedTreeSides) — picking a layout with a sidebar is an explicit
  // intent to see it, so open the side through its store (truthful toggle),
  // the same way resetLayoutTree reopens bound sides.
  setSidebarOpen(true)

  // Dock invariants normally run once at boot, against whatever tree existed
  // then — the SOLO tree, which has no sessions column for a left-docking
  // pane to anchor to. That pass burns the ledger entry, so re-running
  // adoption alone left those panes stranded as tabs in the chat zone. Reopen
  // the window first, now that the layout they should dock into exists.
  resetEnforcedDocks()
  adoptContributedPanes()

  // LAST, because panes can be a CONSEQUENCE of the assembly above: a plugin
  // registers more panes the moment one of its own becomes visible. Sweeping
  // before that point swept a tree those arrivals had not happened in yet,
  // and Basic still landed with an empty Cronjobs column beside the chat.
  dismissUndeclared()
}

/**
 * A layout pick, from the chat card. The FIRST one also performs the solo→app
 * transition (see module header); later picks re-arrange the app that is
 * already there.
 *
 * The window is grown once, on that first pick. `grow` moves the edges OUTWARD
 * by a delta, so re-growing per pick would ratchet the window bigger every
 * time the user toggled between two layouts.
 */
export function assembleChatOnboarding(id: string, tree: LayoutNode): void {
  const firstPick = $chatOnboardingSolo.get()

  if (firstPick) {
    const growth = LAYOUT_GROWTH.get(id) ?? { left: 220 }

    window.hermesDesktop?.chatOnboarding?.grow({
      bottom: growth.bottom ?? 0,
      left: growth.left ?? 0,
      right: growth.right ?? 0,
      // Every layout here docks a sessions sidebar, and the deltas above are
      // measured against the panes, not against the viewport the sidebar needs
      // to stay docked. Growing the solo card by a sidebar's width can still
      // land under the breakpoint — at first-run zoom, Basic did — and the
      // sidebar then arrives as a floating Sheet over the chat instead of
      // sliding in beside it. Ask for the floor and let main clamp it to the
      // display.
      minWidth: DOCKED_SIDEBAR_MIN_PX,
      top: growth.top ?? 0
    })
  }

  reconcileLayout(id, tree)

  $chatOnboardingSolo.set(false)
}

/** Skip the guided setup: assemble the default layout so the user lands in
 *  the full app immediately, and mark onboarding done so nothing resumes it.
 *  The guided chat stays in the transcript — skipping is about ending the
 *  questionnaire, not destroying the conversation. */
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

/**
 * True while the user is inside the first-run story — the solo guided chat,
 * or any thread the flow owns afterwards (Setup's chat, the first
 * build). Chrome that would read as noise over those conversations checks
 * this: floating panels, the profile-swap spinner.
 *
 * Threads are matched on BOTH ids because consumers key sessions differently
 * (the thread list by stored id, the composer by runtime id).
 */
export function useOnboardingChatActive(): boolean {
  const solo = useStore($chatOnboardingSolo)
  const threadIds = useStore($chatOnboardingThreadIds)
  const runtimeId = useStore($activeSessionId)
  const storedId = useStore($selectedStoredSessionId)

  return (
    solo || (runtimeId != null && threadIds.includes(runtimeId)) || (storedId != null && threadIds.includes(storedId))
  )
}
