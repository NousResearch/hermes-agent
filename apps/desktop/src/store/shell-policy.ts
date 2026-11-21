import { computed, type ReadableAtom } from 'nanostores'

import { $portalPreferred } from './shell-variant'

// Every BEHAVIOUR the redesign switch changes, one named gate each, so this file is the
// complete answer to "what does turning the redesign off do besides restyle?". Presentation
// reads `usePortalShell()` / `$portalShell` (per window); behaviour reads these, which follow the
// window-independent preference so a HUD or pop-out never runs the other shell's flow. Each gate
// is reactive: a runtime flip re-runs its consumers in both directions without a reload.

const gate = (enabled: (portal: boolean) => boolean): ReadableAtom<boolean> => computed($portalPreferred, enabled)

/** First run is the setup guide, not the classic intro + onboarding chat. Without it, the classic
 *  run's chat-scoped rules apply: onboarding-only cards, no slash completions or commands in that
 *  chat, an agent-applied layout or a palette run ending the intro. */
export const $setupGuideOnboarding = gate(portal => portal)

/** The sidebar lists every profile's sessions: the agents list is the filter, not a gateway swap. */
export const $sessionListSpansProfiles = gate(portal => portal)

/** Capabilities is a read-only inventory; installs and toggles live in each agent's settings. */
export const $capabilitiesManagedPerAgent = gate(portal => portal)

/** Bots are a section of the Agents page, so the Bots plugin registers no sidebar pane. */
export const $botsInAgentsPage = gate(portal => portal)

/** Deleting a chat leaves the toast stack alone; classic dismisses every toast on a delete. */
export const $deleteKeepsNotifications = gate(portal => portal)

/** Navigation (IA) metrics measure the redesigned information architecture only. */
export const $recordsIaMetrics = gate(portal => portal)

/** A send the gateway refuses is rolled back whole: its bubble leaves the thread, the composer holds
 *  its words, and one toast says why. Classic keeps the bubble with an in-thread error card. */
export const $failedSendRollsBack = gate(portal => portal)
