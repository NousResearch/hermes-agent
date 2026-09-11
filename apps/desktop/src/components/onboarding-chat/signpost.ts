/**
 * THE HANDOFF TOUR — three steps, at the one moment the ground moves.
 *
 * The handoff is the only point in the run where the user changes profile
 * without asking to: they were talking to Hermes on its own profile, and they
 * land mid-build in a session of their own. Nothing on screen says where the
 * welcome chat went, or that the sessions list they now see belongs to a
 * different profile than the one they were in a moment ago.
 *
 * The guide cannot narrate this itself: the tour bridge only paints for the
 * session the user is looking at, and after the handoff the guide is a
 * background session (desktop AGENTS.md: offer, don't hijack). So the app runs
 * the same three steps the guide would have asked for, in the user's language,
 * and the guide's one line in its own chat says nothing about them.
 */
import { translateNow } from '@/i18n'

/** Tour handles (`data-tour`), the same selectors the model gets back from a
 *  targets scan, so a curated step and a model-driven one point at one thing. */
const RAIL = '[data-tour="profile-rail"]'
const SESSIONS = '[data-tour="sessions-sidebar"]'

/** The rail mounts a render or two after the handoff swaps profiles, so wait
 *  for the node rather than firing into an empty DOM (the engine would return
 *  a no-match and the moment would pass silently). Gives up quietly. */
async function waitFor(selector: string, timeoutMs = 6000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if (document.querySelector(selector)) {
      return true
    }

    await new Promise(resolve => setTimeout(resolve, 120))
  }

  return false
}

/** Run the handoff tour. Never throws, never blocks the handoff. */
export async function showHandoffTour(): Promise<void> {
  if (!(await waitFor(RAIL)) || !(await waitFor(SESSIONS, 1500))) {
    return
  }

  const copy = (key: string) => translateNow(`handoffTour.${key}`)
  // Imported here, not at the top: this module is reachable from the boot path
  // through the handoff hook, and driver.js plus its stylesheet are exactly
  // what run-tour.ts keeps off it.
  const { startTour } = await import('@/lib/tour')

  await startTour([
    { accent: true, selector: RAIL, side: 'right', text: copy('profileText'), title: copy('profileTitle') },
    { selector: SESSIONS, side: 'right', text: copy('sessionsText'), title: copy('sessionsTitle') },
    { accent: true, selector: RAIL, side: 'right', text: copy('stayText'), title: copy('stayTitle') }
  ])
}
