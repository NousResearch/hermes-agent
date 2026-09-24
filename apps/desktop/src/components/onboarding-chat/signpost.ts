/**
 * The handoff tour: three steps shown when the build session first appears, because the profile changed under
 * the user without their asking. The user was talking to Hermes on its own profile and now sits mid-build in a
 * session of their own. Nothing on screen says where the welcome chat went, or that the sessions list now
 * belongs to a different profile.
 *
 * The guide cannot describe this itself: the tour bridge only runs a tour for the session the user is looking
 * at, and after the handoff the guide is a background session (desktop AGENTS.md requires offering rather than
 * taking over). The app runs the same three steps instead, in the user's language, and the guide's own note in
 * the welcome chat does not mention them.
 *
 * On a machine that can run a local model, a last step points at the model pill: that is where the offer lives
 * (the "Run locally" row at the top of its menu). Every guided user reaches this tour, so it is the reliable half
 * of the local-setup offer (store/local-setup-offer.ts). The step is left out when the machine does not qualify or
 * eligibility does not answer in time, rather than promising a model that won't run.
 */
import { translateNow } from '@/i18n'
import { readLocalSetupEligibility } from '@/store/local-setup-offer'

/** Tour handles (`data-tour`). A targets scan returns these same selectors, so a curated step and a
 *  model-driven step point at the same node. */
const RAIL = '[data-tour="profile-rail"]'
const SESSIONS = '[data-tour="sessions-sidebar"]'
const MODEL_PILL = '[data-tour="model-pill"]'

/** Eligibility is one status + catalog read, usually already cached. The tour does not wait longer than this. */
const LOCAL_FIT_WAIT_MS = 1500

/** Waits for a visible node. The profile rail mounts a render or two after the handoff switches profiles, and
 *  the tour engine returns a no-match for a selector that is not in the DOM yet. Returns false on timeout. */
async function waitFor(selector: string, timeoutMs = 6000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    const visible = [...document.querySelectorAll(selector)].some(node => {
      const { width, height } = node.getBoundingClientRect()

      return width > 0 && height > 0 && !node.closest('[data-pane-hidden]')
    })

    if (visible) {
      return true
    }

    await new Promise(resolve => setTimeout(resolve, 120))
  }

  return false
}

/** The caller does not await this, so the tour does not delay the handoff. */
export async function showHandoffTour(): Promise<void> {
  if (!(await waitFor(RAIL))) {
    return
  }

  const [sessionsVisible, localModel, pillVisible] = await Promise.all([
    waitFor(SESSIONS, 1500),
    localStepModel(),
    waitFor(MODEL_PILL, 1500)
  ])

  const localVisible = localModel !== null && pillVisible
  const copy = (key: string, ...args: unknown[]) => translateNow(`handoffTour.${key}`, ...args)
  // Imported here instead of at the top: this module is reachable from the boot path through the handoff
  // hook, and run-tour.ts keeps driver.js and its stylesheet out of that path.
  const { startTour } = await import('@/lib/tour')

  await startTour([
    { accent: true, selector: RAIL, side: 'right', text: copy('profileText'), title: copy('profileTitle') },
    ...(sessionsVisible
      ? [{ selector: SESSIONS, side: 'right' as const, text: copy('sessionsText'), title: copy('sessionsTitle') }]
      : []),
    { accent: true, selector: RAIL, side: 'right', text: copy('stayText'), title: copy('stayTitle') },
    ...(localVisible
      ? [
          {
            accent: true,
            selector: MODEL_PILL,
            side: 'top' as const,
            text: copy('localText', localModel),
            title: copy('localTitle')
          }
        ]
      : [])
  ])
}

/** Display name of the model this machine can run, or null when it doesn't qualify or the read is slow. */
async function localStepModel(): Promise<null | string> {
  let timer: ReturnType<typeof setTimeout> | undefined

  const timeout = new Promise<null>(resolve => {
    timer = setTimeout(() => resolve(null), LOCAL_FIT_WAIT_MS)
  })

  const fit = readLocalSetupEligibility().then(({ fit }) => fit?.model.display_name ?? null)

  try {
    return await Promise.race([fit, timeout])
  } finally {
    clearTimeout(timer)
  }
}
