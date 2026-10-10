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
async function waitFor(selector: string, timeoutMs = 6000, ready = () => true): Promise<boolean> {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    const visible =
      ready() &&
      [...document.querySelectorAll(selector)].some(node => {
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

/** The caller does not await this, so the tour does not delay the handoff. `onHandoffChat` is true while the
 *  started task chat is the one on screen: the tour waits for it to open and drops if the user moves on. */
export async function showHandoffTour(onHandoffChat: () => boolean): Promise<void> {
  if (!(await waitFor(RAIL, 6000, onHandoffChat))) {
    return
  }

  const [sessionsVisible, localModel, pillVisible] = await Promise.all([
    waitFor(SESSIONS, 1500),
    localStepModel(),
    waitFor(MODEL_PILL, 1500)
  ])

  const localVisible = localModel !== null && pillVisible
  const copy = (key: string, ...args: unknown[]) => translateNow(`handoffTour.${key}`, ...args)
  // Imported here instead of at the top: this module is reachable from the boot path through intro.ts
  // (finishGuidedOnboarding), and run-tour.ts keeps driver.js and its stylesheet out of that path.
  const { startTour } = await import('@/lib/tour')

  if (!onHandoffChat()) {
    return
  }

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
