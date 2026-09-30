import { atom } from 'nanostores'

import { setOnboardingSurfaceActive } from './onboarding-presence'

/**
 * The first-run intro as this window shows it, in memory only: whether it runs at all is the
 * backend's `onboarding.state`. `starting` holds the boot overlay while the setup chat opens,
 * `intro` is the setup chat in the demo layout, `ended` means the user left it (or finished),
 * and `off` means it never ran or failed to start. Only the main window moves out of `off`.
 */
export type IntroView = 'off' | 'starting' | 'intro' | 'ended'

export const $introView = atom<IntroView>('off')

/** The demo layout is on screen: the chat alone, narrower, minimal composer, no status bar. Kept here,
 *  free of layout imports, so the shell can read it; onboarding-chat/assembly.ts owns the switch. */
export const $chatOnboardingSolo = atom(false)

// While the intro runs, the provider picker, the free-tier ready screen and the tips wait for it.
$introView.subscribe(view => setOnboardingSurfaceActive('intro', view === 'starting' || view === 'intro'))
