import { atom } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'

// One-time consent prompt for real-profile browsing, shown when a Browser
// pane opens while `browser.use_real_profile` is off. Mirrors the embed
// consent split: "Don't show again" persists across launches; "Not now"
// mutes for this app run only, so the offer comes back next launch rather
// than nagging within one.
export const $realProfilePromptDismissed = persistentAtom<boolean>(
  'hermes.desktop.real-profile-prompt-dismissed',
  false,
  Codecs.bool
)

/** "Not now" for this app run — session-scoped on purpose (see above). */
export const $realProfilePromptMuted = atom(false)

// Several Browser panes can be mounted at once (split zones, popout during a
// dock transition). The FIRST mounted pane claims the prompt; the rest render
// nothing, so one open never stacks N identical dialogs.
export const $realProfilePromptClaim = atom<null | string>(null)

export function claimRealProfilePrompt(id: string) {
  if ($realProfilePromptClaim.get() === null) {
    $realProfilePromptClaim.set(id)
  }
}

export function releaseRealProfilePrompt(id: string) {
  if ($realProfilePromptClaim.get() === id) {
    $realProfilePromptClaim.set(null)
  }
}
