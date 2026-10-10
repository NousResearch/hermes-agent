import type { Translations } from './types'

// Shell notices (remote-display toast, butterbar), spread into en.ts.
export const enNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Software rendering active — remote display detected (${reason}). GPU acceleration is disabled to prevent flickering.`
  },
  previewDraft: {
    discardTitle: 'Discard unsaved changes?',
    discardBody: label => `${label} has edits you haven't saved. Closing the tab throws them away.`,
    discardConfirm: 'Discard changes'
  },
  butterbar: {
    goTo: (index, total) => `Show notice ${index} of ${total}`,
    legal: {
      before: 'Use of Hermes Agent is subject to our ',
      terms: 'Terms of Service',
      between: ' and ',
      privacy: 'Privacy Policy',
      after: '.'
    }
  }
} satisfies Pick<Translations, 'remoteDisplayBanner' | 'previewDraft' | 'butterbar'>
