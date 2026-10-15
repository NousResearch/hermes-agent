import type { Translations } from './types'

// Shell notices (remote-display toast, butterbar, keep-awake offer), spread into en.ts.
export const enNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Software rendering active — remote display detected (${reason}). GPU acceleration is disabled to prevent flickering.`
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
  },
  keepAwakeOffer: {
    title: 'Your computer slept while Hermes was working',
    message:
      'Hermes can keep your computer awake while a task runs. The screen still turns off, and on battery your computer may still sleep.',
    action: 'Keep awake while working'
  }
} satisfies Pick<Translations, 'remoteDisplayBanner' | 'butterbar' | 'keepAwakeOffer'>
