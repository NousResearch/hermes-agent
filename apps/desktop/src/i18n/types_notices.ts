// Shell notice strings: the remote-display toast, the butterbar and the keep-awake offer; `Translations`
// spreads this in.
export interface NoticeTranslations {
  remoteDisplayBanner: {
    message: (reason: string) => string
  }

  butterbar: {
    goTo: (index: number, total: number) => string
    legal: { before: string; terms: string; between: string; privacy: string; after: string }
  }

  /** Once per computer, after a timed sleep interrupted a turn with keep-awake off. */
  keepAwakeOffer: { title: string; message: string; action: string }
}
