// Shell notice strings: the remote-display toast, the butterbar and send-outcome toasts;
// `Translations` spreads this in.
export interface NoticeTranslations {
  remoteDisplayBanner: {
    message: (reason: string) => string
  }

  butterbar: {
    goTo: (index: number, total: number) => string
    legal: { before: string; terms: string; between: string; privacy: string; after: string }
  }

  promptNotices: {
    /** An explicit retry of a send whose identityless (legacy) acknowledgement was lost. */
    legacySendUnconfirmed: string
  }
}
