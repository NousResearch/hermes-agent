// Shell notice strings: the remote-display toast and the butterbar; `Translations`
// spreads this in.
export interface NoticeTranslations {
  remoteDisplayBanner: {
    message: (reason: string) => string
  }

  /** "Discard unsaved changes?" for closing a file preview holding a draft. */
  previewDraft: {
    discardTitle: string
    discardBody: (label: string) => string
    discardConfirm: string
  }

  butterbar: {
    goTo: (index: number, total: number) => string
    legal: { before: string; terms: string; between: string; privacy: string; after: string }
  }
}
