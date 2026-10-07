// Shell notice strings: the remote-display toast and the butterbar; `Translations`
// spreads this in.
export interface NoticeTranslations {
  remoteDisplayBanner: {
    message: (reason: string) => string
  }

  butterbar: {
    goTo: (index: number, total: number) => string
  }
}
