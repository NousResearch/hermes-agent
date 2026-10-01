export interface CommonTranslations {
  /** Shared-metrics consent: first-run dialog + Settings › Safety toggles. */
  sharedMetrics: {
    consentTitle: string
    consentBody: string
    whatIsCollected: string
    collectedIntro: string
    collectedActivity: string
    collectedModels: string
    collectedNames: string
    collectedMilestones: string
    collectedReliability: string
    collectedUsage: string
    collectedMachine: string
    installId: string
    consentWindow: string
    readDocs: string
    share: string
    local: string
    off: string
    changeLater: string
    saveFailed: string
    collectLabel: string
    collectDesc: string
    sendLabel: string
    sendDesc: string
    unavailable: string
    stripBody: string
    stripChoices: { share: string; local: string; off: string }
    stripDetails: string
  }

  common: {
    apply: string
    back: string
    save: string
    saving: string
    cancel: string
    change: string
    choose: string
    clear: string
    close: string
    collapse: string
    confirm: string
    connect: string
    connecting: string
    continue: string
    bots: string
    copied: string
    copy: string
    copyFailed: string
    delete: string
    docs: string
    done: string
    error: string
    expand: string
    failed: string
    formatJson: string
    free: string
    loading: string
    notSet: string
    refresh: string
    remove: string
    replace: string
    retry: string
    run: string
    send: string
    set: string
    skip: string
    update: string
    tryHint: (term: string) => string
    on: string
    off: string
  }

  billingBlock: {
    titleNous: string
    titleProvider: (provider: string) => string
    fallbackMessage: string
    openBilling: string
    addCredits: string
    dismiss: string
  }

  ui: {
    search: {
      clear: string
    }
    logs: {
      bottom: string
      search: string
      top: string
    }
    pagination: {
      label: string
      previous: string
      previousAria: string
      next: string
      nextAria: string
    }
    sidebar: {
      title: string
      description: string
      toggle: (open: boolean) => string
    }
  }
}
