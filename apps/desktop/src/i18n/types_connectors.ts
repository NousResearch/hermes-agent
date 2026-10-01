export interface ConnectorsTranslations {
  connectors: {
    title: string
    connect: string
    skip: string
    cancel: string
    retry: string
    grant: string
    connected: string
    checking: string
    notConnected: string
    skipped: string
    disabled: string
    failed: string
    needsAuth: string
    opening: string
    waiting: string
    timeout: string
    refresh: string
    connectError: string
    connectErrorFor: (app: string) => string
    unavailable: string
    ownerMissing: string
    search: string
    empty: string
    disclaimer: string
    execution: string
    setup: (server: string) => string
    openInBrowser: string
    setupCancel: string
    authorizedToolsUnavailable: string
    required: string
  }

  connectorsPage: {
    title: string
    searchPlaceholder: (count: number) => string
    filterCategory: string
    categoryAll: string
    uncategorised: string
    residencyLocal: string
    segment: {
      all: string
      available: string
      connected: string
      off: string
    }
    group: {
      connected: string
      connectedNote: string
      available: string
      off: string
      offNote: string
    }
    card: {
      kindManaged: string
      kindCatalog: string
      kindCustom: string
      kindPlugin: (plugin: string) => string
      inCatalog: string
      hostedTwin: string
      alsoLocal: string
      open: (name: string) => string
      turnServerOn: (name: string) => string
      turnServerOff: (name: string) => string
      state: {
        accessExpired: string
        available: string
        connected: string
        connecting: string
        connectionUnknown: string
        couldNotConnect: string
        offByYourOrganisation: string
        offForYou: string
        serverConnecting: string
        serverError: string
        serverNeedsAuth: string
        serverOff: string
        serverOn: string
        serverOnUnused: string
      }
      fact: {
        tools: (count: number) => string
        toolsOff: (count: number) => string
        toolsOn: (count: number) => string
        toolsSomeOn: (total: number, on: number) => string
      }
      verb: {
        authenticate: string
        connect: string
        install: string
        openLogs: string
        reconnect: string
        stopWaiting: string
        tryAgain: string
        turnBackOn: string
      }
      reason: {
        finishSignIn: string
        reconnect: string
        serverError: string
        serverNeedsAuth: string
      }
    }
    page: {
      loading: string
      emptyTitle: string
      noMatchTitle: string
      noMatchBody: string
      clearSearch: string
      hostedFailedTitle: string
      hostedFailedBody: string
      retry: string
      matchesElsewhere: (count: number) => string
      showAllMatches: string
      segmentNoMatch: (segment: string) => string
      freeTierNote: string
      signInLine: string
      signIn: string
      managedUnavailable: string
      writeFailed: string
      refreshFailed: string
      disconnectNoAccount: string
      disconnectRefused: string
    }
    add: {
      action: string
      title: string
      hint: string
      pasteLabel: string
      pastePlaceholder: string
      pasteNoMatch: string
      name: string
      nameTaken: string
      type: string
      typeStdio: string
      typeHttp: string
      command: string
      args: string
      addArg: string
      envVars: string
      addEnvVar: string
      passthrough: string
      addPassthrough: string
      cwd: string
      url: string
      headers: string
      addHeader: string
      auth: string
      authNone: string
      authOauth: string
      authBearer: string
      keyPlaceholder: string
      valuePlaceholder: string
      removeRow: string
      editJson: string
      saveFailed: string
    }
    dialog: {
      disconnect: string
      disconnectTitle: (name: string) => string
      disconnectBody: string
      menuRefreshTools: string
      moreActions: string
      removeServerTitle: (name: string) => string
      removeServerBody: string
      appSwitch: (name: string) => string
      waysTitle: (name: string) => string
      wayNotConnected: (name: string) => string
      wayHosted: string
      bothOn: (name: string) => string
      turnOffLocal: string
      providedByPlugin: (plugin: string) => string
      openPlugins: string
      nousLine: string
      rulesReadOnly: string
      rulesAppOff: (name: string) => string
      rulesSignIn: string
      orgNote: (count: number) => string
      orgLink: string
      connectEnded: string
      connectOpenAgain: string
      tokensPerCall: string
      usesPerMonth: string
      advanced: string
      advancedHint: string
    }
    tools: {
      title: string
      notInstalledBody: string
      summaryTitle: (name: string) => string
      summaryPreviewTitle: (name: string) => string
      summaryCount: (count: number) => string
      summaryAllTools: string
      summaryOther: string
      allToolsSwitch: string
      summaryAllOn: string
      summarySomeOn: (on: number, total: number) => string
      summaryOff: string
      showAllTools: (count: number) => string
      showSummary: string
      facetSwitch: (facet: string) => string
      moreHints: (count: number) => string
      staleSignIn: string
      searchCountPlaceholder: (count: number) => string
      toolList: (name: string) => string
      categorySelect: (count: number) => string
      showDeprecated: (count: number) => string
      hideDeprecated: (count: number) => string
      quickReadOnly: string
      quickNoDestructive: string
      quickEverythingOn: string
      lockedHint: string
      turnToolOn: (tool: string) => string
      turnToolOff: (tool: string) => string
      showDetails: (tool: string) => string
      hideDetails: (tool: string) => string
      noMatch: string
      loading: string
      unavailableLine: string
      needsAuthTitle: (name: string) => string
      needsAuthBody: string
      retry: string
      goneTitle: (name: string) => string
      goneBody: string
      remove: string
      offTitle: (name: string) => string
      offBody: string
      signedOutTitle: string
      signedOutBody: string
      conflictTitle: string
      conflictBody: (theyOff: number, theyOn: number) => string
      conflictReload: string
      conflictSave: string
      saveFailed: string
      footerDirty: (off: number, backOn: number) => string
      discard: string
      save: string
      saving: string
    }
    vocabulary: Record<
      | 'facetDestructive'
      | 'facetRead'
      | 'facetUnclassified'
      | 'facetWrite'
      | 'hintCreate'
      | 'hintDelete'
      | 'hintDestructive'
      | 'hintIdempotent'
      | 'hintOpenWorld'
      | 'hintReadOnly'
      | 'hintUpdate',
      { label: string; long: string }
    >
  }

  sessionImport: {
    title: string
    subtitle: string
    action: string
    readingFrom: string
    connectedComputer: string
    destination: string
    all: string
    search: string
    scanning: string
    scanError: string
    scanHelp: string
    empty: string
    emptyHelp: string
    noMatches: string
    searchHelp: string
    skipped: string
    more: string
    messages: string
    choose: string
    chooseHelp: string
    previewLoading: string
    previewError: string
    previewHelp: string
    previewLimit: string
    you: string
    snapshot: string
    copyNotice: string
    importing: string
    open: string
    continue: string
    importError: string
  }
}
