// Desktop i18n type contract.
//
// `Translations` is the single source of truth for every translatable string
// surface. Each locale file (`en.ts`, `zh.ts`, …) must satisfy this interface,
// so a missing key is a compile error — that's the completeness guard for
// "full" coverage as more surfaces are migrated off hardcoded English.

export type Locale = 'en' | 'zh'

interface ModeOptionCopy {
  label: string
  description: string
}

export interface Translations {
  common: {
    save: string
    saving: string
    cancel: string
    close: string
    confirm: string
    delete: string
    refresh: string
    retry: string
    on: string
    off: string
  }

  boot: {
    ready: string
    desktopBootFailedWithMessage: (message: string) => string
    steps: {
      connectingGateway: string
      loadingSettings: string
      loadingSessions: string
      startingDesktopConnection: string
      startingHermesDesktop: string
    }
    errors: {
      backgroundExited: string
      backgroundExitedDuringStartup: string
      backendStopped: string
      desktopBootFailed: string
      gatewaySignInRequired: string
      ipcBridgeUnavailable: string
    }
    failure: {
      title: string
      description: string
      remoteTitle: string
      remoteDescription: string
      retry: string
      repairInstall: string
      useLocalGateway: string
      openLogs: string
      repairHint: string
      remoteSignInHint: string
      hideRecentLogs: string
      showRecentLogs: string
      signedInTitle: string
      signedInMessage: string
      signInIncompleteTitle: string
      signInIncompleteMessage: string
      signInFailed: string
      signInToRemoteGateway: string
      signInWithProvider: (provider: string) => string
      identityProvider: string
    }
  }

  notifications: {
    region: string
    hide: string
    show: string
    more: (count: number) => string
    clearAll: string
    dismiss: string
    details: string
    copyDetail: string
    copyDetailFailed: string
    backendOutOfDateTitle: string
    backendOutOfDateMessage: string
    updateHermes: string
    updateReadyTitle: string
    updateReadyMessage: (count: number) => string
    seeWhatsNew: string
    errors: {
      elevenLabsNeedsKey: string
      elevenLabsRejectedKey: string
      methodNotAllowed: string
      microphonePermission: string
      openaiRejectedApiKey: string
      openaiRejectedApiKeyWithStatus: (status: string) => string
      openaiTtsNeedsKey: string
    }
  }

  titlebar: {
    hideSidebar: string
    showSidebar: string
    search: string
    searchTitle: string
    swapSidebarSides: string
    swapSidebarSidesTitle: string
    hideRightSidebar: string
    showRightSidebar: string
    muteHaptics: string
    unmuteHaptics: string
    openSettings: string
  }

  language: {
    label: string
    description: string
    saving: string
    saveError: string
  }

  settings: {
    closeSettings: string
    exportConfig: string
    importConfig: string
    resetToDefaults: string
    resetConfirm: string
    exportFailed: string
    resetFailed: string
    nav: {
      gateway: string
      apiKeys: string
      mcp: string
      archivedChats: string
      about: string
    }
    sections: Record<string, string>
    searchPlaceholder: Record<'about' | 'config' | 'gateway' | 'keys' | 'mcp' | 'sessions', string>
    modeOptions: Record<'light' | 'dark' | 'system', ModeOptionCopy>
    appearance: {
      title: string
      intro: string
      colorMode: string
      colorModeDesc: string
      toolViewTitle: string
      toolViewDesc: string
      product: string
      productDesc: string
      technical: string
      technicalDesc: string
      themeTitle: string
      themeDesc: string
    }
    fieldLabels: Record<string, string>
    fieldDescriptions: Record<string, string>
    about: {
      heading: string
      version: (value: string) => string
      versionUnavailable: string
      updates: string
      checkNow: string
      checking: string
      seeWhatsNew: string
      releaseNotes: string
      onLatest: string
      installing: string
      cantUpdate: string
      cantReach: string
      tapCheck: string
      updateReady: (count: number) => string
      lastChecked: (age: string) => string
      justNowSuffix: string
      automaticUpdates: string
      automaticUpdatesDesc: string
      branchCommit: (branch: string, commit: string) => string
      never: string
      justNow: string
      minAgo: (count: number) => string
      hoursAgo: (count: number) => string
      daysAgo: (count: number) => string
    }
  }

  skills: {
    tabSkills: string
    tabToolsets: string
    all: string
    searchSkills: string
    searchToolsets: string
    refresh: string
    refreshing: string
    loading: string
    noSkillsTitle: string
    noSkillsDesc: string
    noToolsetsTitle: string
    noToolsetsDesc: string
    noDescription: string
    configured: string
    needsKeys: string
    toolsetsEnabled: (enabled: number, total: number) => string
    configureToolset: (label: string) => string
    toggleToolset: (label: string) => string
    skillsLoadFailed: string
    toolsetsRefreshFailed: string
    skillEnabled: string
    skillDisabled: string
    toolsetEnabled: string
    toolsetDisabled: string
    appliesToNewSessions: (name: string) => string
    failedToUpdate: (name: string) => string
  }

  agents: {
    close: string
    title: string
    subtitle: string
    emptyTitle: string
    emptyDesc: string
    running: string
    failed: string
    done: string
    streaming: string
    files: string
    moreFiles: (count: number) => string
    delegation: (index: number) => string
    workers: (count: number) => string
    workersActive: (count: number) => string
    agentsCount: (count: number) => string
    activeCount: (count: number) => string
    failedCount: (count: number) => string
    toolsCount: (count: number) => string
    filesCount: (count: number) => string
    updatedAgo: (age: string) => string
    ageNow: string
    ageSeconds: (seconds: number) => string
    ageMinutes: (minutes: number) => string
    ageHours: (hours: number) => string
    durationSeconds: (seconds: string) => string
    durationMinutes: (minutes: number, seconds: number) => string
    tokensK: (k: string) => string
    tokens: (value: number) => string
  }

  commandCenter: {
    close: string
    searchPlaceholder: string
    sections: Record<'sessions' | 'system' | 'usage', string>
    sectionDescriptions: Record<'sessions' | 'system' | 'usage', string>
    nav: Record<'newChat' | 'settings' | 'skills' | 'messaging' | 'artifacts', { title: string; detail: string }>
    sectionEntries: Record<'sessions' | 'system' | 'usage', { title: string; detail: string }>
    providerNavigate: string
    providerSessions: string
    refresh: string
    refreshing: string
    noResults: string
    pinSession: string
    unpinSession: string
    exportSession: string
    deleteSession: string
    noSessions: string
    gatewayRunning: string
    gatewayStopped: string
    hermesActiveSessions: (version: string, count: number) => string
    restartMessaging: string
    updateHermes: string
    actionRunning: string
    actionDone: string
    actionFailed: string
    actionStartedWaiting: string
    loadingStatus: string
    recentLogs: string
    noLogs: string
    days: (count: number) => string
    statSessions: string
    statApiCalls: string
    statTokens: string
    statCost: string
    actualCost: (cost: string) => string
    loadingUsage: string
    noUsage: (period: number) => string
    retry: string
    dailyTokens: string
    input: string
    output: string
    noDailyActivity: string
    topModels: string
    noModelUsage: string
    topSkills: string
    noSkillActivity: string
    actions: (count: string) => string
  }

  messaging: {
    search: string
    loading: string
    loadFailed: string
    states: Record<string, string>
    unknown: string
    hintPendingRestart: string
    hintGatewayStopped: string
    credentialsSet: string
    needsSetup: string
    gatewayStopped: string
    getCredentials: string
    openSetupGuide: string
    required: string
    recommended: string
    advanced: (count: number) => string
    noTokenNeeded: string
    enabled: string
    disabled: string
    unsavedChanges: string
    saving: string
    saveChanges: string
    saved: string
    replaceValue: string
    openDocs: string
    clearField: (key: string) => string
    enableAria: (name: string) => string
    disableAria: (name: string) => string
    platformEnabled: (name: string) => string
    platformDisabled: (name: string) => string
    restartToApply: string
    setupSaved: (name: string) => string
    restartToReconnect: string
    keyCleared: (key: string) => string
    setupUpdated: (name: string) => string
    failedUpdate: (name: string) => string
    failedSave: (name: string) => string
    failedClear: (key: string) => string
    fieldCopy: Record<string, { label?: string; help?: string; placeholder?: string }>
    platformIntro: Record<string, string>
  }

  profiles: {
    close: string
    nameHint: string
    title: string
    count: (count: number) => string
    loading: string
    newProfile: string
    noProfiles: string
    selectPrompt: string
    refresh: string
    refreshing: string
    default: string
    skills: (count: number) => string
    env: string
    defaultBadge: string
    rename: string
    copySetup: string
    copying: string
    modelLabel: string
    skillsLabel: string
    notSet: string
    soulDesc: string
    unsavedChanges: string
    loadingSoul: string
    emptySoul: string
    saving: string
    saveSoul: string
    deleteTitle: string
    deleteDescPrefix: string
    deleteDescMid: string
    deleteDescSuffix: string
    deleting: string
    createDesc: string
    nameLabel: string
    cloneFromDefault: string
    cloneFromDefaultDesc: string
    invalidName: (hint: string) => string
    nameRequired: string
    creating: string
    createAction: string
    renameTitle: string
    renameDescPrefix: string
    renameDescSuffix: string
    newNameLabel: string
    renaming: string
    created: string
    renamed: string
    deleted: string
    setupCopied: string
    soulSaved: string
    failedLoad: string
    failedDelete: string
    failedCopy: string
    failedLoadSoul: string
    failedSaveSoul: string
    failedCreate: string
    failedRename: string
  }

  cron: {
    close: string
    search: string
    refresh: string
    refreshing: string
    loading: string
    states: Record<string, string>
    deliveryLabels: Record<string, string>
    scheduleLabels: Record<string, string>
    scheduleHints: Record<string, string>
    days: Record<string, string>
    dayFallback: (value: string) => string
    everyDayAt: (time: string) => string
    weekdaysAt: (time: string) => string
    everyDayOfWeekAt: (day: string, time: string) => string
    monthlyOnDayAt: (dayOfMonth: string, time: string) => string
    topOfHour: string
    everyHourAt: (minute: string) => string
    active: (enabled: number, total: number) => string
    newCron: string
    createFirst: string
    emptyDescNew: string
    emptyDescSearch: string
    emptyTitleNew: string
    emptyTitleSearch: string
    last: string
    next: string
    actionsFor: (title: string) => string
    actionsTitle: string
    resume: string
    pause: string
    resumeTitle: string
    pauseTitle: string
    triggerNow: string
    edit: string
    deleteTitle: string
    deleteDescPrefix: string
    deleteDescSuffix: string
    deleting: string
    resumed: string
    paused: string
    triggered: string
    deleted: string
    created: string
    updated: string
    failedLoad: string
    failedUpdate: string
    failedTrigger: string
    failedDelete: string
    failedSave: string
    editTitle: string
    createTitle: string
    editDesc: string
    createDesc: string
    nameLabel: string
    namePlaceholder: string
    promptLabel: string
    promptPlaceholder: string
    frequencyLabel: string
    deliverLabel: string
    customScheduleLabel: string
    customPlaceholder: string
    customHint: string
    optional: string
    promptScheduleRequired: string
    saveChanges: string
    createAction: string
  }

  artifacts: {
    search: string
    refresh: string
    refreshing: string
    indexing: string
    tabAll: string
    tabImages: string
    tabFiles: string
    tabLinks: string
    noArtifactsTitle: string
    noArtifactsDesc: string
    failedLoad: string
    openFailed: string
    itemsImage: string
    itemsLink: string
    itemsFile: string
    itemsGeneric: string
    zero: string
    rangeOf: (start: number, end: number, total: number) => string
    goToPage: (itemLabel: string, page: number) => string
    colTitleLink: string
    colTitleFile: string
    colTitleDefault: string
    colLocationLink: string
    colLocationFile: string
    colLocationDefault: string
    colSession: string
    kindImage: string
    kindFile: string
    kindLink: string
    chat: string
    copyUrl: string
    copyPath: string
  }

  sidebar: {
    nav: Record<string, string>
    searchAria: string
    searchPlaceholder: string
    clearSearch: string
    noMatch: (query: string) => string
    results: string
    pinned: string
    sessions: string
    groupAriaGrouped: string
    groupAriaUngrouped: string
    groupTitleGrouped: string
    groupTitleUngrouped: string
    allPinned: string
    shiftClickHint: string
    noWorkspace: string
    newSessionIn: (label: string) => string
    reorderWorkspace: (label: string) => string
    showMoreIn: (count: number, label: string) => string
    loading: string
    loadMore: string
    loadCount: (step: number) => string
    row: {
      pin: string
      unpin: string
      copyId: string
      export: string
      rename: string
      archive: string
      copyIdFailed: string
      actionsFor: (title: string) => string
      sessionActions: string
      sessionRunning: string
      needsInput: string
      waitingForAnswer: string
      renamed: string
      renameFailed: string
      renameTitle: string
      renameDesc: string
      untitledPlaceholder: string
      ageNow: string
      ageDay: string
      ageHour: string
      ageMin: string
    }
  }

  composer: {
    message: string
    placeholderStarting: string
    placeholderReconnecting: string
    placeholderFollowUp: string
    newSessionPlaceholders: readonly string[]
    followUpPlaceholders: readonly string[]
    startVoice: string
    queueMessage: string
    stop: string
    send: string
    speaking: string
    transcribing: string
    thinking: string
    muted: string
    listening: string
    muteMic: string
    unmuteMic: string
    stopListening: string
    stopShort: string
    endConversation: string
    endShort: string
    stopDictation: string
    transcribingDictation: string
    voiceDictation: string
    commonCommands: string
    hotkeys: string
    helpFooter: string
    commandDescs: Record<string, string>
    hotkeyDescs: Record<string, string>
    attachUrlTitle: string
    attachUrlDesc: string
    urlPlaceholder: string
    urlHintPre: string
    attach: string
    queued: (count: number) => string
    attachmentOnly: string
    emptyTurn: string
    attachments: (count: number) => string
    editingInComposer: string
    editQueued: string
    sendQueuedNow: string
    deleteQueued: string
    previewUnavailable: string
    previewLabel: (label: string) => string
    couldNotPreview: (label: string) => string
    removeAttachment: (label: string) => string
    dictating: string
    preparingAudio: string
    speakingResponse: string
    readingAloud: string
    themeSuggestions: string
    noMatchingThemes: string
    themeTryPre: string
    themeTryPost: string
    attachLabel: string
    files: string
    folder: string
    images: string
    pasteImage: string
    url: string
    promptSnippets: string
    tipPre: string
    tipPost: string
    snippetsTitle: string
    snippetsDesc: string
    snippets: Record<string, { label: string; description: string; text: string }>
  }
}
