// Desktop i18n type contract.
//
// `Translations` is the single source of truth for every translatable string
// surface. Fully translated locale files may satisfy this interface directly;
// partial locales should use `defineLocale()` so missing desktop-only strings
// fall back to English while new keys remain type-checked.

export type Locale = 'en' | 'zh' | 'zh-hant' | 'ja'

interface ModeOptionCopy {
  label: string
  description: string
}

interface AuxTaskCopy {
  label: string
  hint: string
}

interface ToolTitleCopy {
  done: string
  pending: string
}

interface UninstallOptionCopy {
  title: string
  description: string
  consequence: string
}

export interface Translations {
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
    copied: string
    copy: string
    copyFailed: string
    delete: string
    docs: string
    done: string
    error: string
    failed: string
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
      // Friendly, user-facing replacements for common raw bootstrap errors
      // (the raw transcript stays available behind the "show recent logs"
      // expander). `unknown` is the generic fallback when no pattern matches.
      errorMap: {
        cancelled: string
        prerequisites: string
        network: string
        unknown: string
      }
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
    voice: {
      configureSpeechToText: string
      couldNotStartSession: string
      microphoneAccessDenied: string
      microphoneConstraintsUnsupported: string
      microphoneFailed: string
      microphoneInUse: string
      microphonePermissionDenied: string
      microphoneStartFailed: string
      microphoneUnsupported: string
      noMicrophone: string
      noSpeechDetected: string
      playbackFailed: string
      recordingFailed: string
      transcriptionFailed: string
      transcriptionUnavailable: string
      tryRecordingAgain: string
      unavailable: string
    }
    // Native OS notification copy (titles + generic fallback bodies). Dynamic
    // bodies (the agent's reply, a command, an error) are passed through raw.
    native: {
      approvalTitle: string
      approveAction: string
      rejectAction: string
      inputTitle: string
      inputBody: string
      turnDoneTitle: string
      turnDoneBody: string
      turnErrorTitle: string
      backgroundDoneTitle: string
      backgroundFailedTitle: string
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
    openKeybinds: string
  }

  keybinds: {
    title: string
    subtitle: (open: string) => string
    rebind: string
    reset: string
    resetAll: string
    pressKey: string
    set: string
    conflictWith: (label: string) => string
    categories: Record<string, string>
    actions: Record<string, string>
  }

  language: {
    label: string
    description: string
    saving: string
    saveError: string
    switchTo: string
    searchPlaceholder: string
    noResults: string
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
      providers: string
      providerAccounts: string
      providerApiKeys: string
      gateway: string
      apiKeys: string
      keysTools: string
      keysSettings: string
      mcp: string
      archivedChats: string
      about: string
      notifications: string
    }
    notifications: {
      title: string
      intro: string
      enableAll: string
      enableAllDesc: string
      focusedHint: string
      kinds: Record<
        'approval' | 'backgroundDone' | 'input' | 'turnDone' | 'turnError',
        { label: string; description: string }
      >
      test: string
      testTitle: string
      testBody: string
      testSent: string
      testUnsupported: string
      completionSoundTitle: string
      completionSoundDesc: string
      completionSoundPreview: string
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
      translucencyTitle: string
      translucencyDesc: string
      haptics: string
      hapticsDesc: string
      product: string
      productDesc: string
      technical: string
      technicalDesc: string
      themeTitle: string
      themeDesc: string
      themeProfileNote: (profile: string) => string
      installTitle: string
      installDesc: string
      installPlaceholder: string
      installButton: string
      installing: string
      installError: string
      installed: (name: string) => string
      removeTheme: string
      importedBadge: string
    }
    // 个性化 — the consumer landing section (人格 picker + SOUL.md + the former
    // About content).
    personalization: {
      personalityTitle: string
      personalityIntro: string
      soulTitle: string
      soulIntro: string
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
      // Engine (runtime) opt-in update — R5/R6 of runtime 3-end consistency.
      engineSection: string
      engineVersion: (value: string) => string
      engineVersionUnavailable: string
      engineCheck: string
      engineChecking: string
      engineUpToDate: string
      engineTapCheck: string
      engineFound: (value: string) => string
      engineFoundGeneric: string
      engineCompatNotes: string
      engineApply: string
      engineApplying: string
      engineCantReach: string
      engineConfirmTitle: string
      engineConfirmBody: (value: string) => string
      engineConfirmBodyGeneric: string
      engineConfirmApply: string
    }
    config: {
      none: string
      noneParen: string
      notSet: string
      commaSeparated: string
      loading: string
      emptyTitle: string
      emptyDesc: string
      failedLoad: string
      autosaveFailed: string
      imported: string
      invalidJson: string
    }
    credentials: {
      pasteKey: string
      pasteLabelKey: (label: string) => string
      optional: string
      enterValueFirst: string
      couldNotSave: string
      remove: string
      or: string
      escToCancel: string
      getKey: string
      saving: string
    }
    envActions: {
      actionsFor: (label: string) => string
      credentialActions: string
      docs: string
      hideValue: string
      revealValue: string
      replace: string
      set: string
      clear: string
    }
    gateway: {
      loading: string
      unavailableTitle: string
      unavailableDesc: string
      title: string
      envOverride: string
      intro: string
      appliesTo: string
      allProfiles: string
      defaultConnection: string
      profileConnection: (profile: string) => string
      envOverrideTitle: string
      envOverrideDesc: string
      localTitle: string
      localDesc: string
      remoteTitle: string
      remoteDesc: string
      remoteUrlTitle: string
      remoteUrlDesc: string
      probing: string
      probeError: string
      signedIn: string
      signIn: string
      signOut: string
      signInWith: (provider: string) => string
      authTitle: string
      authSignedInPassword: string
      authSignedInOauth: string
      authNeedsPassword: string
      authNeedsOauth: (provider: string) => string
      tokenTitle: string
      tokenDesc: string
      existingToken: (value: string) => string
      savedToken: string
      pasteSessionToken: string
      testRemote: string
      saveForRestart: string
      saveAndReconnect: string
      diagnostics: string
      diagnosticsDesc: string
      openLogs: string
      incompleteTitle: string
      incompleteSignIn: string
      incompleteToken: string
      incompleteSignInTest: string
      incompleteTokenTest: string
      enterUrlFirst: string
      restartingTitle: string
      savedTitle: string
      restartingMessage: string
      savedMessage: string
      connectedTo: (baseUrl: string, version?: string) => string
      reachableTitle: string
      signedOutTitle: string
      signedOutMessage: string
      failedLoad: string
      signInFailed: string
      signOutFailed: string
      testFailed: string
      applyFailed: string
      saveFailed: string
    }
    keys: {
      loading: string
      failedLoad: string
      empty: string
    }
    mcp: {
      loading: string
      failedLoad: string
      nameRequiredTitle: string
      nameRequiredMessage: string
      objectRequired: string
      invalidJson: string
      saveFailed: string
      removeFailed: string
      gatewayUnavailableTitle: string
      gatewayUnavailableMessage: string
      reloadedTitle: string
      reloadedMessage: string
      reloadFailed: string
      savedTitle: string
      savedMessage: (name: string) => string
      newServer: string
      reload: string
      reloading: string
      emptyTitle: string
      emptyDesc: string
      disabled: string
      editServer: string
      name: string
      serverJson: string
      remove: string
      saveServer: string
    }
    model: {
      loading: string
      appliesDesc: string
      provider: string
      model: string
      applying: string
      defaultsLabel: string
      reasoning: string
      reasoningOff: string
      defaultsFailed: string
      auxiliaryTitle: string
      resetAllToMain: string
      auxiliaryDesc: string
      setToMain: string
      change: string
      autoUseMain: string
      providerDefault: string
      requestFailed: string
      activate: string
      activating: string
      setUpProvider: (name: string) => string
      pasteKeyPlaceholder: (env: string) => string
      needsApiKeyHint: (name: string) => string
      oauthHint: (name: string) => string
      staleAux: (count: number, names: string, provider: string) => string
      staleAuxOtherProviders: string
      moa: {
        title: string
        desc: string
        presetPlaceholder: string
        setDefault: string
        newPresetPlaceholder: string
        addPreset: string
        defaultLabel: string
        reference: (index: number) => string
        addReference: string
        aggregator: string
      }
      tasks: Record<string, AuxTaskCopy>
    }
    uninstall: {
      dangerZone: string
      checking: string
      title: string
      chooseDesc: string
      confirmTitle: string
      confirmBody: (consequence: string) => string
      appPath: (path: string) => string
      uninstalling: string
      confirmYes: string
      startFailed: string
      options: {
        gui: UninstallOptionCopy
        lite: UninstallOptionCopy
        full: UninstallOptionCopy
      }
    }
    providers: {
      connectAccount: string
      haveApiKey: string
      intro: string
      connected: string
      collapse: string
      connectAnother: string
      otherProviders: string
      disconnect: string
      disconnectInTerminal: string
      removeConfirm: (provider: string) => string
      removeExternalGeneric: (provider: string) => string
      removeKeyManaged: (provider: string) => string
      removeTerminalConfirm: (provider: string, command: string) => string
      removeTerminalRunning: (provider: string) => string
      removedTitle: string
      removedMessage: (provider: string) => string
      failedRemove: (provider: string) => string
      noProviderKeys: string
      searchKeys: string
      noKeysMatch: string
      loading: string
    }
    // hc-444: "Connect Feishu" card copy.
    feishu: {
      title: string
      intro: string
      connectedTitle: string
      connectedTo: (agent: string) => string
      connectedGeneric: string
      statusOk: string
      statusExpired: string
      statusInvalid: string
      statusStale: string
      sync: string
      resync: string
      syncing: string
      disconnect: string
      disconnectConfirm: string
      signInFirstTitle: string
      signInFirst: string
      noEntryTitle: string
      noEntry: string
      openBind: string
      afterBind: string
      syncedTitle: string
      syncedMessage: string
      disconnectedTitle: string
      disconnectedMessage: string
      syncFailed: string
      sessionExpired: string
      loading: string
    }
    sessions: {
      loading: string
      archivedTitle: string
      archivedIntro: string
      emptyArchivedTitle: string
      emptyArchivedDesc: string
      unarchive: string
      deletePermanently: string
      messages: (count: number) => string
      restored: string
      deleteConfirm: (title: string) => string
      defaultDirTitle: string
      defaultDirDesc: string
      defaultDirUpdated: string
      defaultsTo: (label: string) => string
      change: string
      choose: string
      clear: string
      notSet: string
      failedLoad: string
      unarchiveFailed: string
      deleteFailed: string
      updateDirFailed: string
      clearDirFailed: string
    }
    toolsets: {
      loadingConfig: string
      savedTitle: string
      savedMessage: (key: string) => string
      removedTitle: string
      removedMessage: (key: string) => string
      failedSave: (key: string) => string
      failedRemove: (key: string) => string
      failedReveal: (key: string) => string
      removeConfirm: (key: string) => string
      set: string
      notSet: string
      selectedTitle: string
      selectedMessage: (provider: string) => string
      failedSelect: (provider: string) => string
      failedLoad: string
      noProviderOptions: string
      noProviders: string
      ready: string
      nousIncluded: string
      noApiKeyRequired: string
      postSetupHint: (step: string) => string
      postSetupRun: string
      postSetupRunning: string
      postSetupStarting: string
      postSetupCompleteTitle: string
      postSetupCompleteMessage: (step: string) => string
      postSetupErrorTitle: string
      postSetupErrorMessage: (step: string) => string
      postSetupFailed: (step: string) => string
    }
  }

  skills: {
    tabSkills: string
    tabSkillsSubtitle: string
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
    paletteTitle: string
    back: string
    searchPlaceholder: string
    goTo: string
    goToSession: string
    commandCenter: string
    appearance: string
    settings: string
    changeTheme: string
    changeColorMode: string
    installTheme: {
      title: string
      placeholder: string
      loading: string
      error: string
      empty: string
      install: string
      installing: string
      installed: string
      installs: (count: string) => string
    }
    settingsFields: string
    mcpServers: string
    archivedChats: string
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
    restartGateway: string
    gatewayRestartFailed: string
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
    connectionError: string
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
    allProfiles: string
    showAllProfiles: string
    switchToProfile: (name: string) => string
    manageProfiles: string
    actionsFor: (name: string) => string
    color: string
    colorFor: (name: string) => string
    setColor: (color: string) => string
    autoColor: string
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
    soulOptional: string
    soulPlaceholder: (mode: string) => string
    soulPlaceholderCloned: string
    soulPlaceholderEmpty: string
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
    cloneFrom: string
    cloneFromNone: string
    cloneFromDesc: string
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

  // 个人资料 — the profile stats page (avatar header + usage stats off the
  // local analytics API). Distinct from `profiles` (the multi-profile manager).
  profileStats: {
    close: string
    signedOut: string
    loading: string
    failedLoad: string
    emptyTitle: string
    emptyDesc: string
    stats: {
      sessions: string
      tokens: string
      apiCalls: string
      activeDays: string
      skillsUsed: string
    }
    heatmap: {
      title: string
      daily: string
      weekly: string
      cumulative: string
      less: string
      more: string
      cellTitle: (date: string, tokens: string) => string
    }
    insights: {
      title: string
      busiestDay: string
      avgPerActiveDay: string
      topModel: string
      longestStreak: string
      streakDays: (days: number) => string
      estimatedCost: string
    }
    topSkills: {
      title: string
      uses: (count: string) => string
    }
  }

  cron: {
    close: string
    search: string
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
    newCron: string
    emptyDescNew: string
    emptyDescSearch: string
    emptyTitleNew: string
    emptyTitleSearch: string
    last: string
    next: string
    noRuns: string
    manage: string
    showRuns: string
    hideRuns: string
    runHistory: string
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

  // Goal-mode long-running tasks (one-shot cron jobs surfaced on /tasks).
  tasks: {
    newTask: string
    tabRunning: string
    tabDone: string
    emptyRunning: string
    emptyDone: string
    emptyDetail: string
    pending: string
    started: string
    runAgain: string
    goalLabel: string
    goalPlaceholder: string
    stuckHint: string
    stuckDetail: string
    waitingToStart: string
    progressLabel: string
    stepsOf: (completed: number, total: number) => string
    currentStepLabel: string
    latestOutputLabel: string
    runHistory: string
    noRuns: string
    phases: Record<'done' | 'failed' | 'running', string>
    newTaskTitle: string
    newTaskDesc: string
    goalRequired: string
    timeRequired: string
    whenLabel: string
    whenNow: string
    whenIn: string
    whenAt: string
    delayLabel: string
    atLabel: string
    persistNote: string
    startTask: string
    created: string
    startedNow: string
    failedStart: string
    deleted: string
    failedDelete: string
    deleting: string
    deleteTitle: string
    deleteDescPrefix: string
    deleteDescSuffix: string
    // Native OS notification copy fired by the task notifier (store/tasks.ts).
    notify: {
      doneTitle: string
      failedTitle: string
    }
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
    projects: string
    sessions: string
    cronJobs: string
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
    engineUpdate: {
      found: string
      updating: string
      failedRolledBack: string
    }
    // 壳(应用本体)更新胶囊:downloaded 后的「重启以更新 vX.Y.Z」。
    shellUpdate: {
      restartToUpdate: (version: string) => string
    }
    row: {
      pin: string
      unpin: string
      copyId: string
      export: string
      rename: string
      archive: string
      newWindow: string
      copyIdFailed: string
      actionsFor: (title: string) => string
      sessionActions: string
      sessionRunning: string
      needsInput: string
      waitingForAnswer: string
      handoffOrigin: (platform: string) => string
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

  home: {
    title: string
  }

  composer: {
    message: string
    approvalMode: {
      label: string
      review: { label: string; desc: string }
      full: { label: string; desc: string }
    }
    wakingProfile: (profile: string) => string
    placeholderStarting: string
    placeholderReconnecting: string
    placeholderFollowUp: string
    newSessionPlaceholders: readonly string[]
    followUpPlaceholders: readonly string[]
    startVoice: string
    queueMessage: string
    steer: string
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
    lookupLoading: string
    lookupNoMatches: string
    lookupTry: string
    lookupOr: string
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
    editingQueuedInComposer: string
    queueEdit: string
    queueSendNext: string
    queueSend: string
    queueDelete: string
    queueStuckTitle: string
    queueStuckBody: string
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
    dropFiles: string
    dropSession: string
  }

  statusStack: {
    agents: string
    background: (count: number) => string
    subagents: (count: number) => string
    todos: (done: number, total: number) => string
    running: string
    stop: string
    dismiss: string
    exit: (code: number) => string
  }

  updates: {
    stages: Record<string, string>
    checking: string
    checkFailedTitle: string
    tryAgain: string
    notAvailableTitle: string
    unsupportedMessage: string
    connectionRetry: string
    latestBody: string
    latestBodyBackend: string
    allSetTitle: string
    availableTitle: string
    availableBody: string
    availableTitleBackend: string
    availableBodyBackend: string
    availableBodyNoChangelog: string
    updateNow: string
    maybeLater: string
    moreChanges: (count: number) => string
    manualTitle: string
    manualBody: string
    manualPickedUp: string
    copy: string
    copied: string
    done: string
    applyingBody: string
    applyingBodyBackend: string
    applyingClose: string
    errorTitle: string
    errorBody: string
    notNow: string
    applyStatus: {
      preparing: string
      pulling: string
      restarting: string
      notAvailable: string
      failed: string
      noReturn: string
    }
  }

  install: {
    stageStates: Record<string, string>
    // Localized labels for the installer's known stage ids (Prerequisites,
    // Repository, Venv, …). Keyed by the raw stage name from the bootstrap
    // protocol; unknown ids fall back to formatStageName() in the overlay.
    stageLabels: Record<string, string>
    /** hc-452: rough per-step duration hint shown next to a PENDING stage row
     *  (first-install ballpark; an incremental update skips most of these).
     *  Same key space as stageLabels; an id with no entry renders no hint. */
    stageDurationHints: Record<string, string>
    oneTimeTitle: string
    unsupportedDesc: (platform: string) => string
    installCommand: string
    copyCommand: string
    viewDocs: string
    installTo: string
    retryAfterRun: string
    failedTitle: string
    settingUpTitle: string
    /** hc-452: shown instead of settingUpTitle when this run is an opt-in
     *  runtime version update rather than a first-ever install. `version` may
     *  be null before the target version resolves. */
    settingUpTitleUpdate: (version: string | null) => string
    finishingTitle: string
    failedDesc: string
    activeDesc: string
    /** hc-452: update-flow counterpart to activeDesc -- must NOT claim this is
     *  a one-time thing or that future launches skip this step (both false
     *  for a recurring runtime update). `version` may be null; see
     *  settingUpTitleUpdate. */
    activeDescUpdate: (version: string | null) => string
    progress: (completed: number, total: number) => string
    currentStage: (stage: string) => string
    fetchingManifest: string
    error: string
    hideOutput: string
    showOutput: string
    lines: (count: number) => string
    noOutput: string
    cancelling: string
    cancelInstall: string
    transcriptSaved: string
    copiedOutput: string
    copyOutput: string
    reloadRetry: string
  }

  onboarding: {
    headerTitle: string
    headerDesc: string
    /** ApexNodes managed-LLM (zero-key) first-run sign-in panel. */
    managed: {
      subtitle: string
      emailPlaceholder: string
      passwordPlaceholder: string
      signIn: string
      signingIn: string
      useOwnProvider: string
      /** Divider between the email/password form and the browser-login buttons. */
      dividerOr: string
      /** "用 Google 登录" browser (loopback) sign-in button. */
      signInGoogle: string
      /** "用 APEX 登录" browser (loopback) sign-in button. */
      signInApex: string
    }
    /** Success toast once a provider is connected and onboarding completes
     *  (store/onboarding notifyReady). `message` receives the connected
     *  provider's label; a locale may ignore it. */
    ready: {
      title: string
      message: (provider: string) => string
    }
    /** Clean prompt shown when a provider is seeded (DeepSeek) but its key is
     *  missing — replaces the raw "no usable credentials" runtime error. */
    addKeyToStart: string
    preparingInstall: string
    starting: string
    lookingUpProviders: string
    collapse: string
    /** "More — needs VPN" disclosure label hiding the international providers. */
    moreProvidersVpn: string
    otherProviders: string
    haveApiKey: string
    chooseLater: string
    recommended: string
    connected: string
    featuredPitch: string
    openRouterPitch: string
    apiKeyOptions: Record<string, { short: string; description: string }>
    backToSignIn: string
    getKey: string
    replaceCurrent: string
    pasteApiKey: string
    localApiKeyPlaceholder: string
    couldNotSave: string
    connecting: string
    update: string
    flowSubtitles: Record<string, string>
    startingSignIn: (provider: string) => string
    verifyingCode: (provider: string) => string
    connectedProvider: (provider: string) => string
    connectedPicking: (provider: string) => string
    signInFailed: string
    pickDifferentProvider: string
    signInWith: (provider: string) => string
    openedBrowser: (provider: string) => string
    authorizeThere: string
    copyAuthCode: string
    pasteAuthCode: string
    reopenAuthPage: string
    autoBrowser: (provider: string) => string
    reopenSignInPage: string
    waitingAuthorize: string
    externalPending: (provider: string) => string
    signedIn: string
    deviceCodeOpened: (provider: string) => string
    reopenVerification: string
    copy: string
    defaultModel: string
    freeTier: string
    pro: string
    free: string
    price: (input: string, output: string) => string
    change: string
    startChatting: string
    docs: (provider: string) => string
  }

  /** hc-511: managed relay-key recovery after a chat turn hit a relay auth
   *  error (HTTP 401/403). Either self-heal + retry, or a visible re-sign-in. */
  managedRecovery: {
    /** Shown when the relay key was self-healed. */
    healed: {
      title: string
      /** The active turn is being retried automatically. */
      retrying: string
      /** A background turn healed — the user should resend it. */
      resend: string
    }
    /** Shown when recovery is impossible without a re-login (no reusable token
     *  or an expired JWT) — a `*.local`/env seed key or an expired session. */
    signInRequired: {
      title: string
      message: string
      /** Reason banner surfaced on the managed sign-in panel. */
      reason: string
    }
  }

  /** Desktop auth boot-gate: the full-window login screen + bottom-left account
   *  panel (Codex-faithful, minimal). Chinese-first (China-first Desktop V0.2). */
  auth: {
    /** Login screen. */
    login: {
      /** Hero line under the logo ("开始使用"). */
      title: string
      /** Primary button — sign in with the Apex account. */
      signInApex: string
      /** Secondary button — quick sign-in with Google. */
      signInGoogle: string
      /** In-flight label while a browser sign-in is pending. */
      signingIn: string
      /** Generic sign-in failure line. */
      failed: string
      /** Account-abnormal (403 account_disabled) message shown on the gate. */
      accountDisabled: string
      /** Session-expired / login-lost (401) message shown on the gate. */
      sessionExpired: string
    }
    /** Bottom-left account panel + its popover menu. */
    account: {
      /** Fallback display name when no email/name is known (e.g. "账户"). */
      fallbackName: string
      /** Menu item — open profile. */
      profile: string
      /** Menu item — open settings. */
      settings: string
      /** Menu item — remaining usage / quota. */
      usage: string
      /** Menu item — sign out. */
      logout: string
    }
  }

  modelPicker: {
    title: string
    current: string
    unknown: string
    search: string
    noModels: string
    addProvider: string
    loadFailed: string
    noAuthenticatedProviders: string
    pro: string
    proNeedsSubscription: string
    free: string
    freeTier: string
    priceTitle: string
  }

  modelVisibility: {
    title: string
    search: string
    noAuthenticatedProviders: string
    addProvider: string
  }

  shell: {
    windowControls: string
    paneControls: string
    appControls: string
    connectingOverlay: string
    modelMenu: {
      search: string
      noModels: string
      editModels: string
      refreshModels: string
      loadFailed: string
      moaPresets: string
      moaPresetItem: (preset: string) => string
      fast: string
      medium: string
    }
    modelOptions: {
      noOptions: string
      options: string
      thinking: string
      fast: string
      effort: string
      minimal: string
      low: string
      medium: string
      high: string
      max: string
      updateFailed: string
      fastFailed: string
    }
    gatewayMenu: {
      gateway: string
      connected: string
      connecting: string
      offline: string
      inferenceReady: string
      inferenceNotReady: string
      checkingInference: string
      disconnected: string
      openSystem: string
      connection: (label: string) => string
      recentActivity: string
      viewAllLogs: string
      messagingPlatforms: string
    }
    statusbar: {
      unknown: string
      restart: string
      update: string
      updateInProgress: string
      commitsBehind: (count: number, branch: string) => string
      backendVersion: (version: string) => string
      backendLabel: (version: string) => string
      closeCommandCenter: string
      openCommandCenter: string
      gateway: string
      gatewayReady: string
      gatewayNeedsSetup: string
      gatewayChecking: string
      gatewayConnecting: string
      gatewayOffline: string
      gatewayRestarting: string
      gatewayTitle: string
      agents: string
      closeAgents: string
      openAgents: string
      subagents: (count: number) => string
      failed: (count: number) => string
      running: (count: number) => string
      cron: string
      openCron: string
      turnRunning: string
      currentTurnElapsed: string
      contextUsage: string
      session: string
      runtimeSessionElapsed: string
      modelNone: string
      noModel: string
      switchModel: string
      openModelPicker: string
      modelTitle: (provider: string, model: string) => string
      providerModelTitle: (provider: string, model: string) => string
    }
  }

  rightSidebar: {
    aria: string
    panelsAria: string
    files: string
    terminal: string
    noFolderSelected: string
    changeCwdTitle: string
    remotePickerTitle: string
    remotePickerDescription: string
    remotePickerSelect: string
    folderTip: (cwd: string) => string
    openFolder: string
    refreshTree: string
    collapseAll: string
    previewUnavailable: string
    couldNotPreview: (path: string) => string
    noProjectTitle: string
    noProjectBody: string
    unreadableTitle: string
    unreadableBody: (error: string) => string
    emptyTitle: string
    emptyBody: string
    treeErrorTitle: string
    treeErrorBody: string
    tryAgain: string
    loadingTree: string
    loadingFiles: string
    terminalHide: string
    addToChat: string
  }

  preview: {
    tab: string
    closeTab: (label: string) => string
    closePane: string
    loading: string
    unavailable: string
    opening: string
    hide: string
    openPreview: string
    sourceLineTitle: string
    source: string
    renderedPreview: string
    unknownSize: string
    binaryTitle: string
    binaryBody: (label: string) => string
    largeTitle: string
    largeBody: (label: string, size: string) => string
    previewAnyway: string
    truncated: string
    noInlineTitle: string
    noInlineBody: (mimeType: string) => string
    console: {
      deselect: string
      select: string
      copyFailed: string
      copyEntry: string
      sendEntry: string
      messages: (count: number) => string
      resize: string
      title: string
      selected: (count: number) => string
      sendToChat: string
      copySelected: string
      copyAll: string
      copy: string
      clear: string
      empty: string
      promptHeader: string
      sentTitle: string
      sentMessage: (count: number) => string
    }
    web: {
      appFailedToBoot: string
      serverNotFound: string
      failedToLoad: string
      tryAgain: string
      restarting: string
      askRestart: string
      lookingRestart: (taskId: string) => string
      restartingTitle: string
      restartingMessage: string
      startRestartFailed: string
      restartFailed: string
      hideConsole: string
      showConsole: string
      hideDevTools: string
      openDevTools: string
      finishedRestarting: (message?: string) => string
      failedRestarting: (message: string) => string
      unknownError: string
      restartedTitle: string
      reloadingNow: string
      restartFailedTitle: string
      restartFailedMessage: string
      stillWorking: string
      workspaceReloading: string
      fileChanged: (url: string) => string
      filesChanged: (count: number, url: string) => string
      watchFailed: string
      moduleMimeDescription: string
      loadFailedConsole: (code: number | undefined, message: string) => string
      unreachableDescription: string
      openTarget: (url: string) => string
      fallbackTitle: string
    }
  }

  assistant: {
    thread: {
      loadingSession: string
      showEarlier: string
      loadingResponse: string
      thinking: string
      today: (time: string) => string
      yesterday: (time: string) => string
      copy: string
      refresh: string
      moreActions: string
      branchNewChat: string
      dismissError: string
      readAloudFailed: string
      preparingAudio: string
      stopReading: string
      readAloud: string
      editMessage: string
      scrollToBottom: string
      stop: string
      restorePrevious: string
      restoreCheckpoint: string
      restoreFromHere: string
      restoreTitle: string
      restoreBody: string
      restoreConfirm: string
      restoreNext: string
      goForward: string
      sendEdited: string
      attachingFile: string
      compacting: string
      steered: string
      processOutput: string
    }
    approval: {
      gatewayDisconnected: string
      sendFailed: string
      run: string
      command: string
      moreOptions: string
      allowSession: string
      alwaysAllowMenu: string
      jumpToApproval: string
      reject: string
      alwaysTitle: string
      alwaysDescription: (pattern: string) => string
      alwaysAllow: string
    }
    clarify: {
      notReady: string
      gatewayDisconnected: string
      sendFailed: string
      loadingQuestion: string
      other: string
      placeholder: string
      shortcutSuffix: string
      back: string
      skip: string
      send: string
    }
    tool: {
      code: string
      copyCode: string
      renderingImage: string
      copyOutput: string
      copyCommand: string
      copyContent: string
      copyUrl: string
      copyResults: string
      copyQuery: string
      copyFile: string
      copyPath: string
      outputAlt: string
      rawResponse: string
      copyActivity: string
      recoveredOne: string
      recoveredMany: (count: number) => string
      failedOne: string
      failedMany: (count: number) => string
      statusRunning: string
      statusError: string
      statusRecovered: string
      statusDone: string
      errorDetails: string
      searchResults: string
      stdoutLabel: string
      stderrLabel: string
      detailLabels: {
        details: string
        snapshotSummary: string
        commandOutput: string
      }
      titles: {
        browser_click: ToolTitleCopy
        browser_fill: ToolTitleCopy
        browser_navigate: ToolTitleCopy
        browser_snapshot: ToolTitleCopy
        browser_take_screenshot: ToolTitleCopy
        browser_type: ToolTitleCopy
        clarify: ToolTitleCopy
        cronjob: ToolTitleCopy
        edit_file: ToolTitleCopy
        execute_code: ToolTitleCopy
        image_generate: ToolTitleCopy
        list_files: ToolTitleCopy
        patch: ToolTitleCopy
        read_file: ToolTitleCopy
        search_files: ToolTitleCopy
        session_search_recall: ToolTitleCopy
        terminal: ToolTitleCopy
        todo: ToolTitleCopy
        vision_analyze: ToolTitleCopy
        web_extract: ToolTitleCopy
        web_search: ToolTitleCopy
        write_file: ToolTitleCopy
        unknown: ToolTitleCopy
      }
      dynamicTitles: {
        readingHost: (host: string) => string
        readHost: (host: string) => string
        openingHost: (host: string) => string
        openedHost: (host: string) => string
        searchingQuery: (query: string) => string
        searchedQuery: (query: string) => string
        runningCommand: (command: string) => string
        ranCommand: (command: string) => string
        runningCode: (command: string) => string
        ranCode: (command: string) => string
      }
    }
  }

  prompts: {
    gatewayDisconnected: string
    sudoSendFailed: string
    secretSendFailed: string
    sudoTitle: string
    sudoDesc: string
    sudoPlaceholder: string
    secretTitle: string
    secretDesc: string
    secretPlaceholder: string
  }

  desktop: {
    audioReadFailed: string
    sessionUnavailable: string
    createSessionFailed: string
    promptFailed: string
    providerCredentialRequired: string
    emptySlashCommand: string
    desktopCommands: string
    skillCommandsAvailable: (count: number) => string
    warningLine: (message: string) => string
    yoloArmed: string
    yoloOff: string
    yoloSystem: (active: boolean) => string
    yoloTitle: string
    yoloToggleFailed: string
    profileStatus: (current: string) => string
    unknownProfile: string
    noProfileNamed: (target: string, available: string) => string
    newChatsProfile: (name: string) => string
    setProfileFailed: string
    sttDisabled: string
    stopFailed: string
    regenerateFailed: string
    editFailed: string
    resumeFailed: string
    resumeStrandedTitle: string
    resumeStrandedBody: string
    resumeRetry: string
    nothingToBranch: string
    branchNeedsChat: string
    sessionBusy: string
    branchStopCurrent: string
    branchNoText: string
    branchTitle: string
    branchFailed: string
    deleteFailed: string
    archived: string
    archiveFailed: string
    cwdChangeFailed: string
    cwdStagedTitle: string
    cwdStagedMessage: string
    modelSwitchFailed: string
    modelSwitchBusy: string
    modelSwitchRetry: string
    sessionExported: string
    sessionExportFailed: string
    imageSaved: string
    downloadStarted: string
    restartToUseSaveImage: string
    restartToSaveImages: string
    imageDownloadFailed: string
    openImage: string
    downloadImage: string
    generatedImageAlt: string
    savingImage: string
    imagePreviewFailed: string
    imageAttach: string
    imageWriteFailed: string
    imageAttachFailed: string
    attachImages: string
    clipboard: string
    noClipboardImage: string
    clipboardPasteFailed: string
    dropFiles: string
    handoff: {
      pickPlatform: string
      success: (platform: string) => string
      systemNote: (platform: string) => string
      failed: (error: string) => string
      timedOut: string
    }
  }

  errors: {
    genericFailure: string
    boundaryTitle: string
    boundaryDesc: string
    reloadWindow: string
    openLogs: string
  }

  ui: {
    search: {
      clear: string
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
      toggle: string
    }
  }
  operationStatus: {
    browserActive: string
    computerActive: string
    computerWarning: string
    running: string
    stop: string
  }
}
