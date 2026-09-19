export interface BootTranslations {
  boot: {
    ready: string
    desktopBootFailedWithMessage: (message: string) => string
    steps: {
      connectingGateway: string
      loadingSettings: string
      loadingSessions: string
      retryingRemoteBackend: string
      startingDesktopConnection: string
      startingHermesDesktop: string
    }
    errors: {
      backgroundExited: string
      backgroundExitedDuringStartup: string
      backendStopped: string
      restartHermes: string
      openLogs: string
      desktopBootFailed: string
      gatewayConnectionLost: string
      gatewayConnectionLostDetail: string
      reconnectNow: string
      connectionSettings: string
      gatewaySignInRequired: string
      gatewaySignInRequiredDetail: string
      signInAgain: string
      ipcBridgeUnavailable: string
    }
    causes: {
      exitedEarly: string
      timedOut: string
      permission: string
      diskFull: string
      portInUse: string
      installMissing: string
    }
    failure: {
      title: string
      description: string
      details: string
      remoteTitle: string
      remoteDescription: string
      retry: string
      repairInstall: string
      useLocalGateway: string
      gatewaySettings: string
      back: string
      openLogs: string
      repairHint: string
      remoteSignInHint: (signInLabel: string) => string
      signOutAndSignIn: string
      remoteFailureHint: string
      cloudDownTitle: string
      cloudDownDescription: string
      cloudDownHint: string
      cloudDownCheckPortal: string
      cloudDownDiscord: string
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

  remoteDisplayBanner: {
    message: (reason: string) => string
  }

  updates: {
    stages: Record<string, string>
    checking: string
    checkFailedTitle: string
    tryAgain: string
    notAvailableTitle: string
    unsupportedMessage: string
    connectionRetry: string
    gitUnusable: string
    connectionSettings: string
    openDownloadPage: string
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
    /** GUI/backend skew (#45205): backend updated but the running desktop app
     *  package (AppImage/.deb/.rpm) was not changed and must be reinstalled. */
    guiSkewTitle: string
    guiSkewBody: string
    copy: string
    copied: string
    done: string
    applyingBody: string
    applyingBodyBackend: string
    applyingClose: string
    errorTitle: string
    errorBody: string
    blockerTitle: string
    blockerBody: string
    foreignBlockerTitle: string
    foreignBlockerBody: string
    mixedBlockerBody: string
    closePreviewsAndUpdate: string
    closePreviewsAndCheckAgain: string
    localPreview: string
    portLabel: (port: number) => string
    pidLabel: (pid: number) => string
    technicalDetails: string
    notNow: string
    /** Multi-target update flow: client nudge after a backend update, and
     *  per-row fan-out outcomes when updating every registered instance. */
    clientAlsoBehindTitle: string
    clientAlsoBehindMessage: string
    clientAlsoBehindAction: string
    everythingDispatched: string
    everythingSkipped: string
    everythingRowFailed: string
    everythingFanoutFailedTitle: string
    applyStatus: {
      preparing: string
      pulling: string
      restarting: string
      notAvailable: string
      failed: string
      noReturn: string
    }
  }

  /** The guided first run's pre-written opening line — banked, not generated,
   *  so the first paint costs no model time. Translated per locale because the
   *  model is told to speak the user's language from its first real turn, and
   *  an English opener above a Japanese reply reads as two different agents.
   *  `nameSuggestion` offers the OS account name as a default. */
  handoffTour: {
    profileTitle: string
    profileText: string
    sessionsTitle: string
    sessionsText: string
    stayTitle: string
    stayText: string
  }
  guidedGreeting: {
    line: string
    nameSuggestion: (name: string) => string
  }
  install: {
    stageStates: Record<string, string>
    oneTimeTitle: string
    unsupportedDesc: (platform: string) => string
    installCommand: string
    copyCommand: string
    viewDocs: string
    installTo: string
    retryAfterRun: string
    setupChoiceTitle: string
    setupChoiceDesc: string
    connectExistingTitle: string
    connectExistingShort: string
    connectExistingDesc: string
    installLocalTitle: string
    installLocalDesc: string
    localStartUnavailable: string
    remoteSetupTitle: string
    remoteSetupDesc: string
    remoteUrlTitle: string
    remoteUrlDesc: string
    remoteUrlPlaceholder: string
    probing: string
    probeError: string
    probeErrorDetails: string
    identityProvider: string
    authTitle: string
    authNeedsOauth: (provider: string) => string
    authSignedIn: string
    connected: string
    signIn: string
    signInWith: (provider: string) => string
    enterUrlFirst: string
    signInIncomplete: string
    tokenTitle: string
    tokenDesc: string
    pasteSessionToken: string
    incompleteSignInTest: string
    incompleteTokenTest: string
    testConnection: string
    testSucceeded: (baseUrl: string, version?: string) => string
    applyRemote: string
    backToSetup: string
    failedTitle: string
    settingUpTitle: string
    finishingTitle: string
    failedDesc: string
    activeDesc: string
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
    openLogs: string
  }

  onboarding: {
    headerTitle: string
    headerDesc: string
    preparingInstall: string
    starting: string
    lookingUpProviders: string
    collapse: string
    otherProviders: string
    haveApiKey: string
    chooseLater: string
    recommended: string
    connected: string
    featuredPitch: string
    fireworksPitch: string
    localModelsTitle: string
    localModelsPitch: string
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
    signInExpired: string
    signInDidNotFinish: (provider: string) => string
    tryAgain: string
    useApiKeyInstead: string
    errorDetails: string
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

  freeTier: {
    /** Settings › Providers row title while the Nous identity is the free tier. */
    providerRowTitle: string
    /** The featured row's pitch while the identity is the free tier: what signing in adds. */
    providerRowPitch: string
    // First-launch introduction (ready screen + composer strip).
    readyTitle: string
    readyCaption: string
    begin: string
    signInInstead: string
    otherProviders: string
    stripTitle: string
    stripBody: string
    openModelPicker: string
    dismiss: string
    // Statusbar chip.
    /** The status-bar chip's label: the provider name alone; the model id and the sign-in follow it. */
    providerName: string
    statusLabel: (model: string) => string
    // Sign-in dialog.
    signIn: string
    signInHeading: string
    settingUp: string
    codeBody: string
    copyLink: string
    doNotShare: string
    waiting: string
    finishingHeading: string
    finishingBody: string
    signedInAs: (email: string) => string
    signedIn: string
    completedBody: string
    defaultModel: string
    change: string
    done: string
    notNow: string
    tryAgain: string
    startAgain: string
    didNotComplete: string
    rejectedBody: string
    supersededBody: string
    timedOutHeading: string
    timedOutBody: string
    retiredBody: string
    errorBody: string
    /** The account service asked for a short wait mid sign-in (a busy account, a rate limit, the ops pause). */
    busyHeading: string
    busyBody: (wait: string) => string
    /** The account service could not be reached or errored mid sign-in. */
    unreachableBody: string
    alreadySignedInHeading: string
    alreadySignedInBody: string
    // First-launch set-up failure notice: the free tier could not be created at boot.
    // One sentence per backend code (`hermes_cli/anon_auth.py::ANON_*`); the copy never says
    // the free MODEL is off — what is unavailable is using Hermes without signing in.
    setupFailed: {
      gateClosed: string
      paused: string
      rateLimited: (wait: string) => string
      unreachable: string
      serverError: string
      powRequired: string
      locked: string
      generic: string
      /** The sign-in door, when the account service is reachable: the Nous row sits right below. */
      signInBelow: string
      tryAgain: string
      retrying: string
    }
  }
}
