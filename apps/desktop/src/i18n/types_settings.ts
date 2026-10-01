interface ModeOptionCopy {
  label: string
  description: string
}

interface AuxTaskCopy {
  label: string
  hint: string
}

export interface SettingsTranslations {
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
    subpages: Record<string, string>
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
      providerCustomEndpoints: string
      providerLocalModels: string
      gateway: string
      apiKeys: string
      keybinds: string
      keysTools: string
      keysSettings: string
      mcp: string
      archivedChats: string
      sessions: string
      about: string
      billing: string
      notifications: string
      vault: string
    }
    plugins: {
      title: string
      blurb: string
      count: (n: number) => string
      openFolder: string
      rescan: string
      reveal: string
      enable: string
      disable: string
      failed: string
      empty: string
      kinds: { bundled: string; disk: string; runtime: string }
      agentHalfMissing: string
      agentHalfMissingTip: string
      installModal: {
        installFromGit: string
        reviewRepository: string
        repoPlaceholder: string
        title: string
        description: string
        repoLabel: string
        includesHeading: string
        agentLabel: string
        desktopLabel: string
        profileLabel: string
        agentTargetLocal: (profile: string, dir: string) => string
        agentTargetRemote: (profile: string) => string
        catalogPinned: (name: string, sha: string) => string
        reviewedHeading: string
        reviewedIntro: string
        toolsConnected: (n: number) => string
        skillsReady: (names: string[]) => string
        nextChat: string
        serverNotConnected: (server: string, reason: string) => string
        missingEnvAction: string
        alreadyInstalled: (name: string) => string
        desktopTarget: string
        desktopTargetFromPackage: string
        desktopOnlyNote: string
        insecureWarning: string
        securityHeading: string
        securityIntro: string
        sourceHeading: string
        viewRepository: string
        viewPluginFiles: string
        gitCloneLabel: string
        enableAgent: string
        forceReinstall: string
        pinToCommit: string
        pinToCommitPlaceholder: string
        pinToCommitHint: string
        pinToCommitInvalid: string
        install: string
        installing: string
        probing: string
        probeUnavailable: string
        desktopUnavailable: string
        selectComponent: string
        agentSuccess: (name: string) => string
        desktopSuccess: (name: string) => string
        agentFailed: string
        installUncertain: string
        desktopFailed: string
        missingEnv: (name: string, vars: string) => string
      }
    }
    vault: {
      title: string
      blurb: string
      count: (n: number) => string
      loadFailed: string
      empty: string
      emptyDesc: string
      add: string
      addTitle: string
      addDescription: string
      added: string
      adding: string
      addConfirm: string
      kindField: string
      kinds: Record<'address' | 'login' | 'payment', string>
      labelField: string
      labelPlaceholder: string
      labelRequired: string
      originField: string
      originPlaceholder: string
      originPlaceholderCheckout: string
      originInvalid: string
      identifierTypeField: string
      identifierTypes: Record<'email' | 'phone' | 'username', string>
      identifierField: string
      identifierShown: (identifier: string) => string
      passwordField: string
      loginFieldsRequired: string
      cardNumberField: string
      cardNameField: string
      expMonthField: string
      expYearField: string
      cvcField: string
      postalField: string
      addressLine1Field: string
      addressLine2Field: string
      cityField: string
      stateField: string
      countryField: string
      optional: string
      createdOn: (date: string) => string
      deleteAction: string
      otpField: string
      otpPlaceholder: string
      otpHint: string
      twoFactorBadge: string
      deleteTitle: string
      deleteDescription: (label: string) => string
      deleteConfirm: string
      sources: {
        title: string
        blurb: string
        toggleFailed: string
        notInstalled: (name: string) => string
        disabledDesc: string
        lockedDesc: string
        unlockedDesc: string
        statusLocked: string
        statusNotDetected: string
        statusOff: string
        statusUnlocked: string
        unlock: string
        unlocking: string
        lock: string
        unlocked: (name: string) => string
        unlockTitle: (name: string) => string
        unlockDescription: string
        masterPasswordPlaceholder: string
      }
    }
    notifications: {
      title: string
      intro: string
      enableAll: string
      enableAllDesc: string
      focusedHint: string
      kinds: Record<
        'approval' | 'backgroundDone' | 'credits' | 'input' | 'plugin' | 'turnDone' | 'turnError',
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
      hideCodeDiffsTitle: string
      hideCodeDiffsDesc: string
      hideThreadTimelineTitle: string
      hideThreadTimelineDesc: string
      reasoningCollapsedTitle: string
      reasoningCollapsedDesc: string
      chatTextScaleTitle: string
      chatTextScaleDesc: string
      uiScaleTitle: string
      uiScaleDesc: (percent: number) => string
      sessionDensityTitle: string
      sessionDensityDesc: string
      sessionDensityCompact: string
      sessionDensityComfortable: string
      sessionDensityDetailed: string
      tabStripTitle: string
      tabStripDesc: string
      tabStripAuto: string
      tabStripAlways: string
      tabStripNever: string
      appActionsTitle: string
      appActionsDesc: string
      appActionsLeft: string
      appActionsRight: string
      terminalFontTitle: string
      terminalFontDesc: string
      terminalFontPlaceholder: string
      terminalFontPreview: string
      terminalFontReset: string
      chatFontTitle: string
      chatFontDesc: string
      chatFontPlaceholder: string
      chatFontPreview: string
      chatFontSample: string
      chatFontReset: string
      translucencyTitle: string
      translucencyDesc: string
      translucencyGlassDesc: string
      translucencyModeClear: string
      translucencyModeGlass: string
      translucencyTintTitle: string
      translucencyFadeTitle: string
      translucencyFrostTitle: string
      translucencyFrost: {
        'under-window': string
        popover: string
        titlebar: string
        header: string
      }
      translucencyScopeTitle: string
      translucencyScope: {
        window: string
        sidebar: string
      }
      backdropTitle: string
      backdropDesc: string
      userBubbleTitle: string
      userBubbleDesc: string
      textDirectionTitle: string
      textDirectionDesc: string
      textDirection: { auto: string; rtl: string; ltr: string }
      introSplashTitle: string
      introSplashDesc: string
      modelPricingTitle: string
      modelPricingDesc: string
      reactionsTitle: string
      reactionsDesc: string
      tipsTitle: string
      tipsDesc: string
      tipsReset: (count: number) => string
      toursTitle: string
      toursDesc: string
      composerPopoutTitle: string
      composerPopoutDesc: string
      fileBrowserTitle: string
      fileBrowserDesc: string
      vibeHeartsTitle: string
      vibeHeartsDesc: string
      embedsTitle: string
      embedsDesc: string
      embedsAsk: string
      embedsAlways: string
      embedsOff: string
      embedsReset: (count: number) => string
      resumeLastSessionTitle: string
      resumeLastSessionDesc: string
      product: string
      productDesc: string
      technical: string
      technicalDesc: string
      themeTitle: string
      themeDesc: string
      themeSearchPlaceholder: string
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
      pet: {
        title: string
        intro: string
        restartHint: string
        scaleTitle: string
        scaleDesc: string
        roamTitle: string
        roamDesc: string
        chooseTitle: string
        chooseDesc: string
        searchPlaceholder: string
        unreachable: string
        noMatch: (query: string) => string
        installedTag: string
        generatedTag: string
        countCapped: (cap: number, total: number) => string
        count: (n: number) => string
        uninstall: (name: string) => string
        delete: (name: string) => string
        deleteTitle: (name: string) => string
        deleteBody: string
        deleteConfirm: string
        rename: (name: string) => string
        renameTitle: string
        renamePlaceholder: string
        renameSave: string
        exportPet: (name: string) => string
        adoptFailed: (slug: string) => string
        uninstallFailed: (slug: string) => string
        renameFailed: (slug: string) => string
        exportFailed: (slug: string) => string
        noneAvailable: string
        turnOnFailed: string
        turnOffFailed: string
      }
    }
    fieldLabels: Record<string, string>
    fieldDescriptions: Record<string, string>
    uninstallSection: {
      dangerZone: string
      checkingInstalled: string
      uninstallHermes: string
      chooseHowMuch: string
      confirmUninstall: string
      confirmBody: (what: string) => string
      appLabel: string
      couldNotStart: string
      uninstalling: string
      yesUninstall: string
      options: {
        gui: { title: string; description: string; consequence: string }
        lite: { title: string; description: string; consequence: string }
        full: { title: string; description: string; consequence: string }
      }
    }
    poolLimits: {
      warmBotBackendsAria: string
      warmBotBackendsTitle: string
      backendIdleTimeoutAria: string
      backendIdleTimeoutTitle: string
    }
    customEndpoints: {
      active: string
      apiKeySet: string
      use: string
      editTitle: string
      addTitle: string
      fields: {
        name: string
        providerId: string
        endpointUrl: string
        defaultModel: string
        context: string
        apiKey: string
        apiKeyNewPlaceholder: string
        apiKeyPlaceholder: string
        useNewChats: string
        discoverModels: string
      }
      test: string
      save: string
      newEndpoint: string
      apiMode: string
      autoDetect: string
      couldNotLoad: string
      endpointSaved: string
      saveFailed: string
      endpointReachable: string
      endpointReachableTransport: (transport: string) => string
      endpointReachableModels: (reachable: string, count: number) => string
      endpointValidationFailed: string
      validationFailed: string
      activationFailed: string
      deleteConfirm: (name: string) => string
      deleteFailed: string
      title: string
      deleteEndpoint: string
      emptyDescription: string
      emptyTitle: string
      namePlaceholder: string
      contextPlaceholder: string
    }
    computerUse: {
      accessibility: string
      screenRecording: string
      driverHealth: string
    }
    about: {
      updates: string
    }
    config: {
      minimizeToTrayTitle: string
      minimizeToTrayDesc: string
      minimizeToTrayUnavailable: string
      none: string
      noneParen: string
      builtinOnly: string
      notSet: string
      commaSeparated: string
      searchPlaceholder: string
      noResults: string
      systemDefault: string
      loading: string
      emptyTitle: string
      emptyDesc: string
      failedLoad: string
      autosaveFailed: string
      imported: string
      invalidJson: string
      toolsetsWipeConfirm: string
      keepAwakeTitle: string
      keepAwakeDesc: string
      disableF12Title: string
      disableF12Desc: string
      alwaysExternalLinksTitle: string
      alwaysExternalLinksDesc: string
      attachmentSizeTitle: string
      attachmentSizeDesc: string
      attachmentSizeUnit: string
      attachmentSizeLabel: string
      voiceShortcutHintTitle: string
      voiceShortcutHintDesc: string
      showOptions: string
    }
    hudModifier: {
      title: string
      description: string
      permission: string
      unavailable: string
      missingHelper: string
      unsupportedSession: string
    }
    screenshot: {
      enabledTitle: string
      enabledDesc: string
      statusTitle: string
      checking: string
      disabled: string
      starting: string
      ready: string
      inputPermission: string
      screenPermission: string
      openSettings: string
      retry: string
      unavailable: string
      errorTitle: string
      loadFailed: string
      saveFailed: string
      permissionFailed: string
      captureFailed: string
      contextChanged: string
    }
    quickEntry: {
      enabledTitle: string
      enabledDesc: string
      shortcutTitle: string
      shortcutDesc: string
      active: string
      takenBy: string
      invalidShortcut: string
    }
    credentials: {
      pasteKey: string
      pasteLabelKey: (label: string) => string
      optional: string
      enterValueFirst: string
      couldNotSave: string
      remove: string
      getKey: string
      saving: string
    }
    envActions: {
      actions: string
      manageInKeys: string
      docs: string
      hideValue: string
      revealValue: string
      replace: string
      set: string
      clear: string
    }
    // v2 multi-connection registry: Settings → Connections.
    connections: {
      title: string
      intro: string
      stagedNote: string
      launchModeTitle: string
      launchModeDesc: string
      searchPlaceholder: string
      noSearchResults: string
      loadFailed: string
      currentPill: string
      primaryPill: string
      managedPill: string
      addConnection: string
      editConnection: string
      removeConnection: string
      removeConfirmTitle: string
      removeConfirmDesc: (label: string) => string
      makePrimary: string
      testConnection: string
      testOk: string
      testFailed: string
      saveFailed: string
      removeFailed: string
      updateAll: string
      updateAllRunning: string
      updateAllDone: string
      updateAllFailed: string
      updateSkippedCloud: string
      kindLocal: string
      kindRemote: string
      kindCloud: string
      kindSsh: string
      kindLocalDesc: string
      kindRemoteDesc: string
      kindCloudDesc: string
      kindSshDesc: string
      labelTitle: string
      labelDesc: string
      labelPlaceholder: string
      urlTitle: string
      sshHostTitle: string
      headersTitle: string
      headersDesc: string
      headerValuePlaceholder: string
      headerValueSaved: string
      headerAdd: string
      headerRemove: string
      duplicateLocal: string
      duplicateUrl: (label: string) => string
      duplicateSsh: (label: string) => string
      sameBackendHint: (label: string) => string
      localAddHint: string
      cloudAddHint: string
      save: string
      saving: string
      cancel: string
      empty: string
    }
    managedUpdates: {
      title: string
      intro: string
      sshConnection: string
      update: string
      updating: string
      progress: string
      updated: string
      partial: string
      refused: string
      failed: string
      alreadyRunning: string
      receipt: (id: string, outcome: string) => string
      receiptVersions: (pre: string, post: string) => string
      scopesRestored: (profiles: string) => string
      scopeNotRestored: (profile: string, error: string) => string
    }
    gateway: {
      loading: string
      unavailableTitle: string
      unavailableDesc: string
      title: string
      envOverride: string
      intro: string
      envOverrideTitle: string
      envOverrideDesc: string
      modeTitle: string
      localTitle: string
      localDesc: string
      remoteTitle: string
      remoteDesc: string
      remoteAuthHint: string
      cloudTitle: string
      cloudDesc: string
      cloudSignInTitle: string
      cloudSignIn: string
      cloudSignedIn: string
      cloudNeedsSignIn: string
      cloudSignedInDesc: string
      cloudAgentsTitle: string
      cloudOrgPickerTitle: string
      cloudOrgSelect: string
      cloudOrgChange: string
      cloudOrgRole: (role: string) => string
      cloudLoadingAgents: string
      cloudNoAgents: { before: string; linkText: string; after: string }
      cloudRefresh: string
      cloudConnect: string
      cloudSavedTitle: string
      cloudSavedDesc: string
      cloudUseSaved: string
      cloudActive: string
      cloudConnecting: string
      cloudDiscoverFailed: string
      cloudConnectFailed: string
      cloudSignInFailed: string
      cloudSignedOutTitle: string
      cloudSignedOutMessage: string
      cloudConnectedTitle: string
      cloudConnectedPill: string
      cloudConnectedTo: (name: string) => string
      cloudAgentProvisioning: string
      cloudStatusLabel: (status: string) => string
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
      plainTextConfirmTitle: string
      plainTextConfirmDesc: string
      plainTextConfirmAction: string
      plainTextStoredTitle: string
      plainTextStoredDesc: string
      keychainEncryptionTitle: string
      keychainEncryptionDesc: string
      keychainEncryptionFailed: string
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
      sshTitle: string
      sshDesc: string
      sshTrustHint: string
      sshHostTitle: string
      sshHostDesc: string
      sshHostPick: string
      sshHostPickTitle: string
      sshHostPickDesc: string
      sshHostCustom: string
      sshUserTitle: string
      sshUserDesc: string
      sshUserPlaceholder: string
      sshPortTitle: string
      sshPortDesc: string
      sshKeyTitle: string
      sshKeyDesc: string
      sshHermesPathTitle: string
      sshHermesPathDesc: string
      sshHermesPathPlaceholder: string
      sshTestConnection: string
      sshConnect: string
      sshButtonsHint: string
      sshReachable: (host: string, platform: string) => string
      sshIncompleteHost: string
      sshErrUnreachable: string
      sshErrAuth: string
      sshErrHostKey: string
      sshErrNotInstalled: string
      sshErrPlatform: string
      sshErrTimeout: string
      sshErrUpdateRequired: string
      sshErrUnknown: string
    }
    keys: {
      loading: string
      failedLoad: string
      empty: string
    }
    search: {
      placeholder: string
      pill: string
    }
    profileScope: {
      appliesTo: string
      editsProfile: (profile: string) => string
    }
    mcp: {
      loading: string
      invalidJson: string
      saveFailed: string
      removeFailed: string
      reloadFailed: string
      savedTitle: string
      savedMessage: (name: string) => string
      disabled: string
      name: string
      serverJson: string
      remove: string
      test: string
      catalogLoading: string
      catalogInstallFailed: (name: string) => string
      catalogEnvRequired: string
      capabilitySummary: (tools: number, prompts: number, resources: number) => string
      costTokens: (tokens: string) => string
      usage30d: (uses: string) => string
      statusConnecting: string
      statusNeedsAuth: string
      statusError: string
      statusOff: string
      allServers: string
      authenticatedTitle: string
      authenticatedMessage: (server: string, count: number) => string
      authenticate: string
      noOutput: string
      deepLinkTitle: string
      deepLinkDescription: string
      deepLinkStdioWarning: string
      deepLinkConfirm: string
      deepLinkNameInvalid: string
      deepLinkNameConflict: (name: string) => string
      deepLinkErrorTitle: string
      deepLinkErrorName: string
      deepLinkErrorConfig: string
      deepLinkErrorShape: string
      deepLinkErrorUrl: string
      deepLinkErrorTooLarge: string
    }
    model: {
      setupProviderFallback: string
      setUpProvider: (name: string) => string
      staleAuxBefore: (count: number, names: string) => string
      staleAuxAfter: string
      staleAuxOtherProviders: string
      moaEnabled: string
      moaSetDefault: string
      moaNewPresetPlaceholder: string
      moaAddPreset: string
      customModel: string
      customModelPlaceholder: string
      chooseFromList: string
      moaDefault: string
      moaReferenceToggle: (enabled: boolean, index: number) => string
      moaReferenceTitle: (index: number) => string
      moaAddReference: string
      loading: string
      appliesDesc: string
      provider: string
      model: string
      applying: string
      mainAppliedTitle: string
      mainAppliedMessage: (model: string) => string
      defaultsLabel: string
      reasoning: string
      reasoningOff: string
      defaultsFailed: string
      loadFailed: string
      restartRequired: string
      restartBackend: string
      restartingBackend: string
      restartFailed: string
      auxiliaryTitle: string
      resetAllToMain: string
      staleAuxDismiss: string
      auxiliaryDesc: string
      setToMain: string
      change: string
      autoUseMain: string
      inheritMainEffort: string
      providerDefault: string
      fallbackAdd: string
      fallbackEmpty: string
      notInCatalog: string
      moaTitle: string
      moaPreset: string
      moaDescription: string
      moaAggregator: string
      moaAggregatorBilled: string
      moaReferenceHint: string
      tasks: Record<string, AuxTaskCopy>
    }
    localModels: {
      connectionChanged: string
      title: string
      runtimeTitle: string
      runtimeReady: (backend: string) => string
      serverRunning: string
      runtimeInstalled: string
      runtimeInstalledDetail: (tag: string, backend: string) => string
      installTitle: string
      installDetail: string
      installAction: string
      installing: string
      installFailed: string
      hardwareTitle: string
      hardwareLoading: string
      vram: (label: string) => string
      ram: (label: string) => string
      unifiedMemory: string
      modelsTitle: string
      recommended: string
      /** Recommended-badge tooltip by resolver branch; unknown keys (newer
       *  backend) simply show no tooltip. */
      recommendedReason: Record<string, string>
      noRecommendationTitle: string
      noRecommendationDetail: string
      noRecommendationAction: string
      downloaded: string
      downloadAction: (size: string) => string
      downloadProgress: (done: string, total: string) => string
      downloadStatusRunning: string
      downloadSpeed: (rate: string) => string
      downloadEta: (time: string) => string
      /** Duration units the ETA is composed from; hours carries its
       *  remainder so a locale orders the two parts itself. */
      downloadEtaSeconds: (count: number) => string
      downloadEtaMinutes: (count: number) => string
      downloadEtaHours: (hours: number, minutes: number) => string
      downloadPausedLabel: string
      downloadPauseAction: string
      downloadResumeAction: string
      downloadDoneToast: (model: string) => string
      installDoneToast: string
      quickstartTitle: string
      quickstartDetail: (model: string, size: string) => string
      quickstartDetailReady: (model: string) => string
      quickstartAction: string
      quickstartConfigure: string
      quickstartDoneToast: (model: string) => string
      quickstartFailed: string
      quickstartStageEngine: string
      quickstartStageModel: string
      quickstartStageFinish: string
      useAction: string
      activePill: string
      updateTitle: string
      updateDetail: (next: string, current: string) => string
      updateAction: string
      updating: string
      upToDateTitle: string
      upToDateDetail: (tag: string, backend: string) => string
      activeDetail: string
      activeNotLoaded: string
      loadedPill: string
      placementResident: string
      placementSpilled: string
      placementResidentTip: string
      placementSpilledTip: string
      loadingPill: string
      ejectTip: string
      ejected: string
      ejectFailed: string
      stopServer: string
      startServer: string
      runtimeRunningDetail: string
      serverStopped: string
      serverStarted: string
      serverStopFailed: string
      serverStartFailed: string
      activating: string
      activateFailed: (model: string) => string
      activateDoneToast: (model: string) => string
      downloadFailed: (model: string) => string
      downloadPauseFailed: (model: string) => string
      downloadResumeFailed: (model: string) => string
      pillFitsGpu: string
      pillUsesRam: string
      pillTooBig: string
      browseTitle: string
      browseHint: string
      browsePlaceholder: string
      browseSearching: string
      browseListing: string
      browseShowFiles: string
      browseRefresh: string
      browseDownloads: string
      browseLikes: string
      browseGated: string
      browseNoGguf: string
      browseFitUnknown: string
      browseAlreadyDownloaded: string
      addedByYou: string
      browseDownloadStarted: string
      browseDownloadAria: string
      sideloadButton: string
      sideloadTitle: string
      sideloadDone: string
      sideloadAlreadyPresent: string
      pillFullContext: (max: string) => string
      pillFullContextTip: string
      pillUpTo: (max: string) => string
      pillGrowsTip: string
      pillVision: string
      deleteAction: string
      deleteConfirm: (model: string) => string
      deleted: (model: string) => string
      deleteFailed: string
    }
    billing: {
      perMonth: (amount: string) => string
      creditsPerMonth: (amount: string) => string
      usageLabel: (label: string) => string
      freeTier: {
        signIn: string
        title: string
        message: string
        caption: string
        name: string
        footnote: string
        plan: string
        model: string
        connectors: string
        included: string
      }
      amountValidation: {
        reloadTo: string
        greaterThanThreshold: string
        decimal: (label: string) => string
        positive: (label: string) => string
        minimum: (label: string, amount: string) => string
        maximum: (label: string, amount: string) => string
      }
      stepUp: {
        openVerification: string
        dismiss: string
        waiting: string
        verify: string
        deniedTitle: string
        deniedBody: string
        successTitle: string
        successBody: string
      }
      charge: {
        added: (amount: string) => string
        failedTitle: string
        unconfirmedTitle: string
        unconfirmedBody: (message: string) => string
        checkTitle: string
        checkBody: string
        untrackedTitle: string
        untrackedBody: string
        timeoutTitle: string
        timeoutBody: string
        authenticationRequired: string
        expired: string
        declined: string
        failedBody: (reason: string) => string
      }
      title: string
      preview: string
      summary: {
        balance: string
        plan: string
        autoRefill: string
      }
      sections: {
        invoices: string

        plan: string
        paymentAndCredits: string
        usage: string
      }
      usage: {
        title: string
      }
      buyCredits: {
        customAmount: string
        title: string
        buyButton: string
        processing: string
        added: (amount: string) => string
        retry: string
        openPortal: string
      }
      plan: {
        title: string
        changePlan: string
        viewPlans: string
        backAria: string
        current: string
        scheduled: string
        empty: string
        undo: string
        undoing: string
        downgrade: string
        confirmDowngrade: string
        tryAgain: string
        checkingChange: string
        cannotChange: string
        alreadyOn: (name: string) => string
        notScheduleable: string
        scheduling: string
        cancel: string
        effectScheduled: (targetName: string, effectiveAt: string, creditsDelta: string) => string
      }
      autoReload: {
        threshold: string
        thresholdAria: string
        reloadTo: string
        reloadToAria: string
        turnOffConfirm: string
        turnOff: string
        disable: string
        updated: string
        turnedOff: string
        manage: string
        save: string
        saving: string
        cancel: string
      }
      state: {
        notice: {
          loggedOut: { title: string; message: string; action: string }
          openPortal: string
          noCard: { title: string; message: string; action: string }
        }
        paymentMethod: {
          title: string
          description: string
          addAction: string
          updateAction: string
          provenance: {
            autoRefill: string
            customerDefault: string
            subPin: string
            suffix: (label: string) => string
          }
        }
        buyCredits: {
          description: string
        }
        autoRefill: {
          title: string
          genericDescription: string
          offPill: string
          enabledPill: string
          notAvailablePill: string
          manageCaption: string
          turnOnCaption: string
          chargesDescription: (reloadTo: string, threshold: string) => string
          distinctCardCaption: (cardLabel: string) => string
          distinctCardFallback: string
          reconcileAction: string
        }
        usage: {
          subscriptionCredits: {
            title: string
            barLabel: string
            captionResets: (date: string) => string
            valueOf: (remaining: string, monthly: string) => string
            valueOver: (remaining: string, monthly: string, over: string) => string
          }
          topupCredits: {
            title: string
            caption: string
          }
          monthlyCap: {
            title: string
            barLabel: string
            captionDefault: string
            captionSpending: string
            valueUsed: (spent: string, limit: string) => string
          }
        }
        planCard: {
          freeTier: string
          chooseAction: string
          adjustPlanAction: string
          unavailableCaption: string
          downgradeCaption: (tierName: string, when: string) => string
          cancellationCaption: (when: string) => string
          renewsCaption: (date: string) => string
          noSubscriptionCaption: string
        }
      }
      errors: {
        consentRequired: { title: string; message: string }
        insufficientScope: { title: string; message: string }
        remoteSpendingRevoked: {
          title: string
          messageByAdmin: string
          messageBySelf: string
        }
        remoteSpendingReconnect: (who: string) => string
        sessionRevoked: { title: string; message: string }
        cliBillingDisabled: { title: string; message: string }
        roleRequired: { title: string; message: string }
        idempotencyConflict: { title: string; message: string }
        noPaymentMethod: { title: string; message: string }
        orgAccessDenied: { title: string; message: string }
        monthlyCapExceeded: {
          title: string
          messageReached: string
          messageHeadroom: (remaining: string) => string
        }
        rateLimited: {
          title: string
          message: (mins: number) => string
        }
        stripeUnavailable: {
          title: string
          message: (mins: number) => string
        }
        upgradeCapExceeded: { title: string; message: string }
        endpointUnavailable: { title: string; message: string }
        timeout: { title: string; message: string }
        transport: { title: string; message: string }
        default: { title: string; message: string }
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
      localEndpoint: {
        title: string
        description: string
      }
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
      autoArchiveTitle: string
      autoArchiveDesc: string
      autoArchiveDaysLabel: string
      autoArchiveDaysUnit: string
      autoArchiveFailed: string
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
      needsSignIn: string
      needsSetup: string
      activeBackend: string
      activeBackendHint: string
      useBackend: string
      nousIncluded: string
      nousAuthNeededTitle: string
      nousAuthNeededMessage: (provider: string) => string
      nousAuthSignIn: string
      nousAuthDoneTitle: string
      nousAuthDoneMessage: string
      nousAuthFailed: string
      nousAuthFailedMessage: string
      nousAuthTryAgain: string
      noApiKeyRequired: string
      postSetupHint: (step: string) => string
      postSetupInstalledHint: string
      postSetupRun: string
      postSetupRerun: string
      postSetupInstalled: string
      postSetupRunning: string
      postSetupStarting: string
      postSetupCompleteTitle: string
      postSetupCompleteMessage: (step: string) => string
      postSetupErrorTitle: string
      postSetupErrorMessage: (step: string) => string
      postSetupOpenLogs: string
      postSetupRunAgain: string
      postSetupFailed: (step: string) => string
      webSearchActive: (backend: string) => string
      webExtractActive: (backend: string) => string
      webCapabilityUnset: string
      webUseForSearch: string
      webUseForExtract: string
      webUsedForSearch: string
      webUsedForExtract: string
      webCapabilitySelectedMessage: (provider: string, capability: string) => string
      failedSelectCapability: (provider: string) => string
      loadingModels: string
      modelSectionTitle: string
      modelCount: (count: number) => string
      modelInUse: string
      modelDefault: string
      modelInactiveHint: string
      modelSelectedTitle: string
      modelSelectedMessage: (model: string) => string
      failedSelectModel: (model: string) => string
      terminalBackend: {
        sectionTitle: string
        loading: string
        failedLoad: string
        ready: string
        needsSetup: string
        unavailable: string
        inUse: string
        selectedTitle: string
        selectedMessage: (backend: string) => string
        failedSelect: (backend: string) => string
        needsSetupHint: string
        needsSetupConfirmTitle: (backend: string) => string
        needsSetupConfirmDescription: (detail: string) => string
        needsSetupConfirmDescriptionGeneric: string
        needsSetupConfirmAction: string
        unavailableTitle: string
        unavailableMessage: (backend: string) => string
        openBackendSettings: string
        useLocal: string
        switchedToLocal: string
      }
      browserRealProfile: {
        label: string
        description: string
        enabledTitle: string
        enabledMessage: string
        disabledTitle: string
        disabledMessage: string
        failedSave: string
        prompt: {
          title: string
          body: string
          bulletSnapshot: string
          bulletLiveProfile: string
          bulletLocal: string
          dontShowAgain: string
          notNow: string
          enable: string
        }
      }
    }
  }

  modelAssignment: {
    saveFailed: string
    confirmTitle: string
    confirmDetail: string
    confirmAction: string
    declined: string
  }

  modelPicker: {
    title: string
    current: string
    unknown: string
    search: string
    noModels: string
    addProvider: string
    loadFailed: string
    loadingIntoMemory: string
    downloading: string
    localDownloadsHeading: string
    noAuthenticatedProviders: string
    pro: string
    proNeedsSubscription: string
    free: string
    freeTier: string
    priceTitle: string
    wasPrice: string
    customModel: string
    addCustomModelAction: string
    customModelPlaceholder: string
  }

  modelVisibility: {
    title: string
    search: string
    noAuthenticatedProviders: string
    addProvider: string
    addCustomModel: string
    removeCustomModel: string
    resetToDefaults: string
    resetConfirm: string
    resetDescription: string
    resetAction: string
  }

  interfaceMode: {
    title: string
    hint: string
    sessionNote: string
    simple: { label: string; description: string }
    advanced: { label: string; description: string }
  }
}
