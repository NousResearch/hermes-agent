import type { TipId } from '@/lib/tips/catalog'

export interface ChatTranslations {
  composer: {
    message: string
    wakingProfile: (profile: string) => string
    placeholderStarting: string
    placeholderReconnecting: string
    placeholderFollowUp: string
    newSessionPlaceholders: readonly string[]
    followUpPlaceholders: readonly string[]
    startVoice: string
    openDirective: string
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
    voiceControls: string
    voiceEngine: string
    voiceEngineChained: string
    voiceEngineLive: string
    voiceEngineLiveNeedsKey: string
    voiceEngineChangeFailed: string
    voiceEngineChainedShort: string
    voiceEngineLiveShort: string
    voiceDictation: string
    speakReplies: string
    stopSpeakingReplies: string
    wakeWord: (phrase: string) => string
    wakeWordListening: (phrase: string) => string
    wakeWordOff: (phrase: string) => string
    wakeWordPausedVoice: (phrase: string) => string
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
    queuedPaused: (count: number) => string
    attachmentOnly: string
    emptyTurn: string
    hiddenQueued: string
    attachments: (count: number) => string
    editingInComposer: string
    editingQueuedInComposer: string
    restoredDraftNotice: string
    restoredDraftUndo: string
    queueEdit: string
    queueSendNext: string
    queueSend: string
    queueSteer: string
    queueDelete: string
    queueResume: string
    queueResumeTip: string
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
    mcpSuggestions: {
      label: (server: string) => string
      tip: (keyword: string) => string
      connecting: (server: string) => string
      cancelTip: string
      added: (server: string) => string
      addedTip: string
      connectFailed: (server: string) => string
    }
    skillSuggestions: {
      label: (skill: string) => string
      tip: (skill: string) => string
      done: (skill: string) => string
      doneTip: string
    }
    githubSuggestions: {
      label: string
      tip: string
      done: string
      doneTip: string
    }
    repairSuggestions: {
      label: (server: string) => string
      tip: (server: string) => string
      working: (server: string) => string
      workingTip: string
      done: (server: string) => string
      doneTip: string
      failed: (server: string) => string
    }
    cronSuggestions: {
      label: string
      tip: (phrase: string) => string
      prefix: string
      done: string
      doneTip: string
    }
  }

  statusStack: {
    agents: string
    background: (count: number) => string
    goalActive: string
    goalBlocked: string
    goalDone: string
    goalPaused: string
    goalWaiting: string
    subagents: (count: number) => string
    todos: (done: number, total: number) => string
    running: string
    stop: string
    dismiss: string
    exit: (code: number) => string
    control: {
      goalActiveTurns: (turn: number, maxTurns: number) => string
      goalDoneTurns: (turns: number) => string
      goalTurn: (turn: number) => string
      goalActions: string
      viewDetails: string
      addCriterion: string
      addCriterionDialogTitle: string
      addCriterionPlaceholder: string
      criterionLabel: string
      pauseGoal: string
      resumeGoal: string
      resumeNow: string
      clearGoal: string
      clearGoalConfirmTitle: string
      clearGoalConfirmBody: string
      copyCriterion: (index: number) => string
      removeCriterion: (index: number) => string
      removeCriterionConfirmTitle: (index: number) => string
      removeCriterionConfirmBody: (index: number) => string
      clearCriteria: string
      clearCriteriaConfirmTitle: string
      clearCriteriaConfirmBody: string
      criteriaHeader: (count: number) => string
      noCriteria: string
      goalDetailsTitle: string
      objectiveLabel: string
      contractOutcome: string
      contractVerification: string
      contractConstraints: string
      contractBoundaries: string
      contractStopWhen: string
      waitBarrierTitle: string
      waitUntil: (target: string) => string
      waitSession: (target: string) => string
      waitPid: (pid: number) => string
      qualityGatesTitle: string
      gateCommand: string
      gateAttempts: (attempts: number, max: number) => string
      gateTimeout: (seconds: number) => string
      gateLastExit: (code: number | null) => string
      loopActive: string
      loopPaused: string
      loopDeferred: string
      loopFinished: string
      loopRuns: (runs: number) => string
      loopRunCount: (current: number, total: number) => string
      loopNext: (time: string) => string
      loopEverySeconds: (seconds: number) => string
      loopEveryMinutes: (minutes: number) => string
      loopEveryHours: (hours: number) => string
      loopSelfPaced: string
      loopActions: string
      pauseLoop: string
      resumeLoop: string
      stopLoop: string
      stopLoopConfirmTitle: string
      stopLoopConfirmBody: string
      dismissLoop: string
      loopPromptLabel: string
      loopCadenceLabel: string
      loopUntilLabel: string
      loopDeferredNotice: string
      loopAwaitingResponse: string
      heartbeatActive: string
      heartbeatPaused: string
      heartbeatEveryMinutes: (minutes: number) => string
      heartbeatEveryHours: (hours: number) => string
      heartbeatEverySeconds: (seconds: number) => string
      heartbeatNext: (time: string) => string
      heartbeatDueWaitingForIdle: string
      heartbeatActions: string
      pauseHeartbeat: string
      resumeHeartbeat: string
      clearHeartbeat: string
      clearHeartbeatConfirmTitle: string
      clearHeartbeatConfirmBody: string
      heartbeatFiredCount: (count: number) => string
      actionFailed: (msg: string) => string
      actionSucceeded: string
      copySuccess: string
      copyFailure: string
      continuationFailed: string
      continuationQueued: string
      continuationBusy: string
      controlUnavailable: (msg: string) => string
      dismissError: string
      add: string
    }
    coding: {
      title: string
      noBranch: string
      detached: string
      clean: string
      changed: (count: number) => string
      ahead: (count: number) => string
      behind: (count: number) => string
      review: string
      close: string
      openChanges: string
      openFile: string
      stage: string
      unstage: string
      stageAll: string
      viewAsTree: string
      viewAsList: string
      revert: string
      revertAll: string
      revertConfirm: string
      revertAllConfirm: string
      staged: string
      noChanges: string
      notRepo: string
      noDiff: string
      scopeUncommitted: string
      scopeBranch: string
      scopeLastTurn: string
      commit: string
      commitAndPush: string
      commitPlaceholder: (shortcut: string) => string
      generateCommitMessage: string
      stopGenerating: string
      createPr: string
      openPr: string
      ghMissing: string
      agentShip: string
      agentShipUnavailable: string
      agentShipPrompt: string
      newBranch: string
      branchOffFrom: (base: string) => string
      switchTo: (branch: string) => string
      switchFailed: (branch: string) => string
      worktrees: string
    }
  }

  prompts: {
    gatewayDisconnected: string
    reconnect: string
    sudoSendFailed: string
    secretSendFailed: string
    sudoTitle: string
    sudoDesc: string
    sudoCommandUnavailable: string
    sudoPlaceholder: string
    secretTitle: string
    secretDesc: string
    secretPlaceholder: string
    vaultUnlockSendFailed: string
    vaultUnlockTitle: (name: string) => string
    vaultUnlockDesc: (name: string) => string
    vaultSaveSendFailed: string
    vaultSaveTitle: (site: string) => string
    vaultSaveDesc: (origin: string) => string
    vaultSaveIdentifierLabel: string
    vaultSaveIdentifierPlaceholder: string
    vaultSavePasswordPlaceholder: string
    vaultSaveFootnote: string
    vaultSaveDecline: string
    vaultSaveConfirm: string
    vaultCodeSendFailed: string
    vaultCodeTitle: (site: string) => string
    vaultCodeDesc: (site: string) => string
    vaultCodeLabel: string
    vaultCodeFootnote: string
    vaultCodeSkip: string
    vaultCodeConfirm: string
    vaultUnlockPlaceholder: string
    vaultUnlockKeepLocked: string
    vaultUnlockConfirm: string
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
    editTurnUnavailable: string
    resumeFailed: string
    readOnlyTranscriptTitle: string
    readOnlyTranscriptBody: string
    readOnlyTranscriptSendBlocked: string
    resumeStrandedTitle: string
    resumeStrandedBody: string
    poolSlotTimeoutBody: string
    poolSlotTimeoutOpenSettings: string
    resumeRetry: string
    nothingToBranch: string
    branchNeedsChat: string
    sessionBusy: string
    branchStopCurrent: string
    branchNoText: string
    branchTitle: (n: number) => string
    branchFailed: string
    deleteFailed: string
    archived: string
    archiveFailed: string
    cwdChangeFailed: string
    cwdStagedTitle: string
    cwdStagedMessage: string
    modelSwitchConfirmBody: string
    modelSwitchConfirmLabel: string
    modelSwitchConfirmTitle: (model: string) => string
    modelSwitchConfirmTitleFallback: string
    modelSwitchFailed: string
    modelSwitchKeepLabel: string
    modelSwitchStaleNotice: string
    hydrationSyncing: (profile: string) => string
    sessionExported: string
    sessionExportFailed: string
    imageSaved: string
    downloadStarted: string
    restartToUseSaveImage: string
    restartToSaveImages: string
    imageDownloadFailed: string
    openImage: string
    downloadImage: string
    savingImage: string
    imagePreviewFailed: string
    imageAttach: string
    imageWriteFailed: string
    imageAttachFailed: string
    pastedContent: string
    pasteAttachFailed: string
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
      startMessaging: string
    }
  }

  tips: {
    close: string
    /** Keyed by `TipId`, so a new tip without copy is a type error. Plus the
     *  campaign tips, which live outside the rotation's catalog: they carry
     *  a button, and `action` is its label. */
    items: Record<TipId, { title: string; text: string }> & {
      'local-runtime-update': { title: string; text: string; action: string }
      'local-setup': { title: string; text: string; action: string }
    }
  }
}
