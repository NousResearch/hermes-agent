export interface DiagnosticsTranslations {
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
    installMethodUnsupportedTitle: string
    updateHermes: string
    updateReadyTitle: string
    updateReadyMessage: (count: number) => string
    updateReadyMessageUnknown: string
    seeWhatsNew: string
    mcp: {
      needsAuthTitle: string
      needsAuthMessage: (name: string) => string
      errorTitle: string
      errorMessage: (name: string) => string
      signIn: string
      view: string
      disable: string
      disabledMessage: (name: string) => string
      disableFailed: (name: string) => string
    }
    errors: {
      elevenLabsNeedsKey: string
      elevenLabsRejectedKey: string
      diskFull: string
      storageFailure: string
      gatewayAuthFailed: string
      methodNotAllowed: string
      microphonePermission: string
      openaiRejectedApiKey: string
      openaiTtsNeedsKey: string
      codeSkewRestartRequired: string
      restartHermesFailed: string
    }
    actions: {
      restartHermes: string
      openKeys: string
      openGateways: string
      openMaintenance: string
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
      sayStopToEnd: (phrase: string) => string
      transcriptionFailed: string
      transcriptionUnavailable: string
      tryRecordingAgain: string
      unavailable: string
      liveEnded: string
      liveEndedConnectionLost: string
      liveEndedClosed: string
      liveError: string
      liveDelegationFailed: string
      liveUnavailable: (reason: string) => string
    }
    // Native OS notification copy (titles + generic fallback bodies). Dynamic
    // bodies (the agent's reply, a command, an error) are passed through raw.
    native: {
      approvalTitle: string
      approvalTitleNamed: (session: string) => string
      approveAction: string
      rejectAction: string
      inputTitle: string
      inputTitleNamed: (session: string) => string
      inputBody: string
      turnDoneTitle: string
      turnDoneBody: string
      turnErrorTitle: string
      backgroundDoneTitle: string
      backgroundFailedTitle: string
      creditsTitle: string
    }
  }

  sendDiagnostics: {
    title: string
    privacyNotice: string
    upload: string
    uploading: string
    cancel: string
    close: string
    copyLink: string
    uploadIdFallback: (id: string) => string
    doneTitle: string
    doneDescription: string
    failedTitle: string
    failedHint: string
    handoffLead: string
    links: {
      discord: string
      github: string
      portal: string
    }
  }

  errors: {
    genericFailure: string
    boundaryTitle: string
    boundaryDesc: string
    boundaryDetails: string
    sendDiagnostics: string
    reloadWindow: string
    openLogs: string
  }
}
