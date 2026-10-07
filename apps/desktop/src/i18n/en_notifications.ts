import type { Translations } from './types'

// The notifications copy (shared profile warning, backend/app version drift, voice errors),
// composed by en.ts.
export const enNotifications: Translations['notifications'] = {
    sharedProfileWarning:
      'Another Hermes installation is using this profile. Both installations share its settings and data, so changes can conflict. You can continue, or close the other installation before making changes.',
    region: 'Notifications',
    hide: 'Hide',
    show: 'Show',
    more: count => `${count} more ${count === 1 ? 'notification' : 'notifications'}`,
    clearAll: 'Clear all',
    dismiss: 'Dismiss notification',
    details: 'Details',
    copyDetail: 'Copy detail',
    copyDetailFailed: 'Could not copy notification detail',
    compressDeferredDone: 'Context compression finished',
    backendOutOfDateTitle: 'Backend out of date',
    backendOutOfDateMessage:
      'Your Hermes backend is older than this desktop build and may not work correctly. Update to align them.',
    desktopOutOfDateTitle: 'Hermes app out of date',
    desktopOutOfDateMessage:
      'This Hermes app is older than the backend it is connected to and may not work correctly. Update the app to align them.',
    updateDesktopApp: 'Update app',
    installMethodUnsupportedTitle: 'Unsupported install method',
    updateHermes: 'Update Hermes',
    updateReadyTitle: 'Update ready',
    updateReadyMessage: count => `${count} new change${count === 1 ? '' : 's'} available.`,
    updateReadyMessageUnknown: 'A new update is available.',
    updateReadyMessageAppInstaller: 'A new version of Hermes is ready. Update now and Windows will finish it for you.',
    seeWhatsNew: "See what's new",
    mcp: {
      needsAuthTitle: 'MCP server needs re-authentication',
      needsAuthMessage: name => `${name} MCP needs re-authentication.`,
      errorTitle: 'MCP server unreachable',
      errorMessage: name => `${name} MCP failed its health check.`,
      signIn: 'Sign in',
      view: 'View',
      disable: 'Disable',
      disabledMessage: name => `${name} MCP disabled. Re-enable it any time from Capabilities → MCP.`,
      disableFailed: name => `Could not disable ${name} MCP.`
    },
    errors: {
      elevenLabsNeedsKey: 'Voice input needs an ElevenLabs key. Add one in Settings → Keys.',
      elevenLabsRejectedKey: "ElevenLabs didn't accept your API key. Update it in Settings → Keys, then try again.",
      diskFull: 'Disk full — free some space, then try again.',
      storageFailure: "Hermes couldn't save to its data folder. Open Maintenance to check and repair it.",
      gatewayAuthFailed:
        'This Hermes no longer accepts your saved sign-in. Open Gateways and sign in again (or paste a new access token), then retry.',
      methodNotAllowed:
        "Hermes' background service is out of step with the app, probably after an update. Restart it to fix this.",
      microphonePermission: 'Microphone permission was denied.',
      openaiRejectedApiKey: "OpenAI didn't accept your API key. Update it in Settings → Keys, then try again.",
      openaiTtsNeedsKey: 'Voice needs an OpenAI key. Add one in Settings → Keys.',
      codeSkewRestartRequired:
        'Hermes was updated but is still running the old version. Restart it to finish the update.',
      rpcOutOfSync: 'The app and the backend are on different versions. Update both.',
      restartHermesFailed: "Couldn't restart Hermes"
    },
    actions: {
      restartHermes: 'Restart Hermes',
      openKeys: 'Open Keys',
      openGateways: 'Open Gateways',
      openMaintenance: 'Open Maintenance'
    },
    voice: {
      configureSpeechToText: 'Configure speech-to-text to use voice mode.',
      couldNotStartSession: 'Could not start voice session',
      microphoneAccessDenied: 'Microphone access denied.',
      microphoneConstraintsUnsupported: 'Microphone constraints are not supported by this device.',
      microphoneFailed: 'Microphone failed',
      microphoneInUse: 'Microphone is already in use by another app.',
      microphonePermissionDenied: 'Microphone permission was denied.',
      microphoneSecureContextRequired: 'Microphone recording requires HTTPS, localhost, or the native desktop app.',
      microphoneStartFailed: 'Could not start microphone recording.',
      microphoneUnsupported: 'This runtime does not support microphone recording.',
      noMicrophone: 'No microphone was found.',
      noSpeechDetected: 'No speech detected',
      playbackFailed: 'Voice playback failed',
      recordingFailed: 'Voice recording failed',
      sayStopToEnd: phrase => `Say "${phrase}" to end the voice chat.`,
      transcriptionFailed: 'Voice transcription failed',
      transcriptionUnavailable: 'Voice transcription is not available yet.',
      tryRecordingAgain: 'Try recording again.',
      unavailable: 'Voice unavailable',
      liveEnded: 'Live voice session ended',
      liveEndedConnectionLost: 'The live voice session lost its connection.',
      liveEndedClosed: 'The live voice session was closed by the service.',
      liveError: 'Live voice',
      liveDelegationFailed: 'Could not hand the request to Hermes',
      liveUnavailable: reason => `GPT-Live voice chat is not available: ${reason}. Using speech-to-text instead.`
    },
    native: {
      approvalTitle: 'Approval needed',
      approvalTitleNamed: session => `Approval needed — ${session}`,
      approveAction: 'Approve',
      rejectAction: 'Reject',
      inputTitle: 'Input needed',
      inputTitleNamed: session => `Input needed — ${session}`,
      inputBody: 'Hermes is waiting for your response.',
      turnDoneTitle: 'Hermes finished',
      turnDoneBody: 'Message complete.',
      turnErrorTitle: 'Turn failed',
      backgroundDoneTitle: 'Background task finished',
      backgroundFailedTitle: 'Background task failed',
      creditsTitle: 'Credits'
    }
}
