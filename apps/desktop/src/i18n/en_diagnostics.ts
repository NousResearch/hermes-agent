import type { Translations } from './types'

export const enDiagnostics = {
  notifications: {
    region: 'Notifications',
    hide: 'Hide',
    show: 'Show',
    more: count => `${count} more ${count === 1 ? 'notification' : 'notifications'}`,
    clearAll: 'Clear all',
    dismiss: 'Dismiss notification',
    details: 'Details',
    copyDetail: 'Copy detail',
    copyDetailFailed: 'Could not copy notification detail',
    backendOutOfDateTitle: 'Backend out of date',
    backendOutOfDateMessage:
      'Your Hermes backend is older than this desktop build and may not work correctly. Update to align them.',
    installMethodUnsupportedTitle: 'Unsupported install method',
    updateHermes: 'Update Hermes',
    updateReadyTitle: 'Update ready',
    updateReadyMessage: count => `${count} new change${count === 1 ? '' : 's'} available.`,
    updateReadyMessageUnknown: 'A new update is available.',
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
      methodNotAllowed: "Hermes' background service is out of step with the app, probably after an update. Restart it to fix this.",
      microphonePermission: 'Microphone permission was denied.',
      openaiRejectedApiKey: "OpenAI didn't accept your API key. Update it in Settings → Keys, then try again.",
      openaiTtsNeedsKey: 'Voice needs an OpenAI key. Add one in Settings → Keys.',
      codeSkewRestartRequired: 'Hermes was updated but is still running the old version. Restart it to finish the update.',
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
      turnDoneBody: '',
      turnErrorTitle: 'Turn failed',
      backgroundDoneTitle: 'Background task finished',
      backgroundFailedTitle: 'Background task failed',
      creditsTitle: 'Credits'
    }
  },

  errors: {
    genericFailure: 'Something went wrong',
    boundaryTitle: 'Something broke in the interface',
    boundaryDesc: 'The view hit an unexpected error. Your chats and settings are safe.',
    boundaryDetails: 'Details',
    sendDiagnostics: 'Send diagnostics',
    reloadWindow: 'Reload window',
    openLogs: 'Open logs'
  },

  sendDiagnostics: {
    title: 'Send diagnostics to Nous',
    privacyNotice:
      'This uploads a debug bundle to Nous-internal storage (not a public paste). It includes system info (OS, versions, provider, which API keys are configured — never the keys themselves) and full agent, gateway, and desktop logs (up to 512 KB each), which likely contain conversation content, tool outputs, and file paths. Secrets are redacted before upload. The bundle is viewable only by Nous staff and allowlisted Discord moderators, and auto-deletes after 14 days.',
    upload: 'Upload',
    uploading: 'Uploading…',
    cancel: 'Cancel',
    close: 'Close',
    copyLink: 'Copy link',
    uploadIdFallback: id => `No view link returned — quote upload ID ${id} to support`,
    doneTitle: 'Diagnostics sent',
    doneDescription:
      'Your bundle was uploaded privately. Share the link below in your support thread so the team can see your logs.',
    failedTitle: 'Upload failed',
    failedHint:
      'You can also run `hermes debug share --nous` from a terminal, or `hermes debug share --local` to print the report without uploading.',
    handoffLead: 'Pick up the discussion in:',
    links: {
      github: 'GitHub Issues',
      portal: 'Nous Portal Support',
      discord: 'Discord'
    }
  }
} satisfies Pick<Translations, 'notifications' | 'errors' | 'sendDiagnostics'>
