import type { Translations } from './types'

export const enBoot = {
  boot: {
    ready: 'Hermes Desktop is ready',
    desktopBootFailedWithMessage: message => `Desktop boot failed: ${message}`,
    steps: {
      connectingGateway: 'Connecting live desktop gateway',
      loadingSettings: 'Loading Hermes settings',
      loadingSessions: 'Loading recent sessions',
      retryingRemoteBackend: 'Reconnecting to the remote Hermes backend…',
      startingDesktopConnection: 'Starting desktop connection',
      startingHermesDesktop: 'Starting Hermes Desktop…'
    },
    errors: {
      backgroundExited: 'The service that runs your chats closed unexpectedly. Restart it to keep going — your chats and settings are safe.',
      backgroundExitedDuringStartup: 'Hermes stopped right after it started.',
      backendStopped: 'Hermes stopped working in the background',
      restartHermes: 'Restart Hermes',
      openLogs: 'Open logs',
      desktopBootFailed: "Hermes couldn't start",
      gatewayConnectionLost: 'Hermes lost its connection',
      gatewayConnectionLostDetail:
        'Still trying to reconnect. You can keep reading and drafting. If this keeps up, reconnect now or check your connection settings.',
      reconnectNow: 'Reconnect now',
      connectionSettings: 'Connection settings',
      gatewaySignInRequired: 'Your remote Hermes signed you out',
      gatewaySignInRequiredDetail: 'Sign in again to reconnect. Your chats and settings are safe.',
      signInAgain: 'Sign in again',
      ipcBridgeUnavailable: "Hermes Desktop couldn't talk to its own background layer. Restart the app."
    },
    // Plain causes for a local backend boot failure (`classifyBootFailure`);
    // the raw output stays behind "Show recent logs".
    causes: {
      exitedEarly: "Hermes' background service stopped right after starting.",
      timedOut: "Hermes' background service didn't answer in time.",
      permission: "Hermes couldn't write to its data folder (permission problem).",
      diskFull: 'The disk is full, so Hermes could not start.',
      portInUse: 'Another program is using the network port Hermes needs.',
      installMissing: "Part of Hermes' installation is missing. Choose Repair install to put it back."
    },
    failure: {
      title: "Hermes couldn't start",
      description:
        "Hermes' background service didn't come up. Try one of the recovery steps below. Nothing here deletes your chats or settings.",
      details: 'Details',
      remoteTitle: 'Remote gateway sign-in required',
      remoteDescription:
        'Your remote gateway session has expired. Sign in again to reconnect. Nothing here deletes your chats or settings.',
      retry: 'Retry',
      repairInstall: 'Repair install',
      useLocalGateway: 'Use local gateway',
      gatewaySettings: 'Gateway settings',
      back: 'Back',
      openLogs: 'Open logs',
      repairHint: 'Repair re-runs the installer and can take a few minutes on a fresh machine.',
      remoteSignInHint: signInLabel =>
        `Signs out of the saved remote browser session, then opens ${signInLabel}. Use local gateway to switch to the bundled backend instead.`,
      signOutAndSignIn: 'Sign out & sign in',
      remoteFailureHint: 'Check the gateway URL and sign-in under Gateway settings, or switch to the local gateway.',
      cloudDownTitle: 'Nous Cloud agent is down',
      cloudDownDescription:
        'The Nous-managed cloud agent this gateway connects to is returning a server error. It cannot be restarted from here — check its status, switch to the local gateway, or get support.',
      cloudDownHint:
        'The buttons below open the Nous Portal (instance status and controls) and our Discord for support.',
      cloudDownCheckPortal: 'Check Portal status',
      cloudDownDiscord: 'Get help on Discord',
      hideRecentLogs: 'Hide recent logs',
      showRecentLogs: 'Show recent logs',
      signedInTitle: 'Signed in',
      signedInMessage: 'Reconnecting to the remote gateway…',
      signInIncompleteTitle: 'Sign-in incomplete',
      signInIncompleteMessage: 'The login window closed before authentication finished.',
      signInFailed: 'Sign-in failed',
      signInToRemoteGateway: 'Sign in to remote gateway',
      signInWithProvider: provider => `Sign in with ${provider}`,
      identityProvider: 'your identity provider'
    }
  },

  onboarding: {
    headerTitle: "Let's get you setup with Hermes Agent",
    headerDesc: 'Connect a model provider to start chatting. Most options take one click.',
    preparingInstall: 'Hermes is finishing install. This usually takes under a minute on first run.',
    starting: 'Starting Hermes…',
    lookingUpProviders: 'Looking up providers...',
    collapse: 'Collapse',
    otherProviders: 'Other providers',
    haveApiKey: 'I have an API key',
    chooseLater: "I'll choose a provider later",
    recommended: 'Recommended',
    connected: 'Connected',
    featuredPitch: 'One subscription, 300+ frontier models — the recommended way to run Hermes',
    fireworksPitch: 'Direct model API — Fireworks-hosted frontier models',
    localModelsTitle: 'Run models locally',
    localModelsPitch: 'No account needed — download a model and run it on this machine',
    openRouterPitch: 'One key, hundreds of models — a solid default',
    apiKeyOptions: {
      fireworks: {
        short: 'direct model API',
        description: 'Direct access to models hosted by Fireworks AI.'
      },
      openrouter: {
        short: 'one key, many models',
        description: 'Hosts hundreds of models behind a single key. Good default for new installs.'
      },
      openai: { short: 'GPT-class models', description: 'Direct access to OpenAI models.' },
      gemini: { short: 'Gemini models', description: 'Direct access to Google Gemini models.' },
      xai: { short: 'Grok models', description: 'Direct access to xAI Grok models.' },
      local: {
        short: 'self-hosted',
        description: 'Point Hermes at a local or self-hosted OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama, etc).'
      }
    },
    backToSignIn: 'Back to sign in',
    getKey: 'Get a key',
    replaceCurrent: 'Replace current value',
    pasteApiKey: 'Paste API key',
    localApiKeyPlaceholder: 'API key (optional — only if your endpoint requires one)',
    couldNotSave: 'Could not save credential.',
    connecting: 'Connecting',
    update: 'Update',
    flowSubtitles: {
      pkce: 'Opens your browser to sign in, then continues here',
      device_code: 'Opens a verification page in your browser — Hermes connects automatically',
      external: 'Sign in once in your terminal, then come back to chat'
    },
    startingSignIn: provider => `Starting sign-in for ${provider}...`,
    verifyingCode: provider => `Verifying your code with ${provider}...`,
    connectedProvider: provider => `${provider} connected`,
    connectedPicking: provider => `${provider} connected. Picking a default model...`,
    signInFailed: 'Sign-in failed. Try again.',
    signInExpired:
      'The sign-in page timed out before you finished. Try again and complete the browser step within a few minutes, or use an API key instead.',
    signInDidNotFinish: provider =>
      `Sign-in with ${provider} did not finish. Check your internet connection and try again, or pick a different provider.`,
    tryAgain: 'Try again',
    useApiKeyInstead: 'Use an API key',
    errorDetails: 'Details',
    pickDifferentProvider: 'Pick a different provider',
    signInWith: provider => `Sign in with ${provider}`,
    openedBrowser: provider => `We opened ${provider} in your browser.`,
    authorizeThere: 'Authorize Hermes there.',
    copyAuthCode: 'Copy the authorization code and paste it below.',
    pasteAuthCode: 'Paste authorization code',
    reopenAuthPage: 'Re-open authorization page',
    autoBrowser: provider =>
      `We opened ${provider} in your browser. Authorize Hermes there and you'll be connected automatically — nothing to copy or paste.`,
    reopenSignInPage: 'Re-open sign-in page',
    waitingAuthorize: 'Waiting for you to authorize...',
    externalPending: provider =>
      `${provider} signs in through its own CLI. Run this command in a terminal, then come back and pick "I've signed in":`,
    signedIn: "I've signed in",
    deviceCodeOpened: provider => `We opened ${provider} in your browser. Enter this code there:`,
    reopenVerification: 'Re-open verification page',
    copy: 'Copy',
    defaultModel: 'Default model',
    freeTier: 'Free tier',
    pro: 'Pro',
    free: 'Free',
    price: (input, output) => `${input} in / ${output} out per Mtok`,
    change: 'Change',
    startChatting: 'Begin',
    docs: provider => `${provider} docs`
  },

  install: {
    stageStates: {
      pending: 'Pending',
      running: 'Installing',
      succeeded: 'Done',
      skipped: 'Skipped',
      failed: 'Failed'
    },
    oneTimeTitle: 'Hermes needs a one-time install',
    unsupportedDesc: platform =>
      `Automated first-launch install isn’t available on ${platform} yet. Open Terminal and run the command below, then relaunch this app. Subsequent launches will skip this step.`,
    installCommand: 'Install command',
    copyCommand: 'Copy command',
    viewDocs: 'View install docs',
    installTo: 'Will install to',
    retryAfterRun: 'I’ve run it -- retry',
    setupChoiceTitle: 'Set up Hermes Desktop',
    setupChoiceDesc:
      'Connect this app to a Hermes gateway you already run, or install Hermes locally on this computer.',
    connectExistingTitle: 'Connect to existing Hermes',
    connectExistingShort: 'Connect existing',
    connectExistingDesc: 'Use a remote backend with a session token or browser sign-in. No local install will start.',
    installLocalTitle: 'Install Hermes locally',
    installLocalDesc: 'Download Hermes, create its Python environment, and run the backend on this computer.',
    localStartUnavailable: 'Local installation could not start. Restart Hermes Desktop and try again.',
    remoteSetupTitle: 'Connect to existing Hermes',
    remoteSetupDesc: 'Enter your gateway URL. Hermes Desktop will detect whether it needs a token or browser sign-in.',
    remoteUrlTitle: 'Gateway URL',
    remoteUrlDesc: 'Use the base URL of the Hermes gateway, including https:// when remote.',
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    probing: 'Detecting gateway authentication...',
    probeError: "Hermes can't reach that address. Check the URL and that the other computer is running Hermes — sign-in options appear once it answers.",
    probeErrorDetails: 'Details',
    identityProvider: 'your identity provider',
    authTitle: 'Authentication',
    authNeedsOauth: provider => `Sign in with ${provider} before testing this gateway.`,
    authSignedIn: 'Browser sign-in completed.',
    connected: 'Connected',
    signIn: 'Sign in',
    signInWith: provider => `Sign in with ${provider}`,
    enterUrlFirst: 'Enter a gateway URL first.',
    signInIncomplete: 'The sign-in window closed before authentication completed.',
    tokenTitle: 'Session token',
    tokenDesc: 'Paste the session token from the remote gateway .env file.',
    pasteSessionToken: 'Paste session token',
    incompleteSignInTest: 'Sign in before testing this OAuth-gated gateway.',
    incompleteTokenTest: 'Enter a session token before testing this gateway.',
    testConnection: 'Test connection',
    testSucceeded: (baseUrl, version) => `Connected to ${baseUrl}${version ? ` (${version})` : ''}.`,
    applyRemote: 'Apply and reconnect',
    backToSetup: 'Back',
    failedTitle: 'Installation failed',
    settingUpTitle: 'Setting up Hermes Agent',
    finishingTitle: 'Finishing up',
    failedDesc:
      'One of the setup steps did not finish. This can happen when another copy of Hermes is running, the internet connection dropped, or antivirus blocked the installer. Close other Hermes windows, then choose Reload and retry. If it fails again, open the logs and send them to support.',
    activeDesc:
      'This is a one-time setup. The Hermes installer is downloading dependencies and configuring your machine. Subsequent launches will skip this step.',
    progress: (completed, total) => `${completed} of ${total} steps complete`,
    currentStage: stage => ` -- now: ${stage}`,
    fetchingManifest: 'Fetching installer manifest...',
    error: 'Error',
    hideOutput: 'Hide installer output',
    showOutput: 'Show installer output',
    lines: count => `${count} line${count === 1 ? '' : 's'}`,
    noOutput: 'No output yet.',
    cancelling: 'Cancelling...',
    cancelInstall: 'Cancel install',
    transcriptSaved: 'Full transcript saved to',
    copiedOutput: 'Copied!',
    copyOutput: 'Copy output',
    reloadRetry: 'Reload and retry',
    openLogs: 'Open logs'
  },

  updates: {
    stages: {
      idle: 'Getting ready…',
      prepare: 'Getting ready…',
      fetch: 'Downloading…',
      pull: 'Almost there…',
      pydeps: 'Finishing up…',
      update: 'Updating Hermes…',
      rebuild: 'Rebuilding the desktop app…',
      restart: 'Restarting Hermes…',
      done: 'Update complete',
      manual: 'Update from your terminal',
      guiSkew: 'Update the desktop app',
      error: 'Update paused'
    },
    checking: 'Looking for updates…',
    checkFailedTitle: 'Couldn’t check for updates',
    tryAgain: 'Try again',
    notAvailableTitle: 'Update not available',
    unsupportedMessage: 'This version of Hermes can’t update itself from inside the app.',
    connectionRetry:
      "Hermes couldn't reach the update server. Check your internet connection and try again. If you use a remote Hermes, make sure it is online.",
    gitUnusable: 'Hermes could not run Git on this computer, so it could not check for updates.',
    connectionSettings: 'Connection settings',
    openDownloadPage: 'Open download page',
    latestBody: 'You’re running the latest version.',
    latestBodyBackend: 'The backend is running the latest version.',
    allSetTitle: 'You’re all set',
    availableTitle: 'New update available',
    availableBody: 'A new version of Hermes is ready to install.',
    availableTitleBackend: 'Backend update available',
    availableBodyBackend: 'A newer version of the connected Hermes backend is ready to install.',
    availableBodyNoChangelog: 'A newer version is ready. Release notes aren’t available for this install type.',
    updateNow: 'Update now',
    maybeLater: 'Maybe later',
    moreChanges: count => `+ ${count} more change${count === 1 ? '' : 's'} included.`,
    manualTitle: 'Update from your terminal',
    manualBody: 'You installed Hermes from the command line, so updates run there too. Paste this into your terminal:',
    manualPickedUp: 'Hermes will pick up the new version next time you launch it.',
    guiSkewTitle: 'Update the desktop app',
    guiSkewBody:
      'The backend was updated, but this desktop app package wasn’t changed. Update or reinstall the Hermes desktop app (your AppImage / .deb / .rpm) to match.',
    copy: 'Copy',
    copied: 'Copied',
    done: 'Done',
    applyingBody:
      'The Hermes updater takes over in its own window and reopens Hermes automatically when it’s done. Please don’t reopen Hermes yourself while it’s updating.',
    applyingBodyBackend:
      'The remote backend is applying the update and will restart. Hermes reconnects automatically when it’s back.',
    applyingClose: 'This window will close while the update runs, then Hermes reopens on its own.',
    errorTitle: 'Update didn’t finish',
    errorBody: 'No worries — nothing was lost. You can try again now.',
    blockerTitle: 'Close local previews to update Hermes?',
    blockerBody:
      'Hermes needs to stop these local previews before updating. This will not modify or delete your files.',
    foreignBlockerTitle: 'Close other processes to update Hermes',
    foreignBlockerBody:
      'Hermes can’t safely close these processes automatically. Close the app, terminal, or service that owns each one, then try the update again.',
    mixedBlockerBody:
      'Hermes can close the local previews listed below. Other processes must be closed manually before the update can continue.',
    closePreviewsAndUpdate: 'Close previews and update',
    closePreviewsAndCheckAgain: 'Close previews and check again',
    localPreview: 'Local preview',
    portLabel: port => `Port ${port}`,
    pidLabel: pid => `PID ${pid}`,
    technicalDetails: 'Technical details',
    notNow: 'Not now',
    clientAlsoBehindTitle: 'Desktop app is behind',
    clientAlsoBehindMessage:
      'The backend is up to date, but this desktop app is still on an older version. Update it to pick up the latest fixes.',
    clientAlsoBehindAction: 'Update desktop app',
    everythingDispatched: 'Update dispatched',
    everythingSkipped: 'Skipped',
    everythingRowFailed: 'Update failed',
    everythingFanoutFailedTitle: 'Couldn’t update other instances',
    applyStatus: {
      preparing: 'Updating backend…',
      pulling: 'Backend updating…',
      restarting: 'Backend restarting to load the update…',
      notAvailable: 'Update not available for this backend.',
      failed: 'Backend update failed.',
      noReturn: 'Backend didn’t come back online. The update may not have completed — check the backend host.'
    }
  },

  handoffTour: {
    profileTitle: 'Your first task runs on the default profile',
    profileText:
      'This rail switches profiles. The one lit up now is default, where the task session lives. The other one is the setup profile, where the welcome chat lives.',
    sessionsTitle: 'Each profile keeps its own sessions',
    sessionsText:
      'This list belongs to the default profile. New session starts one on whichever profile is selected. Switch profiles on the rail and the list changes with it.',
    stayTitle: 'Hermes is one click away',
    stayText: 'Switch to the setup profile and open Welcome to Hermes whenever you want a hand. It stays there.'
  },

  guidedGreeting: {
    line: "Hey, come on in. I'm Hermes. Give me two minutes to set the place up around you, then we'll put me to work on something you actually want done.\n\nFirst though, what should I call you?",
    nameSuggestion: (name: string) => `(I can also just call you ${name}, if you prefer.)`
  },

  remoteDisplayBanner: {
    message: reason =>
      `Software rendering active — remote display detected (${reason}). GPU acceleration is disabled to prevent flickering.`
  },

  freeTier: {
    providerRowTitle: 'Nous · free tier',
    providerRowPitch: 'Sign in with a Nous account to unlock more models and tools.',
    readyTitle: 'Hermes is ready.',
    readyCaption: 'Free · connectors included',
    begin: 'Begin',
    signInInstead: 'Sign in with a Nous account instead',
    otherProviders: 'Other providers',
    stripTitle: 'Free Nous inference and connectors are now available.',
    stripBody: 'Open the model picker to try them, or sign in with a Nous account.',
    openModelPicker: 'Open model picker',
    dismiss: 'Dismiss',
    providerName: 'Nous',
    statusLabel: model => `Nous · ${model}`,
    signIn: 'Sign in',
    signInHeading: 'Sign in with a Nous account to unlock more models and tools.',
    settingUp: 'Setting up free inference…',
    codeBody: 'Enter this code in your browser to finish signing in.',
    copyLink: 'Copy link',
    doNotShare: 'Do not share this code.',
    waiting: 'Waiting for sign-in…',
    finishingHeading: 'Finishing sign-in…',
    finishingBody: 'Approved in the browser. Collecting your account tokens.',
    signedInAs: email => `Signed in as ${email}`,
    signedIn: 'Signed in.',
    completedBody: 'Your account now carries inference and tools.',
    defaultModel: 'Default model',
    change: 'Change',
    done: 'Done',
    notNow: 'Not now',
    tryAgain: 'Try again',
    startAgain: 'Start again',
    didNotComplete: "Sign-in didn't finish",
    rejectedBody: "No problem, you're still on the free Nous service. Sign in whenever you're ready.",
    supersededBody: 'A newer sign-in code replaced this one. Use the newest one, or start again.',
    timedOutHeading: 'That sign-in link has expired',
    timedOutBody: "Start again whenever you're ready. You're still on the free Nous service.",
    retiredBody:
      'Your session ended before the sign-in finished. Hermes will start a new one; then sign in again whenever you\'re ready.',
    errorBody: "Sign-in didn't finish. Try again whenever you're ready.",
    busyHeading: 'Almost there',
    busyBody: wait =>
      `Hermes couldn't finish signing you in because the Nous service is busy. Try again in ${wait}. Your session is still here in the meantime.`,
    unreachableBody:
      "Hermes couldn't reach the Nous service to finish signing you in. Check your internet connection and try again. Your session is still here.",
    alreadySignedInHeading: 'Already signed in.',
    alreadySignedInBody: 'This Hermes is already signed in to a Nous account.',
    setupFailed: {
      gateClosed:
        "This version of Hermes can't start without a Nous account. Sign in or create one, it's free and only takes a minute.",
      paused:
        'Using Hermes without signing in is paused for a moment. Hermes will keep checking. Signing in is free and gets you going right now.',
      rateLimited: wait =>
        `Lots of people are getting started right now, so Hermes will try again in ${wait}. Signing in is free and skips the wait.`,
      unreachable:
        "Hermes couldn't reach the Nous service. Check your internet connection, then tap Try again. Or connect another provider for now.",
      serverError: 'The Nous service had a hiccup. Tap Try again in a moment, or connect another provider for now.',
      powRequired:
        "The Nous server asked for a proof of work, but that isn't implemented in your Agent yet. Sign in or create a free Nous account to continue.",
      locked: "This session can't continue without signing in. Sign in or create a free Nous account to keep going.",
      generic: "Hermes couldn't set up free access without signing in. Signing in is free, or connect another provider.",
      signInBelow: 'Signing in is free. Pick Nous below.',
      tryAgain: 'Try again',
      retrying: 'Trying again…'
    }
  }
} satisfies Pick<
  Translations,
  'boot' | 'onboarding' | 'install' | 'updates' | 'handoffTour' | 'guidedGreeting' | 'remoteDisplayBanner' | 'freeTier'
>
