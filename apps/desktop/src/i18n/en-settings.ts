import { FIELD_DESCRIPTIONS, FIELD_LABELS } from '@/app/settings/constants'

import { settingsRiskCopyEn } from './settings-risk-copy'
import type { Translations } from './types'

export const enSettings: Translations['settings'] = {
  subpages: {
    appearanceTheme: 'Theme',
    appearanceTypography: 'Typography',
    appearanceWindowLayout: 'Window & layout',
    appearanceChatDisplay: 'Chat display',
    appearancePet: 'Pet',
    appearanceGeneral: 'General',
    modelMain: 'Main model',
    modelAuxiliary: 'Auxiliary models',
    modelMoa: 'Mixture of Agents',
    modelFallbacks: 'Fallback models',
    chatBehavior: 'Behavior',
    chatAttachments: 'Attachments',
    workspaceProjects: 'Projects & discovery',
    workspaceShell: 'Shell environment',
    workspaceFiles: 'Files & execution',
    safetyApprovals: 'Approvals',
    safetyPrivacy: 'Privacy & network',
    safetyCheckpoints: 'Checkpoints',
    browserProfile: 'Browser profile',
    browserNetwork: 'Local & private URLs',
    memoryPersistent: 'Persistent memory',
    memoryContext: 'Context & compression',
    voiceConversation: 'Voice conversation',
    voiceTranscription: 'Speech to text',
    voiceSpeech: 'Text to speech',
    advancedRuntime: 'Agent limits',
    advancedTools: 'Tool access',
    advancedTerminal: 'Terminal backend',
    advancedOutput: 'Output limits',
    advancedDelegation: 'Subagents',
    advancedDesktop: 'Desktop & startup',
    gatewayConnection: 'This window',
    gatewayDevices: 'Saved connections',
    gatewayManagedUpdates: 'Remote updates',
    gatewayManagedUpdatesUnavailable: 'Remote updates need a desktop version with managed SSH update support.',
    gatewayManagedUpdatesEmpty: 'Add an SSH connection in Saved connections to manage its updates here.',
    keyboardShortcuts: 'Key bindings',
    hudGesture: 'HUD gesture',
    screenCapture: 'Screen capture',
    notificationAlerts: 'Desktop alerts',
    notificationSounds: 'Sounds',
    archivedSessions: 'Archive & retention',
    defaultDirectory: 'Default project folder',
    vaultCredentials: 'Saved credentials',
    vaultSources: 'Password managers',
    appUpdates: 'Version & updates',
    uninstall: 'Uninstall',
    billingOverview: 'Overview',
    billingPlans: 'Plans'
  },
  closeSettings: 'Close settings',
  exportConfig: 'Export config',
  importConfig: 'Import config',
  resetToDefaults: 'Reset to defaults',
  resetConfirm: 'Reset all settings to Hermes defaults?',
  exportFailed: 'Export failed',
  resetFailed: 'Reset failed',
  nav: {
    providers: 'Providers',
    providerAccounts: 'Accounts',
    providerApiKeys: 'API keys',
    providerCustomEndpoints: 'Custom Endpoints',
    providerLocalModels: 'Local Models',
    gateway: 'Gateways',
    apiKeys: 'Tools & Keys',
    keybinds: 'Keyboard Shortcuts',
    keysTools: 'Tools',
    keysSettings: 'Settings',
    mcp: 'MCP',
    archivedChats: 'Archived Chats',
    sessions: 'Sessions',
    about: 'About',
    billing: 'Billing',
    notifications: 'Notifications',
    vault: 'Passwords & Logins'
  },
  plugins: {
    title: 'Desktop plugins',
    blurb:
      'Extend this app, not an agent — installed once for the whole app, whichever profile, gateway, or machine you connect to. Bundled or dropped into the desktop-plugins folder; toggles apply live.',
    count: n => `${n} installed`,
    openFolder: 'Open Desktop plugins folder',
    rescan: 'Rescan',
    reveal: 'Reveal in file manager',
    enable: 'Enable',
    disable: 'Disable',
    failed: 'failed',
    empty: 'No desktop plugins installed yet.',
    kinds: {
      bundled: 'bundled',
      disk: 'on disk',
      runtime: 'runtime'
    },
    agentHalfMissing: 'agent half missing here',
    agentHalfMissingTip:
      'This is the desktop half of a bundled plugin, but its agent half is not installed on the currently connected backend/profile. Install it from Capabilities → Plugins.',
    installModal: {
      installFromGit: 'Install from Git',
      reviewRepository: 'Review repository',
      repoPlaceholder: 'https://github.com/owner/repo',
      title: 'Install plugin',
      description: 'Review what this repository contains before installing anything.',
      repoLabel: 'Repository',
      includesHeading: 'This package includes',
      agentLabel: 'Agent plugin',
      desktopLabel: 'Desktop UI',
      profileLabel: 'Install for profile',
      agentTargetLocal: (profile, dir) => `Installs into the ${profile} backend (${dir})`,
      agentTargetRemote: profile => `Installs into the connected ${profile} backend`,
      catalogPinned: (name, sha) =>
        `Hermes catalog entry "${name}" — the agent component installs at the reviewed pin${sha ? ` ${sha}` : ''}, not the branch tip.`,
      reviewedHeading: 'Reviewed catalog entry',
      reviewedIntro: 'This entry was human-reviewed at its pinned commit. You can still inspect the exact code below.',
      toolsConnected: n => (n === 1 ? '1 tool connected' : `${n} tools connected`),
      skillsReady: names => (names.length === 1 ? `skill ${names[0]} ready` : `${names.length} skills ready`),
      nextChat: 'more tools available in your next chat',
      serverNotConnected: (server, reason) => `MCP server ${server} is not connected${reason ? `: ${reason}` : '.'}`,
      missingEnvAction: 'Set it up',
      alreadyInstalled: (name: string) => `${name} is already installed.`,
      desktopTarget: "Installs into this app's local desktop-plugins folder",
      desktopTargetFromPackage: 'Loaded into this app from the package above — same for every profile',
      desktopOnlyNote: 'Desktop-only packages do not install a backend agent plugin.',
      insecureWarning: 'This URL uses an insecure or local scheme. Prefer https:// or git@ for production installs.',
      securityHeading: 'Before you install',
      securityIntro:
        'Install only from sources you trust — review the repository below if you want to see what will be added.',
      sourceHeading: 'Source code',
      viewRepository: 'View repository',
      viewPluginFiles: 'View plugin files',
      gitCloneLabel: 'Git clone URL',
      enableAgent: 'Enable agent plugin after install',
      forceReinstall: 'Force reinstall (replace if already installed)',
      pinToCommit: 'Pin to commit (optional)',
      pinToCommitPlaceholder: 'Full 40-character commit SHA',
      pinToCommitHint:
        'Everyone installing this SHA gets the same code; the plugin then refuses updates until re-pinned. Leave empty for the latest commit.',
      pinToCommitInvalid: 'Must be a full 40-character commit SHA (branches and tags are not accepted).',
      install: 'Install',
      installing: 'Installing…',
      probing: 'Inspecting repository…',
      probeUnavailable: 'Plugin inspection is unavailable in this environment.',
      desktopUnavailable: 'Desktop plugin install is unavailable in this environment.',
      selectComponent: 'Select at least one component to install.',
      agentSuccess: name => `Agent plugin ${name} installed`,
      desktopSuccess: name => `Desktop plugin ${name} installed`,
      agentFailed: 'Agent plugin install failed',
      installUncertain:
        'Hermes stopped waiting for the install result, but the plugin may still be installing. Close this dialog and use Rescan in Plugins before trying Install again.',
      desktopFailed: 'Desktop plugin install failed',
      missingEnv: (name, vars) =>
        `${name} is installed but needs a key before it can work: ${vars}. Add it now, or the plugin's tools will fail.`,
      restartToApply: 'Restart the gateway for the plugin to take effect.',
      restartNow: 'Restart gateway'
    },
    sourceTooLarge: 'The plugin file exceeds this app’s read limit. Reduce its size before reloading it.',
    sourcePreviewTruncated:
      'This older app can only read the first 512 KiB of plugin.js. Update Hermes Desktop to load this plugin.',
    loadFailed: name => `Plugin “${name}” failed to load`
  },
  vault: {
    title: 'Passwords & Logins',
    blurb:
      'Say "log into GitHub" and the agent signs in for you. The first time it meets a sign-in page it asks you for the login right there; after that it just works. Passwords are encrypted on this machine and filled straight into the page — the model never sees them.',
    count: n => `${n} saved`,
    loadFailed: 'Could not load vault items',
    empty: 'Nothing saved yet',
    emptyDesc:
      "You don't have to add anything here. Ask the agent to sign into a site and it will ask you for the login once, on the spot. Use Add if you prefer to enter one ahead of time.",
    add: 'Add',
    addTitle: 'Add a login, card or address',
    addDescription: 'Stored encrypted on this machine. The agent never sees the password.',
    added: 'Saved.',
    adding: 'Saving…',
    addConfirm: 'Save',
    kindField: 'Kind',
    kinds: {
      login: 'Login',
      payment: 'Payment card',
      address: 'Address'
    },
    labelField: 'Label',
    labelPlaceholder: 'e.g. GitHub work account',
    labelRequired: 'A label is required.',
    originField: 'Site origin',
    originPlaceholder: 'https://github.com',
    originPlaceholderCheckout: 'https://shop.example.com',
    originInvalid: 'Enter a valid URL like https://example.com.',
    identifierTypeField: 'Identifier type',
    identifierTypes: {
      email: 'Email',
      phone: 'Phone',
      username: 'Username'
    },
    identifierField: 'Identifier',
    identifierShown: identifier => identifier,
    passwordField: 'Password',
    loginFieldsRequired: 'Identifier and password are required.',
    cardNumberField: 'Card number',
    cardNameField: 'Name on card',
    expMonthField: 'Exp. month',
    expYearField: 'Exp. year',
    cvcField: 'CVC',
    postalField: 'Postal code',
    addressLine1Field: 'Address line 1',
    addressLine2Field: 'Address line 2',
    cityField: 'City',
    stateField: 'State / region',
    countryField: 'Country',
    optional: '(optional)',
    createdOn: date => `Added ${date}`,
    deleteAction: 'Remove saved item',
    otpField: 'Authenticator key',
    otpPlaceholder: 'Base32 secret or otpauth:// link',
    otpHint: 'The "setup key" the site shows when you enable 2FA. With it saved, Hermes generates the codes itself.',
    twoFactorBadge: '2FA auto',
    deleteTitle: 'Delete this item?',
    deleteDescription: label => `"${label}" will be removed. This cannot be undone.`,
    deleteConfirm: 'Delete',
    sources: {
      title: 'Password managers',
      blurb:
        'Installed password managers are picked up automatically. The agent asks you to unlock one the first time it needs a login from it (once per session); only a session token stays in memory, and the agent never sees your master password or any login.',
      toggleFailed: 'Could not update password manager',
      notInstalled: name =>
        `Not detected. Install the ${name} command-line tool and sign in to it; Hermes picks it up automatically.`,
      disabledDesc: 'Detected but turned off for Hermes.',
      lockedDesc: 'Detected. The agent will ask you to unlock it when it needs a login, or unlock now.',
      unlockedDesc: 'Unlocked for this session. Locks automatically after 30 minutes idle or when Hermes closes.',
      statusLocked: 'Locked',
      statusNotDetected: 'Not detected',
      statusOff: 'Off',
      statusUnlocked: 'Unlocked',
      unlock: 'Unlock',
      unlocking: 'Unlocking…',
      lock: 'Lock',
      unlocked: name => `${name} unlocked for this session.`,
      unlockTitle: name => `Unlock ${name}`,
      unlockDescription:
        'Enter your master password. It is handed to the password manager on this machine and discarded — it is never stored, logged, or shown to the agent.',
      masterPasswordPlaceholder: 'Master password'
    }
  },
  notifications: {
    title: 'Notifications',
    intro: 'OS notifications (not in-app toasts). Per device.',
    enableAll: 'Enable notifications',
    enableAllDesc: 'Off silences every notification below.',
    focusedHint: 'Completion alerts only fire while Hermes is in the background.',
    kinds: {
      approval: {
        label: 'Approval needed',
        description: 'A command is waiting for you to approve or reject it.'
      },
      input: {
        label: 'Input needed',
        description: 'Hermes asked a question or needs a password or secret.'
      },
      turnDone: {
        label: 'Response ready',
        description: 'A turn finished while Hermes was in the background.'
      },
      turnError: {
        label: 'Turn failed',
        description: 'Background turn errors.'
      },
      backgroundDone: {
        label: 'Background task finished',
        description: 'A backgrounded terminal command completed.'
      },
      credits: {
        label: 'Credit alerts',
        description: 'Credit access is paused or restored.'
      },
      plugin: {
        label: 'Plugin notifications',
        description: 'A desktop plugin sent a notification while Hermes was in the background.'
      }
    },
    test: 'Send test notification',
    testTitle: 'Hermes',
    testBody: 'Notifications are working.',
    testSent: 'Test sent. If nothing appears, check your OS notification permissions and Focus/Do Not Disturb.',
    testUnsupported: 'This system does not support native notifications.',
    completionSoundTitle: 'Completion Sound',
    completionSoundDesc: 'Plays when an agent turn finishes. Pick a preset and preview it here.',
    completionSoundPreview: 'Preview'
  },
  sections: {
    model: 'Model',
    chat: 'Chat',
    appearance: 'Appearance',
    workspace: 'Workspace',
    safety: 'Safety',
    memory: 'Memory & Context',
    voice: 'Voice',
    advanced: 'Advanced',
    browser: 'Browser'
  },
  searchPlaceholder: {
    about: 'About Hermes Desktop',
    config: 'Search settings...',
    gateway: 'Gateway connection...',
    keys: 'Search API keys...',
    mcp: 'Search MCP servers...',
    sessions: 'Search archived sessions...'
  },
  modeOptions: {
    light: {
      label: 'Light',
      description: 'Bright desktop surfaces'
    },
    dark: {
      label: 'Dark',
      description: 'Low-glare workspace'
    },
    system: {
      label: 'System',
      description: 'Follow OS appearance'
    }
  },
  appearance: {
    chatTextScaleTitle: 'Chat Text Size',
    chatTextScaleDesc:
      'Scales conversation text and the message editor relative to UI Scale. Sidebars and controls stay the same size.',
    title: 'Appearance',
    intro: 'Desktop-only. Mode is brightness; theme is palette and chat chrome.',
    colorMode: 'Color Mode',
    colorModeDesc: 'Pick a fixed mode or let Hermes follow your system setting.',
    toolViewTitle: 'Tool Call Display',
    toolViewDesc: 'Product hides raw tool payloads; Technical shows full input/output.',
    hideCodeDiffsTitle: 'Hide code diffs',
    hideCodeDiffsDesc: 'Show file edits as inline tool rows with added/removed line counts, without the code.',
    hideThreadTimelineTitle: 'Hide thread timeline bars',
    hideThreadTimelineDesc: 'Hide the navigation bars along the right edge of each conversation.',
    reasoningCollapsedTitle: 'Collapse thinking by default',
    reasoningCollapsedDesc: 'Keep streamed reasoning available without expanding it until you open it.',
    uiScaleTitle: 'UI Scale',
    uiScaleDesc: (percent: number) =>
      `Scales text and controls across the whole app. Cmd/Ctrl with +, - and 0 also works. Current: ${percent}%.`,
    sessionDensityTitle: 'Session List Density',
    sessionDensityDesc: 'Choose how much context appears beneath session titles in the sidebar.',
    sessionDensityCompact: 'Compact',
    sessionDensityComfortable: 'Comfortable',
    sessionDensityDetailed: 'Detailed',
    tabStripTitle: 'Tab Strip',
    tabStripDesc: 'Show tabs above a zone. Auto hides them for a single pane unless another chat or tile zone is open.',
    tabStripAuto: 'Auto',
    tabStripAlways: 'Always',
    tabStripNever: 'Never',
    appActionsTitle: 'App Actions',
    appActionsDesc: 'Where Settings, Layout, and HUD sit in the titlebar. Right leaves room for tabs on the left.',
    appActionsLeft: 'Left',
    appActionsRight: 'Right',
    terminalFontTitle: 'Terminal Font',
    terminalFontDesc:
      'Choose an installed font for Desktop terminals. Nerd Fonts render Powerlevel10k and shell icons; leave blank to use bundled JetBrains Mono.',
    terminalFontPlaceholder: 'MesloLGS NF or a CSS font stack',
    terminalFontPreview: 'Glyph preview',
    terminalFontReset: 'Use default',
    chatFontTitle: 'Chat Font',
    chatFontDesc:
      "Choose an installed font for chat and the rest of the app. Handy for readability faces such as OpenDyslexic; leave blank to use the theme's font.",
    chatFontPlaceholder: 'OpenDyslexic or a CSS font stack',
    chatFontPreview: 'Preview',
    chatFontSample: 'The quick brown fox jumps over the lazy dog. 0123456789',
    chatFontReset: 'Use theme font',
    translucencyTitle: 'Window Translucency',
    translucencyDesc: 'See your desktop through the whole window, text and all. Tuned separately for light and dark.',
    translucencyGlassDesc:
      'Matte glass: the desktop shows through as a smooth blur while text stays sharp. Tuned separately for light and dark.',
    translucencyModeClear: 'Clear',
    translucencyModeGlass: 'Glass',
    translucencyTintTitle: 'Tint',
    translucencyFadeTitle: 'Fade',
    translucencyFrostTitle: 'Frost',
    translucencyFrost: {
      'under-window': 'Deep',
      popover: 'Soft',
      titlebar: 'Bright',
      header: 'Glare'
    },
    translucencyScopeTitle: 'Area',
    translucencyScope: {
      window: 'Whole window',
      sidebar: 'Sidebar only'
    },
    backdropTitle: 'Chat Backdrop',
    backdropDesc: 'The faint statue image behind the conversation.',
    userBubbleTitle: 'Message Bubble',
    userBubbleDesc: 'How see-through your own messages are. Solid at 0; only the outline remains at 100.',
    textDirectionTitle: 'Text direction',
    textDirectionDesc:
      'How chat messages and the composer choose their direction. Auto follows the first letter of each paragraph; pick a direction when mixed text lines up the wrong way. Code always stays left-to-right.',
    textDirection: { auto: 'Auto', rtl: 'Right-to-left', ltr: 'Left-to-right' },
    introSplashTitle: 'Intro Splash',
    introSplashDesc: 'The wordmark and prompt shown on an empty chat.',
    modelPricingTitle: 'Model Pricing',
    modelPricingDesc: 'Show input, output, and cache-read prices per million tokens in the model picker.',
    reactionsTitle: 'Message Reactions',
    reactionsDesc: 'iMessage-style emoji tapbacks — react to messages, and Hermes can react to yours.',
    tipsTitle: 'In-App Tips',
    tipsDesc:
      'Occasional hints from the app and Hermes. Each tip appears once. Turns off automatically after your first 30 days; you can turn it back on.',
    tipsReset: (count: number) => `Show ${count} ${count === 1 ? 'tip' : 'tips'} again`,
    toursTitle: 'Guided Tours',
    toursDesc:
      'Let Hermes spotlight each step as it guides you through the app. Turns off automatically after your first 30 days; you can turn it back on.',
    composerPopoutTitle: 'Floating Composer',
    composerPopoutDesc: 'Allow dragging the composer out of its dock. When off, it stays docked at the bottom.',
    fileBrowserTitle: 'File Browser',
    fileBrowserDesc:
      'Show the file browser beside the chat when a workspace is open. The titlebar toggle changes this too.',
    vibeHeartsTitle: 'Vibe Hearts',
    vibeHeartsDesc:
      'Floating hearts when you say thanks, ily, good bot, or send a heart. Separate from Message Reactions above.',
    embedsTitle: 'Inline Embeds',
    embedsDesc:
      'Rich previews load from third-party sites (YouTube, X, …). Ask shows a placeholder until you allow each one; Always loads them automatically; Off keeps plain links.',
    embedsAsk: 'Ask',
    embedsAlways: 'Always',
    embedsOff: 'Off',
    embedsReset: (count: number) => `Reset ${count} allowed ${count === 1 ? 'service' : 'services'}`,
    resumeLastSessionTitle: 'Reopen Last Chat on Launch',
    resumeLastSessionDesc:
      'When enabled, the app reopens your most recent chat on cold start. Turn off to always start with a fresh new chat.',
    product: 'Product',
    productDesc: 'Human-friendly tool activity with concise summaries.',
    technical: 'Technical',
    technicalDesc: 'Include raw tool args/results and low-level details.',
    themeTitle: 'Theme',
    themeDesc: 'Desktop palettes only. The selected mode is applied on top.',
    themeSearchPlaceholder: 'Search your themes or the VS Code Marketplace…',
    themeProfileNote: profile => `Saved for the ${profile} profile — each profile keeps its own theme.`,
    installTitle: 'Install from VS Code',
    installDesc:
      'Paste a Marketplace extension id (e.g. dracula-theme.theme-dracula) to convert its color theme into a desktop palette.',
    installPlaceholder: 'publisher.extension',
    installButton: 'Install',
    installing: 'Installing…',
    installError: 'Could not install that theme.',
    installed: name => `Installed “${name}”.`,
    removeTheme: 'Remove theme',
    importedBadge: 'Imported',
    pet: {
      title: 'Pet',
      intro:
        'Adopt an animated petdex mascot that floats over the app and reacts to what Hermes is doing — running while tools execute, celebrating on success, sulking on errors.',
      restartHint:
        'Pets need a quick restart — the running app started before this feature was added. Quit and reopen Hermes, then come back here.',
      scaleTitle: 'Size',
      scaleDesc: 'Resize the floating mascot. Applies everywhere instantly.',
      roamTitle: 'Roam',
      roamDesc: 'Let the pet wander the window on its own while idle.',
      chooseTitle: 'Choose a pet',
      chooseDesc: 'Picking one installs it (if needed) and makes it active.',
      searchPlaceholder: 'Search pets…',
      unreachable: "Couldn't reach the petdex gallery. Check your connection and reopen this page.",
      noMatch: query => `No pets match "${query}".`,
      installedTag: 'installed',
      generatedTag: 'Generated',
      countCapped: (cap, total) => `Showing ${cap} of ${total} — type to narrow it down.`,
      count: n => `${n} pet${n === 1 ? '' : 's'}.`,
      uninstall: name => `Uninstall ${name}`,
      delete: name => `Delete ${name}`,
      deleteTitle: name => `Delete ${name}?`,
      deleteBody: "This permanently deletes the pet — it can't be reinstalled.",
      deleteConfirm: 'Delete',
      rename: name => `Rename ${name}`,
      renameTitle: 'Rename pet',
      renamePlaceholder: 'Name your pet',
      renameSave: 'Save',
      exportPet: name => `Export ${name}`,
      adoptFailed: slug => `Could not adopt ${slug}`,
      uninstallFailed: slug => `Could not uninstall ${slug}`,
      renameFailed: slug => `Could not rename ${slug}`,
      exportFailed: slug => `Could not export ${slug}`,
      noneAvailable: 'No pets available to turn on right now.',
      turnOnFailed: 'Could not turn the pet on.',
      turnOffFailed: 'Could not turn the pet off.',
      on: 'On',
      off: 'Off'
    },
    themeDescriptions: {
      github: 'GitHub Light Default and Dark Default',
      nous: 'GitHub chrome, Nous blue accent',
      catppuccin: 'Soothing pastels — Latte and Mocha',
      everforest: 'Warm, low-contrast forest greens',
      solarized: 'Fixed-contrast light and dark',
      'nous-alt': 'Glass neutrals, cream on mission-blue',
      midnight: 'Deep blue-violet with cool accents',
      ember: 'Warm crimson and bronze — forge vibes',
      mono: 'Clean grayscale — minimal and focused',
      cyberpunk: 'Neon green on black — matrix terminal',
      slate: 'Cool slate blue — focused developer theme'
    },
    themeMarketplace: 'From the VS Code Marketplace',
    noInstalledThemes: query => `No installed themes match "${query}".`
  },
  fieldLabels: FIELD_LABELS,
  fieldDescriptions: FIELD_DESCRIPTIONS,
  uninstallSection: {
    ...settingsRiskCopyEn.uninstallSection,
    dangerZone: 'Danger zone',
    checkingInstalled: 'Checking what’s installed…',
    uninstallHermes: 'Uninstall Hermes',
    chooseHowMuch:
      'Choose how much to remove. The app closes to finish the job; reopen the installer any time to come back.',
    confirmUninstall: 'Confirm uninstall',
    confirmBody: what => `This removes ${what}. This can’t be undone.`,
    appLabel: 'App:',
    couldNotStart: 'Uninstall could not start.',
    uninstalling: 'Uninstalling…',
    yesUninstall: 'Yes, uninstall',
    options: {
      gui: {
        title: 'Uninstall Chat GUI only',
        description: 'Remove this desktop app. The Hermes agent, your config, and chats all stay.',
        consequence: 'the desktop Chat GUI (this app and its data)'
      },
      lite: {
        title: 'Uninstall GUI + agent, keep my data',
        description: 'Remove the app and the Hermes agent, but keep config, chats, and secrets for a future reinstall.',
        consequence: 'the Chat GUI and the Hermes agent (config, chats, and secrets are kept)'
      },
      full: {
        title: 'Uninstall everything',
        description: 'Remove the app, the agent, and all user data — config, chats, scheduled jobs, secrets, logs.',
        consequence: 'EVERYTHING — the Chat GUI, the Hermes agent, and all of your config, chats, secrets, and logs'
      }
    }
  },
  poolLimits: {
    warmBotBackendsAria: 'Warm bot backends',
    warmBotBackendsTitle: 'Warm Bot Backends',
    backendIdleTimeoutAria: 'Backend idle timeout in milliseconds',
    backendIdleTimeoutTitle: 'Backend Idle Timeout'
  },
  customEndpoints: {
    ...settingsRiskCopyEn.customEndpoints,
    active: 'Active',
    apiKeySet: 'API key set',
    use: 'Use',
    editTitle: 'Edit Endpoint',
    addTitle: 'Add Endpoint',
    fields: {
      name: 'Name',
      providerId: 'Provider ID',
      endpointUrl: 'Endpoint URL',
      defaultModel: 'Default Model',
      context: 'Context',
      apiKey: 'API Key',
      apiKeyNewPlaceholder: 'Leave blank to keep current key',
      apiKeyPlaceholder: 'Optional',
      useNewChats: 'Use for new chats',
      discoverModels: 'Discover models'
    },
    test: 'Test',
    save: 'Save',
    newEndpoint: 'New endpoint',
    apiMode: 'API Mode',
    autoDetect: 'Auto-detect',
    couldNotLoad: 'Could not load custom endpoints',
    endpointSaved: 'Custom endpoint saved.',
    saveFailed: 'Save failed',
    endpointReachable: 'Endpoint is reachable.',
    endpointReachableTransport: transport => `Endpoint is reachable (${transport} route served).`,
    endpointReachableModels: (reachable, count) => `${reachable} Found ${count} models.`,
    endpointValidationFailed: 'Endpoint validation failed.',
    validationFailed: 'Validation failed',
    activationFailed: 'Activation failed',
    deleteConfirm: name => `Delete ${name}?`,
    deleteFailed: 'Delete failed',
    title: 'Custom Endpoints',
    deleteEndpoint: 'Delete endpoint',
    emptyDescription: 'Add an OpenAI-compatible endpoint below.',
    emptyTitle: 'No custom endpoints',
    namePlaceholder: 'Axet Proxy',
    contextPlaceholder: 'Auto'
  },
  computerUse: {
    accessibility: 'Accessibility',
    screenRecording: 'Screen Recording',
    driverHealth: 'Driver health'
  },
  about: {
    updates: 'Updates',
    heading: 'Hermes Desktop',
    version: value => `Version ${value}`,
    versionUnavailable: 'Version unavailable',
    bundleOutOfSync: 'App build out of date',
    bundleOutOfSyncDesc:
      'The Hermes runtime was updated, but the desktop app itself is still an older build — new interface features (like Bot Mode) will be missing until it updates. Run the update below to rebuild the app. If that doesn\u2019t clear this warning, reinstall from the latest desktop installer.',
    bundleOutOfSyncAction: 'Get the installer',
    bundleSwapPending: 'Restart to finish the update',
    bundleSwapPendingDesc:
      'The updated app is already installed — Hermes only needs to restart to load it. Chats and settings are untouched.',
    bundleSwapPendingAction: 'Restart Hermes',
    checkNow: 'Check now',
    checking: 'Checking…',
    seeWhatsNew: "See what's new",
    updateNow: 'Update now',
    releaseNotes: 'Release notes',
    onLatest: "You're on the latest version.",
    installing: 'An update is currently installing.',
    cantUpdate: "This build can't update itself from inside the app.",
    cantReach: "We couldn't reach the update server.",
    tapCheck: 'Tap "Check now" to look for updates.',
    updateReady: count => `A new update is ready (${count} change${count === 1 ? '' : 's'} included).`,
    updateReadyUnknown: 'A new update is ready.',
    lastChecked: age => `Last checked ${age}`,
    justNowSuffix: ' · just now',
    automaticUpdates: 'Automatic updates',
    automaticUpdatesDesc:
      'Hermes checks for updates automatically in the background and lets you know when one is ready.',
    branchCommit: (branch, commit) => `Branch ${branch} · Commit ${commit}`,
    never: 'never',
    justNow: 'just now',
    minAgo: count => `${count} min ago`,
    hoursAgo: count => `${count} hours ago`,
    daysAgo: count => `${count} days ago`
  },
  config: {
    minimizeToTrayTitle: 'Minimize to tray',
    minimizeToTrayDesc:
      'Minimize windows or close the main window to hide them in the system tray (menu bar on macOS) and keep Hermes running. Use Quit Hermes from the tray menu or Cmd+Q to exit. Off by default; applies only to this device.',
    minimizeToTrayUnavailable:
      'The system tray is unavailable. Windows will minimize and close normally. Turn this off and on to retry.',
    none: 'None',
    noneParen: '(none)',
    builtinOnly: 'Built-in only',
    notSet: 'Not set',
    commaSeparated: 'comma-separated values',
    searchPlaceholder: 'Search…',
    noResults: 'No results found',
    systemDefault: 'System default',
    loading: 'Loading Hermes configuration...',
    emptyTitle: 'Nothing to configure',
    emptyDesc: 'This section has no adjustable settings.',
    failedLoad: 'Settings failed to load',
    autosaveFailed: 'Autosave failed',
    imported: 'Config imported',
    invalidJson: 'Invalid config JSON',
    toolsetsWipeConfirm:
      'Remove all enabled toolsets? This disables memory, terminal, web search, delegation, and most other tools until you re-enable them.',
    keepAwakeTitle: 'Keep computer awake',
    keepAwakeDesc:
      'Stop this machine from sleeping. "While working" holds it only while a turn is in flight, so overnight runs survive without pinning the laptop awake all week. The display can still dim.',
    keepAwakeOff: 'Off',
    keepAwakeWhileWorking: 'While working',
    keepAwakeAlways: 'Always',
    disableF12Title: 'Disable F12 DevTools',
    disableF12Desc: 'Block F12 from opening Developer Tools. Ctrl+Shift+I (or Cmd+Opt+I on Mac) still works.',
    alwaysExternalLinksTitle: 'Always open links in external browser',
    alwaysExternalLinksDesc:
      'Open every link you click in your system browser instead of the in-app browser. "Open in in-app browser" in the right-click menu still works.',
    attachmentSizeTitle: 'Max preview / image load size',
    attachmentSizeDesc:
      'How big a local file Desktop will load for previews and image attach, in MB. Default is 16. Remote non-image attach uses a separate 256 MB cap. Setting this very high loads the whole file into memory and can freeze or crash the app.',
    attachmentSizeUnit: 'MB',
    attachmentSizeLabel: 'Max preview / image load size in megabytes',
    voiceShortcutHintTitle: 'Voice recording shortcut',
    voiceShortcutHintDesc:
      'Set the voice recording shortcut in Settings → Keyboard Shortcuts ("Start / stop voice conversation"). The voice.record_key config value only applies to the CLI and TUI.',
    showOptions: 'Show options'
  },
  hudModifier: {
    title: 'Tap to summon HUD',
    description:
      'Tap and release ⌘ + Option on Mac, or Ctrl + Alt on Windows/Linux, to bring the HUD forward from any app. Off by default; applies only to this device.',
    permission:
      'Allow Hermes in System Settings → Privacy & Security → Input Monitoring, then retry. This gesture does not record keystrokes or capture your screen.',
    unavailable:
      'The HUD gesture helper could not start or stopped unexpectedly. Retry, or restart Hermes. The existing HUD shortcut still works inside Hermes.',
    missingHelper:
      'This Hermes installation is missing the HUD gesture helper. Update or reinstall Hermes, then retry.',
    unsupportedSession:
      'This desktop session does not support global modifier taps. Linux requires X11; Wayland is not supported.'
  },
  screenshot: {
    enabledTitle: 'Screenshot shortcut',
    enabledDesc:
      'Press both Command keys together from any app to capture its frontmost window and attach it to your current Hermes draft. Never sends automatically. Off by default; applies only to this Mac. Window contents may be sensitive — review the attachment before sending.',
    statusTitle: 'Screenshot shortcut status',
    checking: 'Checking screenshot shortcut…',
    disabled: 'Screenshot shortcut is off.',
    starting: 'Starting the shortcut listener. It is not ready yet.',
    ready: 'Shortcut is ready. Screenshots attach to your current draft without sending.',
    inputPermission:
      'Input Monitoring permission lets Hermes detect both Command keys while another app is active. Allow Hermes in System Settings → Privacy & Security → Input Monitoring, then return here and retry.',
    screenPermission:
      'Screen Recording permission lets Hermes capture the frontmost app window when you use this shortcut. Allow Hermes in System Settings → Privacy & Security → Screen Recording, then return here and retry. Restart Hermes if macOS asks.',
    openSettings: 'Open System Settings',
    retry: 'Retry',
    unavailable: 'The screenshot shortcut is unavailable. Retry, or turn it off.',
    errorTitle: 'Screenshot shortcut error',
    loadFailed: 'Could not read the shortcut status. Retry to check its current setting.',
    saveFailed: 'Could not confirm the shortcut change. Retry to check its current setting.',
    permissionFailed: 'Could not open System Settings. Open Privacy & Security manually, then retry.',
    captureFailed: 'Could not capture the frontmost window. Nothing was attached or sent.',
    contextChanged: 'The current draft changed during capture. The screenshot was not attached or sent.'
  },
  quickEntry: {
    enabledTitle: 'Quick Entry',
    enabledDesc:
      'Summon a small composer from anywhere with a global shortcut and fire a prompt without opening Hermes.',
    shortcutTitle: 'Quick Entry shortcut',
    shortcutDesc: 'Needs at least one modifier, e.g. CommandOrControl+Shift+Space.',
    active: 'Shortcut is active.',
    takenBy: 'Another app already uses this shortcut — pick a different one.',
    invalidShortcut: 'Not a valid shortcut. Include at least one modifier key.'
  },
  credentials: {
    pasteKey: 'Paste key',
    pasteLabelKey: label => `Paste ${label} key`,
    optional: 'Optional',
    enterValueFirst: 'Enter a value first.',
    couldNotSave: 'Could not save credential.',
    remove: 'Remove',
    getKey: 'Get a key',
    saving: 'Saving'
  },
  envActions: {
    actions: 'Actions',
    manageInKeys: 'Manage in API Keys',
    docs: 'Docs',
    hideValue: 'Hide value',
    revealValue: 'Reveal value',
    replace: 'Replace',
    set: 'Set',
    clear: 'Clear'
  },
  connections: {
    title: 'Registered gateways',
    intro: 'Manage this device and every Hermes gateway it can reach through remote, SSH, or Cloud connections.',
    stagedNote:
      'Switch gateways from Sessions. Profiles, chats, messaging, and cron jobs stay with their gateway; work on other gateways keeps running.',
    launchModeTitle: 'At startup, return to Sessions on the last-used gateway',
    launchModeDesc: 'When off, Sessions opens on the Primary gateway.',
    searchPlaceholder: 'Search gateways…',
    noSearchResults: 'No gateways match your search.',
    loadFailed: 'Could not load connections',
    currentPill: 'Current',
    primaryPill: 'Primary',
    managedPill: 'App-managed',
    addConnection: 'Add connection',
    editConnection: 'Edit',
    removeConnection: 'Remove',
    removeConfirmTitle: 'Remove this connection?',
    removeConfirmDesc: (label: string) =>
      `“${label}” will be removed from this app. The instance itself is not touched — you can add it again any time.`,
    makePrimary: 'Make primary',
    testConnection: 'Test',
    testOk: 'Reachable',
    testFailed: 'Connection test failed',
    saveFailed: 'Could not save the connection',
    removeFailed: 'Could not remove the connection',
    updateAll: 'Update all instances',
    updateAllRunning: 'Updating all instances…',
    updateAllDone: 'Updates dispatched',
    updateAllFailed: 'Update fan-out failed',
    updateSkippedCloud: 'Managed by Hermes Cloud',
    kindLocal: 'Local',
    kindRemote: 'Remote gateway',
    kindCloud: 'Hermes Cloud',
    kindSsh: 'SSH',
    kindLocalDesc: 'The Hermes runtime managed by this app.',
    kindRemoteDesc: 'A Hermes gateway reachable over HTTP(S) — LAN, Tailscale, or the internet.',
    kindCloudDesc: 'A hosted instance discovered through your Hermes Cloud account.',
    kindSshDesc: 'A Hermes install reached over SSH.',
    labelTitle: 'Name',
    labelDesc: 'Required. Shown everywhere this instance appears; must be unique (e.g. “Homelab”, “Work laptop”).',
    labelPlaceholder: 'Homelab',
    urlTitle: 'Gateway URL',
    sshHostTitle: 'SSH host',
    headersTitle: 'Extra gateway headers',
    headersDesc:
      'Sent with every HTTP and WebSocket request to this gateway — for access proxies such as Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). Values are stored encrypted. Headers Hermes manages (Authorization, Cookie, Host…) are ignored.',
    headerValuePlaceholder: 'Value',
    headerValueSaved: 'Saved — leave blank to keep',
    headerAdd: 'Add header',
    headerRemove: 'Remove',
    duplicateLocal: 'This app already manages a local connection — there can only be one.',
    duplicateUrl: (label: string) => `A connection to this gateway URL already exists (“${label}”).`,
    duplicateSsh: (label: string) => `A connection to this SSH host already exists (“${label}”).`,
    sameBackendHint: (label: string) => `Same backend as “${label}”`,
    localAddHint: 'Local is unavailable: the managed local connection already exists (there is only ever one).',
    cloudAddHint:
      'Tip: signing in under Hermes Cloud above discovers your agents automatically — use this form only to register a known instance URL by hand.',
    save: 'Save connection',
    saving: 'Saving…',
    cancel: 'Cancel',
    empty: 'No connections registered yet.'
  },
  managedUpdates: {
    title: 'Managed updates',
    intro:
      'Update Desktop-managed SSH installs transactionally: sessions drain, the remote checkout updates, and every profile is restored with a correlated receipt.',
    sshConnection: 'Desktop-managed SSH install',
    update: 'Update',
    updating: 'Updating…',
    progress: 'Draining sessions, updating the remote install, and restoring profiles…',
    updated: 'Updated',
    partial: 'Updated — restore failed',
    refused: 'Refused',
    failed: 'Update failed',
    alreadyRunning: 'Update already in progress',
    receipt: (id: string, outcome: string) => `Receipt ${id} · ${outcome}`,
    receiptVersions: (pre: string, post: string) => `${pre} → ${post}`,
    scopesRestored: (profiles: string) => `Restored profiles: ${profiles}`,
    scopeNotRestored: (profile: string, error: string) => `Profile “${profile}” not restored: ${error}`,
    receiptOutcomes: {
      success: 'Succeeded',
      failed: 'Failed',
      partial: 'Partially completed',
      running: 'In progress',
      refused: 'Refused'
    }
  },
  gateway: {
    loading: 'Loading gateway settings...',
    unavailableTitle: 'Gateway settings unavailable',
    unavailableDesc: 'Connection settings can only be changed from the Hermes Desktop app on the computer running it.',
    title: 'Gateway Connection',
    envOverride: 'env override',
    intro:
      'Local by default. Use remote when this app should drive a Hermes backend elsewhere. Gateway connections are machine-level; profiles are discovered from the gateways you connect.',
    envOverrideTitle: 'This connection was fixed by the way Hermes was launched.',
    envOverrideDesc:
      'A startup setting outside the app chose this connection, so the options below are read-only. Restart Hermes without that setting — or ask whoever set it up — to change it here.',
    modeTitle: 'Connection mode',
    localTitle: 'Local gateway',
    localDesc: 'Start a private Hermes backend on localhost. This is the default and works offline.',
    remoteTitle: 'Remote gateway',
    remoteDesc: 'Connect this desktop shell to a remote Hermes backend.',
    remoteAuthHint: 'Hosted gateways use OAuth or a username and password; self-hosted ones may use a session token.',
    cloudTitle: 'Hermes Cloud',
    cloudDesc: 'Sign in once to Hermes Cloud and pick from the agents on your account — no URL to paste.',
    cloudSignInTitle: 'Hermes Cloud',
    cloudSignIn: 'Sign in to Hermes Cloud',
    cloudSignedIn: 'Signed in to Hermes Cloud',
    cloudNeedsSignIn: 'Sign in to Hermes Cloud to discover the agents on your account.',
    cloudSignedInDesc: 'You are signed in. Pick an agent below; the session refreshes automatically.',
    cloudAgentsTitle: 'Your agents',
    cloudOrgPickerTitle: 'Choose an organization',
    cloudOrgSelect: 'Select',
    cloudOrgChange: 'Change org',
    cloudOrgRole: role => `Role: ${role}`,
    cloudLoadingAgents: 'Loading your agents…',
    cloudNoAgents: {
      before: 'No agents found on this account. Create one in the ',
      linkText: 'Nous portal',
      after: ', then refresh.'
    },
    cloudRefresh: 'Refresh',
    cloudConnect: 'Connect',
    cloudSavedTitle: 'Saved Cloud gateways',
    cloudSavedDesc:
      'Use a saved gateway without changing your default. Sign in below to add instances. Manage names and sign-in in the saved connections list.',
    cloudUseSaved: 'Use gateway',
    cloudActive: 'Active in this window',
    cloudConnecting: 'Connecting…',
    cloudDiscoverFailed: 'Could not load your Hermes Cloud agents',
    cloudConnectFailed: 'Could not connect to that agent',
    cloudSignInFailed: 'Hermes Cloud sign-in failed',
    cloudSignedOutTitle: 'Signed out of Hermes Cloud',
    cloudSignedOutMessage: 'Cleared the Hermes Cloud session.',
    cloudConnectedTitle: 'Connected',
    cloudConnectedPill: 'Connected',
    cloudConnectedTo: name => `Connected to ${name}.`,
    cloudAgentProvisioning: 'Provisioning…',
    cloudStatusLabel: status => `Status: ${status}`,
    remoteUrlTitle: 'Remote URL',
    remoteUrlDesc: 'Base URL for the remote dashboard backend. Path prefixes are supported, for example /hermes.',
    probing: 'Checking how this gateway authenticates…',
    probeError:
      "Hermes can't reach that address. Check the URL and that the other computer is running Hermes — sign-in options appear once it answers.",
    signedIn: 'Signed in',
    signIn: 'Sign in',
    signOut: 'Sign out',
    signInWith: provider => `Sign in with ${provider}`,
    authTitle: 'Authentication',
    authSignedInPassword:
      'This gateway uses a username and password. You are signed in; the session refreshes automatically.',
    authSignedInOauth: 'This gateway uses OAuth. You are signed in; the session refreshes automatically.',
    authNeedsPassword: 'This gateway uses a username and password. Sign in to authorize this desktop app.',
    authNeedsOauth: provider => `This gateway uses OAuth. Sign in with ${provider} to authorize this desktop app.`,
    tokenTitle: 'Session token',
    tokenDesc: 'The dashboard session token used for REST and WebSocket access. Leave blank to keep the saved token.',
    existingToken: value => `Existing token ${value}`,
    savedToken: 'saved',
    pasteSessionToken: 'Paste session token',
    plainTextConfirmTitle: 'Store the gateway token in plain text?',
    plainTextConfirmDesc:
      'No OS keyring service was found on this machine, so the token would be saved unencrypted in the app’s connection settings file, readable by any process running as this user. Install or enable your system keychain (GNOME Keyring or KWallet on Linux) for encrypted storage.',
    plainTextConfirmAction: 'Save as plain text',
    plainTextStoredTitle: 'Token stored in plain text',
    plainTextStoredDesc:
      'Secure storage is unavailable, so the saved token is stored unencrypted in the app’s connection settings file on this machine. Install or enable your system keychain (GNOME Keyring or KWallet on Linux) to encrypt it.',
    keychainEncryptionTitle: 'Encrypt saved secrets with the OS keychain',
    keychainEncryptionDesc:
      'Off by default. When on, gateway tokens and sign-in credentials are encrypted with your system keychain (Keychain Access, GNOME Keyring, or Windows DPAPI) — your system may ask for permission or a password. When off, they are stored as plain files readable only by your user account.',
    keychainEncryptionFailed: 'Could not change secret encryption',
    testRemote: 'Test remote',
    saveForRestart: 'Save for next restart',
    saveAndReconnect: 'Save and reconnect',
    diagnostics: 'Diagnostics',
    diagnosticsDesc: 'Reveal desktop.log in your file manager — useful when the gateway fails to start.',
    openLogs: 'Open logs',
    incompleteTitle: 'Remote gateway incomplete',
    incompleteSignIn: 'Enter a remote URL and sign in before switching to remote.',
    incompleteToken: 'Enter a remote URL and session token before switching to remote.',
    incompleteSignInTest: 'Enter a remote URL and sign in before testing.',
    incompleteTokenTest: 'Enter a remote URL and session token before testing.',
    enterUrlFirst: 'Enter a remote URL first.',
    restartingTitle: 'Gateway connection restarting',
    savedTitle: 'Gateway settings saved',
    restartingMessage: 'Hermes Desktop will reconnect using the saved settings — the shell stays open.',
    savedMessage: 'Saved for the next restart.',
    connectedTo: (baseUrl, version) => `Connected to ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
    reachableTitle: 'Remote gateway reachable',
    signedOutTitle: 'Signed out',
    signedOutMessage: 'Cleared the remote gateway session.',
    failedLoad: 'Gateway settings failed to load',
    signInFailed: 'Sign-in failed',
    signOutFailed: 'Sign-out failed',
    testFailed: 'Remote gateway test failed',
    applyFailed: 'Could not apply gateway settings',
    saveFailed: 'Could not save gateway settings',
    sshTitle: 'Connect via SSH',
    sshDesc:
      'Hermes is launched on the remote over SSH and tunneled to this app — nothing to start or expose yourself. Requires working key-based SSH access to the host.',
    sshTrustHint: 'The first presented host key is trusted and pinned; later changes fail closed.',
    sshHostTitle: 'Host',
    sshHostDesc: 'user@host, or a Host alias from ~/.ssh/config.',
    sshHostPick: 'Select a host…',
    sshHostPickTitle: 'Host',
    sshHostPickDesc: 'A Host alias from ~/.ssh/config, or Custom to type one.',
    sshHostCustom: 'Custom (enter manually)…',
    sshUserTitle: 'User',
    sshUserDesc: 'Blank = ~/.ssh/config or your current user.',
    sshUserPlaceholder: 'from ~/.ssh/config',
    sshPortTitle: 'Port',
    sshPortDesc: 'Blank = 22 or the ~/.ssh/config port.',
    sshKeyTitle: 'Identity file',
    sshKeyDesc: 'Private key path. Blank = ssh-agent or ~/.ssh/config.',
    sshHermesPathTitle: 'Hermes path (optional)',
    sshHermesPathDesc: 'Full path to the remote hermes binary. Blank = auto-detect.',
    sshHermesPathPlaceholder: 'auto-detect',
    sshTestConnection: 'Test SSH',
    sshConnect: 'Connect',
    sshButtonsHint: 'Save applies on the next launch. Connect reconnects now.',
    sshReachable: (host, platform) => `Reachable: ${host} (${platform}) — Hermes found`,
    sshIncompleteHost: 'Enter an SSH host before connecting.',
    sshErrUnreachable: 'Could not reach that host over SSH. Check the host, port, and your network.',
    sshErrAuth:
      'SSH authentication failed. Load your key into the ssh-agent (ssh-add) or set an IdentityFile in ~/.ssh/config — Hermes runs ssh non-interactively.',
    sshErrHostKey:
      'The host key has CHANGED since you last connected. Verify this is expected, then run ssh-keygen -R <host> and reconnect.',
    sshErrNotInstalled:
      'Hermes is not installed on the remote host. Install it there (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) or set the Hermes path.',
    sshErrPlatform:
      'Unsupported remote platform. Hermes Desktop SSH mode supports Linux, macOS, and Windows remote hosts.',
    sshErrTimeout: 'SSH connection timed out. The host may be unreachable or asleep.',
    sshErrUpdateRequired: 'Update Hermes on the remote host before connecting with Desktop SSH.',
    sshErrUnknown: 'SSH connection failed.'
  },
  keys: {
    loading: 'Loading API keys and credentials...',
    failedLoad: 'API keys failed to load',
    empty: 'Nothing configured in this category yet.'
  },
  search: {
    placeholder: 'Search all settings…',
    pill: 'Search'
  },
  profileScope: {
    appliesTo: 'Applies to',
    editsProfile: profile => `Changes on this page apply to the “${profile}” profile.`
  },
  mcp: {
    loading: 'Loading MCP servers...',
    invalidJson: 'Invalid MCP JSON',
    saveFailed: 'Save failed',
    removeFailed: 'Remove failed',
    reloadFailed: 'MCP reload failed',
    savedTitle: 'MCP server saved',
    savedMessage: name => `${name} applies after MCP reload.`,
    disabled: 'disabled',
    name: 'Name',
    serverJson: 'Server JSON',
    remove: 'Remove',
    test: 'Test connection',
    catalogLoading: 'Loading MCP catalog...',
    catalogInstallFailed: name => `Failed to install ${name}`,
    catalogEnvRequired: 'Fill in the required values before installing.',
    capabilitySummary: (tools, prompts, resources) =>
      `${[`${tools} tools`, ...(prompts ? [`${prompts} prompts`] : []), ...(resources ? [`${resources} resources`] : [])].join(', ')} enabled`,
    costTokens: tokens => `~${tokens} tok/call`,
    usage30d: uses => `${uses} uses/30d`,
    statusConnecting: 'Connecting…',
    statusNeedsAuth: 'Needs authentication',
    statusError: 'Error',
    statusOff: 'Off',
    allServers: 'All servers',
    authenticatedTitle: 'Authenticated',
    authenticatedMessage: (server, count) => `${server}: ${count} tools`,
    authenticate: 'Authenticate',
    noOutput: 'No output yet.',
    deepLinkTitle: 'Add MCP server?',
    deepLinkDescription:
      'A link asked to add this MCP server to Hermes. Review the exact configuration below — it comes from the link, not from Hermes.',
    deepLinkStdioWarning:
      'This server runs a local process on your machine with the command shown below. Only continue if you trust its source.',
    deepLinkConfirm: 'Add server',
    deepLinkNameInvalid: 'Names use 1-64 letters, digits, dots, dashes, or underscores.',
    deepLinkNameConflict: name => `A server named ${name} already exists — choose a different name or cancel.`,
    deepLinkErrorTitle: 'MCP install link rejected',
    deepLinkErrorName: 'The link\u2019s server name is missing or invalid.',
    deepLinkErrorConfig: 'The link\u2019s config is not valid base64-encoded JSON.',
    deepLinkErrorShape: 'The config must be a JSON object with a string `url` or `command` field.',
    deepLinkErrorUrl: 'Only http:// and https:// server URLs are allowed.',
    deepLinkErrorTooLarge: 'The config payload exceeds the 32KB limit.',
    failedLoad: 'MCP config failed to load',
    nameRequiredTitle: 'Name required',
    nameRequiredMessage: 'Give this MCP server a config key.',
    objectRequired: 'Server config must be a JSON object',
    gatewayUnavailableTitle: 'Gateway unavailable',
    gatewayUnavailableMessage: 'Reconnect the gateway before reloading MCP.',
    reloadedTitle: 'MCP tools reloaded',
    reloadedMessage: 'New tool schemas apply to fresh turns.',
    newServer: 'New server',
    reload: 'Reload MCP',
    reloading: 'Reloading...',
    emptyTitle: 'No MCP servers',
    emptyDesc: 'Add a stdio or HTTP server to expose MCP tools.',
    editServer: 'Edit server',
    saveServer: 'Save server',
    testing: 'Testing...',
    testOk: count => `Connected — ${count} tool${count === 1 ? '' : 's'} available`,
    testFailed: 'Connection failed',
    enableServer: name => `Enable ${name}`,
    disableServer: name => `Disable ${name}`,
    serverEnabled: name => `${name} enabled — applies to new sessions.`,
    serverDisabled: name => `${name} disabled — applies to new sessions.`,
    toggleFailed: (name, enabled) => `Failed to turn ${name} ${enabled ? 'on' : 'off'}`,
    tabServers: 'Servers',
    tabCatalog: 'Catalog',
    catalogLoadFailed: 'MCP catalog failed to load',
    catalogEmpty: 'No catalog entries available.',
    catalogInstalled: 'Installed',
    catalogEnabled: 'Enabled',
    catalogNeedsInstall: 'Needs build',
    catalogInstall: 'Install',
    catalogInstalling: 'Installing...',
    catalogInstallStarted: name => `Installing ${name}... applies to new sessions when done.`,
    catalogEnvPrompt: name => `${name} requires credentials`,
    unusedPill: 'unused',
    waitingForBrowser: 'Waiting for browser…',
    unsavedConnect: 'Unsaved — save mcp.json to connect.',
    enableTool: tool => `Enable ${tool}`,
    disableTool: tool => `Disable ${tool}`,
    importButton: 'Import',
    importPlaceholder: 'Paste an mcp.json snippet, npx/docker command, claude mcp add line, URL, or Cursor link…',
    importNoMatch: 'No server config recognized in the pasted text.',
    importConfirm: 'Add to mcp.json',
    importConfirmMany: count => `Add ${count} servers to mcp.json`
  },
  model: {
    setupProviderFallback: 'provider',
    setUpProvider: name => `Set up ${name}`,
    staleAuxBefore: (count, names) => `${count} auxiliary task${count === 1 ? '' : 's'} (${names}) still run on `,
    staleAuxAfter: ', not your main model.',
    staleAuxOtherProviders: 'other providers',
    moaEnabled: 'Enabled',
    moaSetDefault: 'Set default',
    moaNewPresetPlaceholder: 'new preset',
    moaAddPreset: 'Add preset',
    customModel: 'Custom model…',
    customModelPlaceholder: 'Model id',
    chooseFromList: 'Choose from list',
    moaDefault: 'Default:',
    moaReferenceToggle: (enabled, index) => `${enabled ? 'Disable' : 'Enable'} reference ${index}`,
    moaReferenceTitle: index => `Reference ${index}`,
    moaAddReference: 'Add reference model',
    loading: 'Loading model configuration...',
    appliesDesc: 'Applies to new sessions. Use the model picker in the composer to hot-swap the active chat.',
    provider: 'Provider',
    model: 'Model',
    applying: 'Applying...',
    mainAppliedTitle: 'Main model updated',
    mainAppliedMessage: model => `New sessions will use ${model}.`,
    defaultsLabel: 'Defaults',
    reasoning: 'Reasoning',
    reasoningOff: 'Off',
    speed: 'Speed',
    speedStandard: 'Standard',
    defaultsFailed: 'Failed to save model defaults',
    loadFailed: 'Could not load models',
    restartRequired: 'This backend is running old code after an update. Restart it to load the new code.',
    restartBackend: 'Restart backend',
    restartingBackend: 'Restarting backend...',
    restartFailed: 'Could not restart the backend',
    auxiliaryTitle: 'Auxiliary models',
    resetAllToMain: 'Reset all to main',
    staleAuxDismiss: "Don't show again",
    auxiliaryDesc: 'Helper tasks run on the main model by default. Assign a dedicated model to any task to override.',
    setToMain: 'Set to main',
    change: 'Change',
    autoUseMain: 'auto · use main model',
    inheritMainEffort: 'inherit · main model effort',
    providerDefault: '(provider default)',
    fallbackAdd: 'Add fallback',
    fallbackEmpty: 'No fallback models — the default model is used unless it fails.',
    notInCatalog: "isn't in this provider's model list — calls may fall back to a backup.",
    moaTitle: 'Mixture of Agents',
    moaPreset: 'Preset',
    moaDescription:
      'Configure named presets that appear as models under the Mixture of Agents provider. The aggregator is the acting model — it runs every step of the tool loop, and almost all of the run’s cost is billed to its provider. References only advise once per user turn by default.',
    moaAggregator: 'Aggregator',
    moaAggregatorBilled: 'acting model · billed for the run',
    moaReferenceHint: 'advises once per turn by default',
    tasks: {
      vision: {
        label: 'Vision',
        hint: 'Image analysis'
      },
      compression: {
        label: 'Compression',
        hint: 'Context compaction'
      },
      skills_hub: {
        label: 'Skills hub',
        hint: 'Skill search'
      },
      approval: {
        label: 'Approval',
        hint: 'Smart auto-approve'
      },
      mcp: {
        label: 'MCP',
        hint: 'MCP tool routing'
      },
      title_generation: {
        label: 'Title gen',
        hint: 'Session titles'
      },
      review: {
        label: 'Review',
        hint: '/review reviewer subagent'
      },
      triage_specifier: { label: 'Triage specifier', hint: 'Kanban spec fleshing' },
      kanban_decomposer: { label: 'Kanban decomposer', hint: 'Task decomposition' },
      profile_describer: { label: 'Profile describer', hint: 'Auto profile descriptions' },
      curator: {
        label: 'Curator',
        hint: 'Skill-usage review'
      }
    }
  },
  localModels: {
    connectionChanged: 'Local models connection changed',
    title: 'Local Models',
    runtimeTitle: 'Local runtime',
    runtimeReady: backend => `Ready · ${backend}`,
    serverRunning: 'Running',
    runtimeInstalled: 'llama.cpp runtime installed',
    runtimeInstalledDetail: (tag, backend) =>
      `Build ${tag}, ${backend} backend. Hermes starts and manages the server for you.`,
    installTitle: 'Install the local runtime',
    installDetail:
      'Downloads the llama.cpp inference engine (a few hundred MB). Models you download run entirely on this machine — no account, nothing leaves your computer.',
    installAction: 'Install runtime',
    installing: 'Installing runtime…',
    installFailed: 'Runtime install failed',
    hardwareTitle: 'This machine',
    hardwareLoading: 'Checking your hardware…',
    vram: label => `${label} GPU memory`,
    ram: label => `${label} RAM`,
    unifiedMemory: 'Unified memory',
    modelsTitle: 'Models',
    recommended: 'Recommended',
    recommendedReason: {
      'best-quality-resident':
        'The highest-quality model that runs entirely on your GPU at full speed. Picks weigh quality against predicted speed on this hardware.',
      'speed-gated-quality':
        'A higher-quality model fits this machine but would respond too slowly on its memory bandwidth — this is the best model that stays fast.',
      'fastest-resident':
        'No model reaches full speed on this hardware; this one comes closest while running entirely in GPU memory.'
    } as Record<string, string>,
    noRecommendationTitle: 'No automatic recommendation for this machine',
    noRecommendationDetail:
      'Automatic setup requires a curated model that fits entirely in GPU or unified memory. You can still choose a model below or browse more models.',
    noRecommendationAction: 'Browse models',
    downloaded: 'Downloaded',
    downloadAction: size => `Download · ${size}`,
    downloadProgress: (done, total) => `${done} of ${total}`,
    downloadStatusRunning: 'Downloading',
    downloadSpeed: rate => `${rate}`,
    downloadEta: time => `~${time} left`,
    downloadEtaSeconds: count => `${count} sec`,
    downloadEtaMinutes: count => `${count} min`,
    downloadEtaHours: (hours, minutes) => (minutes ? `${hours} h ${minutes} min` : `${hours} h`),
    downloadPausedLabel: 'Paused',
    downloadPauseAction: 'Pause',
    downloadResumeAction: 'Resume',
    downloadDoneToast: model => `${model} is ready.`,
    installDoneToast: 'Local runtime installed and ready.',
    quickstartTitle: 'Run a model on this machine',
    quickstartDetail: (model, size) =>
      `One click sets everything up: the local engine, ${model} (${size} download), and your default for new chats. Nothing leaves this computer.`,
    quickstartDetailReady: model =>
      `One click makes ${model} your default for new chats. Everything runs on this machine.`,
    quickstartAction: 'Set up for me',
    quickstartConfigure: 'Let me choose',
    quickstartDoneToast: model => `${model} is set up — new chats run on this machine.`,
    quickstartFailed: 'Local model setup failed',
    quickstartStageEngine: 'Engine',
    quickstartStageModel: 'Model',
    quickstartStageFinish: 'Finish',
    useAction: 'Use',
    activePill: 'Default',
    updateTitle: 'Engine update available',
    updateDetail: (next, current) =>
      `A newer llama.cpp build (${next}) is ready to install — you're on ${current}. Models keep working during the download.`,
    updateAction: 'Update engine',
    updating: 'Updating engine…',
    upToDateTitle: 'Engine up to date',
    upToDateDetail: (tag, backend) => `Running llama.cpp ${tag} (${backend}).`,
    activeDetail: 'New chats use this model — it loads when you send your first message',
    activeNotLoaded: 'Loads on your first message',
    loadedPill: 'In memory',
    placementResident: 'all on GPU',
    placementSpilled: 'partly in RAM',
    placementResidentTip: 'Running entirely in GPU memory at this context window — full speed.',
    placementSpilledTip:
      'Part of this model runs from system RAM — it works, but slower. A more compact build or a smaller context would fit fully.',
    loadingPill: 'Loading…',
    ejectTip: 'Free GPU memory (loads again on the next message)',
    ejected: 'Model unloaded — GPU memory freed.',
    ejectFailed: 'Could not unload the model',
    stopServer: 'Turn off',
    startServer: 'Turn on',
    runtimeRunningDetail:
      'The local server is running. Turning it off frees all GPU memory and stops new chats from using local models until you turn it back on.',
    serverStopped: 'Local server stopped — GPU memory freed.',
    serverStarted: 'Local server running.',
    serverStopFailed: 'Could not stop the local server',
    serverStartFailed: 'Could not start the local server',
    activating: 'Starting…',
    activateFailed: model => `Could not switch to ${model}`,
    activateDoneToast: model => `New chats use ${model}.`,
    downloadFailed: model => `Download of ${model} failed`,
    downloadPauseFailed: model => `Couldn’t pause the download of ${model}`,
    downloadResumeFailed: model => `Couldn’t resume the download of ${model}`,
    pillFitsGpu: 'Fits your GPU',
    pillUsesRam: 'Uses system RAM',
    pillTooBig: 'Too big for this machine',
    browseTitle: 'Find more models',
    browseHint:
      'Search all of Hugging Face. Models you download here are sized to your machine automatically, but not tested by us.',
    browsePlaceholder: 'Search models by name or author…',
    browseSearching: 'Searching Hugging Face',
    browseListing: 'Reading model files',
    browseShowFiles: 'Show files',
    browseRefresh: 'Refresh',
    browseDownloads: 'downloads',
    browseLikes: 'likes',
    browseGated: 'requires Hugging Face sign-in',
    browseNoGguf: 'No compatible model files found.',
    browseFitUnknown: 'Fit unknown',
    browseAlreadyDownloaded: 'Already downloaded.',
    addedByYou: 'Added by you',
    browseDownloadStarted: 'Downloading {name}',
    browseDownloadAria: 'Download {name}',
    sideloadButton: 'Add model file',
    sideloadTitle: 'Choose a GGUF model file',
    sideloadDone: 'Added {name}.',
    sideloadAlreadyPresent: 'Already in your library.',
    pillFullContext: max => `Full ${max} context`,
    pillFullContextTip: "Runs at the model's complete context window from the start",
    pillUpTo: max => `Up to ${max} context`,
    pillGrowsTip: 'Grows automatically as your conversation needs more room',
    pillVision: 'Sees images',
    deleteAction: 'Delete model',
    deleteConfirm: model => `Delete ${model} from disk?`,
    deleted: model => `${model} deleted.`,
    deleteFailed: 'Delete failed',
    updateToast: next => `A newer local engine build (${next}) is available. Update from Settings → Local Models.`
  },
  billing: {
    perMonth: amount => `${amount}/mo`,
    creditsPerMonth: amount => `${amount} credits/mo`,
    usageLabel: label => `${label} usage`,
    freeTier: {
      signIn: 'Sign in',
      title: "You're on the Nous free tier",
      message: 'Sign in with a Nous account to unlock more models and tools.',
      caption:
        'Runs on nous/welcome with connectors included. Signing in keeps your connectors and adds the tools that need an account and every other model.',
      name: 'Nous · free tier',
      footnote:
        'The free tier has no balance and nothing to pay. Payment and usage appear when you sign in with a Nous account.',
      plan: 'Free tier',
      model: 'Model',
      connectors: 'Connectors',
      included: 'Included'
    },
    amountValidation: {
      reloadTo: 'Reload-to',
      greaterThanThreshold: 'Reload-to amount must be greater than the threshold.',
      decimal: label => `${label}: enter a dollar amount with at most 2 decimal places.`,
      positive: label => `${label}: amount must be greater than $0.`,
      minimum: (label, amount) => `${label}: minimum is ${amount}.`,
      maximum: (label, amount) => `${label}: maximum is ${amount}.`
    },
    stepUp: {
      openVerification: 'Open verification page',
      dismiss: 'Dismiss',
      waiting: 'Waiting for verification link…',
      verify: 'Verify to continue',
      deniedTitle: 'Verification was not approved',
      deniedBody: 'Verification finished without allowing Remote Spending for this terminal.',
      successTitle: 'Verification complete',
      successBody: 'Remote Spending is allowed for this terminal.'
    },
    charge: {
      added: amount => (amount ? `$${amount} added.` : 'Credits added.'),
      failedTitle: 'Charge failed',
      unconfirmedTitle: 'Charge outcome unconfirmed',
      unconfirmedBody: message =>
        `${message} Your last charge's outcome is unconfirmed - check your balance/history before retrying.`,
      checkTitle: 'Could not check charge',
      checkBody: 'Could not check the charge.',
      untrackedTitle: 'Charge could not be tracked',
      untrackedBody: 'The billing service accepted the request but did not return a charge id.',
      timeoutTitle: 'Still processing after 5 minutes',
      timeoutBody: 'Charge may still settle. Check the portal before retrying.',
      authenticationRequired:
        'Your bank requires verification (3DS). Complete it on the portal to finish this purchase.',
      expired: 'Your card has expired. Update it on the portal.',
      declined: 'Your card was declined. Try another card on the portal.',
      failedBody: reason => `The charge didn't go through (${reason}).`
    },
    title: 'Billing',
    preview: 'preview',
    summary: {
      balance: 'Balance',
      plan: 'Plan',
      autoRefill: 'Auto-refill'
    },
    sections: {
      invoices: 'Invoices',

      plan: 'Plan',
      paymentAndCredits: 'Payment & credits',
      usage: 'Usage'
    },
    usage: {
      title: 'Usage'
    },
    buyCredits: {
      customAmount: 'Custom credit amount',
      title: 'Buy credits now',
      buyButton: 'Buy',
      processing: 'Processing… checking settlement',
      added: amount => `${amount} added. Balance is refreshing.`,
      retry: 'Retry',
      openPortal: 'Open portal'
    },
    plan: {
      title: 'Plans',
      changePlan: 'Change plan',
      viewPlans: 'View plans',
      backAria: 'Back to billing',
      current: 'Current plan',
      scheduled: 'Scheduled',
      empty: 'No plans are available to change to right now.',
      undo: 'Undo',
      undoing: 'Undoing…',
      downgrade: 'Downgrade',
      confirmDowngrade: 'Confirm downgrade',
      tryAgain: 'Try again',
      checkingChange: 'Checking this change…',
      cannotChange: 'That change cannot be made here.',
      alreadyOn: name => `You are already on ${name} — nothing to change.`,
      notScheduleable: 'This change cannot be scheduled here.',
      scheduling: 'Scheduling…',
      cancel: 'Cancel',
      effectScheduled: (targetName, effectiveAt, creditsDelta) =>
        `Change to ${targetName} — takes effect ${effectiveAt}. No charge now; you keep your current plan until then.${creditsDelta ? ` Monthly credits change: ${creditsDelta}.` : ''}`
    },
    autoReload: {
      threshold: 'Threshold',
      thresholdAria: 'Auto-refill threshold',
      reloadTo: 'Reload to',
      reloadToAria: 'Auto-refill reload-to amount',
      turnOffConfirm: 'Turn off auto-refill?',
      turnOff: 'Turn off',
      disable: 'Disable',
      updated: 'Auto-refill updated.',
      turnedOff: 'Auto-refill turned off.',
      manage: 'Manage',
      save: 'Save',
      saving: 'Saving…',
      cancel: 'Cancel'
    },
    state: {
      notice: {
        loggedOut: {
          title: 'Connect your Nous account',
          message: 'Sign in with your Nous account to see your balance, plan and usage here.',
          action: 'Sign in'
        },
        openPortal: 'Open portal ↗',
        noCard: {
          title: 'No payment method on file',
          message:
            'Buying top-up credits and auto-refill stay disabled until a card is on file. Add one on the portal.',
          action: 'Add card ↗'
        }
      },
      paymentMethod: {
        title: 'Payment method',
        description: 'Manage the card used for top-ups and subscription renewals.',
        addAction: 'Add payment method',
        updateAction: 'Update',
        provenance: {
          autoRefill: 'auto-refill card',
          customerDefault: 'customer default',
          subPin: 'subscription card',
          suffix: label => ` - ${label}`
        }
      },
      buyCredits: {
        description: 'A single charge on your card, added to your balance today.'
      },
      autoRefill: {
        title: 'Refill when low',
        genericDescription: 'Keep your balance topped up when it drops below your threshold.',
        offPill: 'Off',
        enabledPill: 'Enabled',
        notAvailablePill: '—',
        manageCaption: 'Manage auto-refill from the portal.',
        turnOnCaption: 'Turn on auto-refill from the portal',
        chargesDescription: (reloadTo, threshold) =>
          `Charges ${reloadTo} automatically when your balance falls below ${threshold}.`,
        distinctCardCaption: cardLabel => `Auto-refill charges ${cardLabel} — reconcile on the portal`,
        distinctCardFallback: 'a different card',
        reconcileAction: 'Reconcile ↗'
      },
      usage: {
        subscriptionCredits: {
          title: 'Subscription credits',
          barLabel: 'Subscription credits remaining',
          captionResets: date => `Resets ${date}`,
          valueOf: (remaining, monthly) => `${remaining} of ${monthly} left`,
          valueOver: (remaining, monthly, over) => `${remaining} of ${monthly} left · ${over} over`
        },
        topupCredits: {
          title: 'Top-up credits',
          caption: 'Does not expire'
        },
        monthlyCap: {
          title: 'Monthly spend cap',
          barLabel: 'Monthly spend cap used',
          captionDefault: 'Default ceiling',
          captionSpending: 'Monthly remote spending',
          valueUsed: (spent, limit) => `${spent} of ${limit} used`
        }
      },
      planCard: {
        freeTier: 'Free',
        chooseAction: 'Choose ↗',
        adjustPlanAction: 'Adjust plan ↗',
        unavailableCaption: 'Subscription details are unavailable; opening the portal is still available.',
        downgradeCaption: (tierName, when) => `Changes to ${tierName} on ${when}.`,
        cancellationCaption: when => `Cancels on ${when}.`,
        renewsCaption: date => `Renews ${date}`,
        noSubscriptionCaption: 'No active subscription — paid models draw down top-up credits.'
      }
    },
    errors: {
      consentRequired: {
        title: 'Card confirmation needed',
        message: 'Confirm this card for terminal charges in the portal'
      },
      insufficientScope: {
        title: 'Remote Spending needs approval',
        message: 'This needs Remote Spending allowed. Start a top-up to allow it, then retry.'
      },
      remoteSpendingRevoked: {
        title: 'Remote spending was stopped',
        messageByAdmin: 'An admin stopped remote spending for this terminal.',
        messageBySelf: 'You stopped remote spending for this terminal.'
      },
      remoteSpendingReconnect: who => `${who} Reconnect from Settings -> Gateway to re-authorize this device.`,
      sessionRevoked: {
        title: 'Session logged out',
        message: 'Your session was logged out. Sign in again from Settings → Gateway.'
      },
      cliBillingDisabled: {
        title: 'Remote spending is off',
        message:
          "Remote spending is off for this account — a billing admin can turn it on from the portal's Hermes Agent page."
      },
      roleRequired: {
        title: 'Admin role required',
        message: 'Adding funds needs an org admin/owner. Ask an admin, or manage on the portal.'
      },
      idempotencyConflict: {
        title: 'Start a fresh top-up',
        message: '🔴 That charge key was already used for a different amount. Start a fresh top-up.'
      },
      noPaymentMethod: {
        title: 'No saved card',
        message:
          '💳 No saved card for terminal charges yet. Set one up on the portal ' +
          "(one-time credit buys don't save a reusable card)."
      },
      orgAccessDenied: {
        title: 'Org access denied',
        message: "This token isn't bound to an org you can manage"
      },
      monthlyCapExceeded: {
        title: 'Monthly spend cap reached',
        messageReached: '🔴 Monthly spend cap reached.',
        messageHeadroom: remaining => `🔴 Monthly spend cap reached — $${remaining} headroom left.`
      },
      rateLimited: {
        title: 'Too many charges right now',
        message: mins =>
          mins > 0
            ? `🟡 Too many charges right now (try again in ~${mins} min). This isn't a payment failure.`
            : "🟡 Too many charges right now. This isn't a payment failure."
      },
      stripeUnavailable: {
        title: 'Stripe is having trouble',
        message: mins =>
          mins > 0
            ? `Stripe is having trouble — try again in ~${mins} min`
            : 'Stripe is having trouble — try again shortly'
      },
      upgradeCapExceeded: {
        title: 'Daily plan-change limit reached',
        message: 'Daily plan-change limit reached — try again tomorrow'
      },
      endpointUnavailable: {
        title: 'Billing endpoint unavailable',
        message: 'Billing endpoint returned a non-JSON response (it may not be available on this deployment).'
      },
      timeout: {
        title: 'Billing request timed out',
        message: 'Billing request timed out.'
      },
      transport: {
        title: 'Billing connection failed',
        message: 'Billing request failed before reaching the gateway.'
      },
      default: {
        title: 'Billing request failed',
        message: 'Billing request failed.'
      }
    }
  },
  providers: {
    connectAccount: 'Connect an account',
    haveApiKey: 'Have an API key instead?',
    intro:
      'Sign in with a subscription — no API key to copy. Hermes runs the browser sign-in for you, right here in the app.',
    connected: 'Connected',
    collapse: 'Collapse',
    connectAnother: 'Connect another provider',
    otherProviders: 'Other providers',
    disconnect: 'Disconnect',
    disconnectInTerminal: 'Disconnect (runs the removal command in the terminal)',
    removeConfirm: provider => `Remove ${provider}?`,
    removeExternalGeneric: provider => `${provider} is managed by its own CLI — remove it there.`,
    removeKeyManaged: provider => `${provider} is configured from an API key. Remove it from API Keys.`,
    removeTerminalConfirm: (provider, command) =>
      `Disconnect ${provider}? This runs "${command}" in the terminal to clear the credential.`,
    removeTerminalRunning: provider => `Running ${provider} disconnect in the terminal…`,
    removedTitle: 'Account removed',
    removedMessage: provider => `${provider} was removed.`,
    failedRemove: provider => `Could not remove ${provider}`,
    noProviderKeys: 'No provider API keys available.',
    searchKeys: 'Search providers…',
    noKeysMatch: 'No providers match your search.',
    localEndpoint: {
      title: 'Local / custom endpoint',
      description: 'Point Hermes at any OpenAI-compatible endpoint (Zyphra, vLLM, llama.cpp, Ollama, etc).'
    },
    loading: 'Loading providers...'
  },
  sessions: {
    loading: 'Loading archived sessions…',
    archivedTitle: 'Archived sessions',
    archivedIntro:
      'Archived chats are hidden from the sidebar but keep all their messages. Alt/⌥+Shift-click a chat in the sidebar to archive it.',
    emptyArchivedTitle: 'Nothing archived',
    emptyArchivedDesc: 'Archive a chat to hide it here.',
    unarchive: 'Unarchive',
    deletePermanently: 'Delete permanently',
    messages: count => `${count} ${count === 1 ? 'message' : 'messages'}`,
    restored: 'Restored',
    deleteConfirm: title => `Permanently delete "${title}"? This cannot be undone.`,
    autoArchiveTitle: 'Auto-archive stale chats',
    autoArchiveDesc:
      "Automatically archive chats you haven't touched in a while. Pinned chats are never archived, and nothing is deleted — archived chats just move here.",
    autoArchiveDaysLabel: 'Archive after',
    autoArchiveDaysUnit: 'days of inactivity',
    autoArchiveFailed: 'Could not update auto-archive',
    defaultDirTitle: 'Default project directory',
    defaultDirDesc:
      'New sessions start in this folder unless you pick another. Leave it unset to use your home directory.',
    defaultDirUpdated: 'Default project directory updated — start a new chat (Ctrl/⌘+N) for it to take effect',
    defaultsTo: label => `Defaults to ${label}.`,
    change: 'Change',
    choose: 'Choose',
    clear: 'Clear',
    notSet: 'Not set',
    failedLoad: 'Could not load archived sessions',
    unarchiveFailed: 'Unarchive failed',
    deleteFailed: 'Delete failed',
    updateDirFailed: 'Could not update default directory',
    clearDirFailed: 'Could not clear default directory'
  },
  toolsets: {
    loadingConfig: 'Loading configuration',
    savedTitle: 'Credential saved',
    savedMessage: key => `${key} updated.`,
    removedTitle: 'Credential removed',
    removedMessage: key => `${key} removed.`,
    failedSave: key => `Failed to save ${key}`,
    failedRemove: key => `Failed to remove ${key}`,
    failedReveal: key => `Failed to reveal ${key}`,
    removeConfirm: key => `Remove ${key} from .env?`,
    set: 'Set',
    notSet: 'Not set',
    selectedTitle: 'Provider selected',
    selectedMessage: provider => `${provider} is now active.`,
    failedSelect: provider => `Failed to select ${provider}`,
    failedLoad: 'Tool configuration failed to load',
    noProviderOptions: 'This toolset has no provider options — enable it and it works with your current setup.',
    noProviders: 'No providers are available for this toolset right now.',
    ready: 'Ready',
    needsSignIn: 'Needs sign-in',
    needsSetup: 'Setup required',
    activeBackend: 'Active',
    activeBackendHint: 'This is your active backend',
    useBackend: 'Use this backend',
    nousIncluded: 'Included with a Nous subscription — sign in with your Nous account to activate.',
    nousAuthNeededTitle: 'Sign in with your Nous account',
    nousAuthNeededMessage: provider =>
      `${provider} is saved but will only work once you sign in with your Nous account.`,
    nousAuthSignIn: 'Sign in',
    nousAuthDoneTitle: 'Nous account connected',
    nousAuthDoneMessage: 'Your subscription backends are now active.',
    nousAuthFailed: 'Nous sign-in did not complete',
    nousAuthFailedMessage: 'Try again.',
    nousAuthTryAgain: 'Try again',
    noApiKeyRequired: 'No API key required.',
    postSetupHint: step =>
      `This backend needs a one-time install (${step}). Runs on this machine — may take a few minutes.`,
    postSetupInstalledHint: 'Installed. Re-run setup only if something is broken.',
    postSetupRun: 'Run setup',
    postSetupRerun: 'Re-run setup',
    postSetupInstalled: 'Installed',
    postSetupRunning: 'Installing…',
    postSetupStarting: 'Starting…',
    postSetupCompleteTitle: 'Setup complete',
    postSetupCompleteMessage: step => `${step} installed.`,
    postSetupErrorTitle: 'Setup finished with errors',
    postSetupErrorMessage: step => `Setting up ${step} did not finish. Open the logs to see why, then run setup again.`,
    postSetupOpenLogs: 'Open logs',
    postSetupRunAgain: 'Run again',
    postSetupFailed: step => `Failed to run ${step} setup`,
    webSearchActive: backend => `Search: ${backend}`,
    webExtractActive: backend => `Extract: ${backend}`,
    webCapabilityUnset: 'not set',
    webUseForSearch: 'Use for Search',
    webUseForExtract: 'Use for Extract',
    webUsedForSearch: 'Search backend',
    webUsedForExtract: 'Extract backend',
    webCapabilitySelectedMessage: (provider, capability) => `${provider} now handles web ${capability}.`,
    failedSelectCapability: provider => `Failed to set ${provider}`,
    loadingModels: 'Loading model catalog...',
    modelSectionTitle: 'Model',
    modelCount: count => `${count} model${count === 1 ? '' : 's'}`,
    modelInUse: 'In use',
    modelDefault: 'default',
    modelInactiveHint: 'Select this backend first to change its model.',
    modelSelectedTitle: 'Model selected',
    modelSelectedMessage: model => `${model} applies to new sessions.`,
    failedSelectModel: model => `Failed to select ${model}`,
    terminalBackend: {
      sectionTitle: 'Execution backend',
      loading: 'Checking execution backends…',
      failedLoad: 'Could not load terminal backends',
      ready: 'Ready',
      needsSetup: 'Needs setup',
      unavailable: 'Unavailable',
      inUse: 'In use',
      selectedTitle: 'Backend selected',
      selectedMessage: backend => `Terminal commands now run via ${backend}. Applies to new sessions.`,
      failedSelect: backend => `Failed to select ${backend}`,
      needsSetupHint:
        'This backend is currently selected without full setup — commands will fail until setup is complete.',
      needsSetupConfirmTitle: backend => `Select ${backend} anyway?`,
      needsSetupConfirmDescription: detail =>
        `${detail} Sessions that start after this change will have no terminal or file tools until setup is finished.`,
      needsSetupConfirmDescriptionGeneric:
        "This backend isn't set up yet. Sessions that start after this change will have no terminal or file tools until setup is finished.",
      needsSetupConfirmAction: 'Select anyway',
      unavailableTitle: 'Terminal commands are unavailable',
      unavailableMessage: backend =>
        `Hermes can't run shell commands right now: ${backend} isn't ready. Switch to Local, or finish setting up ${backend} and try again.`,
      openBackendSettings: 'Open terminal settings',
      useLocal: 'Use Local',
      switchedToLocal: 'Terminal commands now run locally. Applies to new sessions.'
    },
    browserRealProfile: {
      label: 'Use My Real Browser Profile',
      description:
        "Copies your default browser's logins and cookies into a managed snapshot the agent browses with. Your live profile is never opened directly. Applies to new sessions.",
      enabledTitle: 'Real-profile browsing on',
      enabledMessage: 'New sessions will browse with a snapshot of your default browser profile.',
      disabledTitle: 'Real-profile browsing off',
      disabledMessage: 'The profile snapshot will be deleted; new sessions use a clean browser.',
      failedSave: 'Could not save the real-profile setting',
      prompt: {
        title: 'Stay signed in to your sites',
        body: 'Let Hermes browse with a snapshot of your default browser profile, so sites open already signed in.',
        bulletSnapshot: 'Cookies and logins are copied into a managed snapshot.',
        bulletLiveProfile: 'Your live browser profile is never opened directly.',
        bulletLocal: 'Nothing leaves this computer.',
        dontShowAgain: "Don't show again",
        notNow: 'Not now',
        enable: 'Use my profile'
      }
    }
  }
}
