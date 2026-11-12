import type { RedesignOverlay } from './types'

// English wording for the redesigned shell: only the keys whose text differs from the base catalog.
export const enRedesign: RedesignOverlay = {
  sharedMetrics: {
    collectedMilestones: 'Setup: which steps finished, provider connections, how many skills, plugins and routines',
    sending:
      'If you share, Hermes sends stats to Nous once a day with a random ID for this agent. Apart from a one-time note that Hermes was installed, stats from before you opt in are never sent. Change this anytime in Settings.',
    sendDesc:
      'Counts only, sent once a day with a random ID for this agent. Never your messages, files, paths or error text.',
    consentBody:
      'Hermes can count how you use it: chat length, which models and tools run, and when something fails. It never records your messages, files, paths or error text.',
    collectedActivity: 'Chats: length, outcome, error type, active time per day'
  },
  appTour: {
    capabilities: {
      text: 'Skills, tools and plugins your agent can use. Add more here.'
    },
    messaging: {
      text: 'Reach your agent from Telegram, Slack, Discord and more.'
    },
    newSession: {
      text: 'A new chat gets its own context. Use one per task.'
    }
  },
  firstRun: {
    intro: {
      greeting: "Hi, I'm your agent."
    },
    guide: {
      upgradeRequired:
        'Hermes on this machine needs an update for chatless setup. On that machine, run hermes update, restart Hermes, then retry. Your chats are unchanged.'
    }
  },
  connectors: {
    disclaimer: 'Connecting is optional. Only authorize the apps you want the agent to use.',
    namedAccount: (app: string, alias: string) => `The agent named this ${app} account ${alias}.`,
    unavailable: 'Connectors are unavailable for this chat.'
  },
  catalog: {
    toolsets: {
      mostUsedBlurb: 'The toolsets this agent uses most.',
      essentialsBlurb: 'Tools for everyday agent tasks.',
      themes: {
        recall: {
          blurb: 'What the agent remembers, and the skills and history it can search.'
        },
        more: {
          blurb: 'Everything else this agent can switch on.'
        }
      }
    },
    installDescription: 'This skill will be available in new chats. Only install sources you trust.',
    installHint: 'Review the source before installing. Changes apply to new chats.'
  },
  connectorsPage: {
    page: {
      noMatchBody: 'Nothing here matches. Add your own MCP server to give the agent more tools.'
    },
    dialog: {
      disconnectBody:
        'This disconnects the account for all agents using the same Nous account, not just this agent. You can connect again at any time.',
      appSwitch: (name: string) => `The agent can use ${name}`,
      bothOn: (name: string) => `Both are on, so the agent sees every ${name} tool twice.`,
      nousLine: 'Nous apps follow your account, not the agent.',
      rulesSignIn: 'Sign in to change what the agent may do here.'
    },
    accounts: {
      removeBody:
        'This removes the connected account for all agents using the same Nous account, not just this agent. Your other accounts stay connected.'
    },
    tools: {
      summaryTitle: (name: string) => `What the agent may do with ${name}`,
      summaryPreviewTitle: (name: string) => `What the agent could do with ${name} once you connect`,
      goneBody: 'The agent cannot call it any more. The row stays until you remove it, so nothing vanishes.'
    }
  },
  common: {
    bots: 'Agents'
  },
  boot: {
    steps: {
      connectingGateway: 'Connecting to the machine',
      loadingSessions: 'Loading recent chats'
    },
    failure: {
      remoteTitle: 'Remote machine sign-in required',
      remoteDescription:
        'Your remote machine session has expired. Sign in again to reconnect. Nothing here deletes your chats or settings.',
      useLocalGateway: 'Use local machine',
      gatewaySettings: 'Machine settings',
      remoteSignInHint: signInLabel =>
        `Signs out of the saved remote browser session, then opens ${signInLabel}. Use this machine to switch to the bundled backend instead.`,
      remoteFailureHint: 'Check the machine URL and sign-in under Machine settings, or switch to the local machine.',
      cloudDownDescription:
        'The Nous-managed cloud agent you are connected to is returning a server error. It cannot be restarted from here — check its status, switch to the local machine, or get support.',
      signedInMessage: 'Reconnecting to the remote machine…',
      signInToRemoteGateway: 'Sign in to remote machine'
    }
  },
  notifications: {
    sharedProfileWarning:
      'Another Hermes installation is using this agent. Both installations share its settings and data, so changes can conflict. You can continue, or close the other installation before making changes.',
    errors: {
      gatewayAuthFailed:
        'This Hermes no longer accepts your saved sign-in. Open Machines and sign in again (or paste a new access token), then retry.'
    },
    actions: {
      openGateways: 'Open Machines'
    },
    voice: {
      liveDelegationFailed: 'Could not hand the request to the agent',
      couldNotStartSession: 'Could not start voice chat',
      liveEnded: 'Live voice chat ended',
      liveEndedConnectionLost: 'The live voice chat lost its connection.',
      liveEndedClosed: 'The live voice chat was closed by the service.'
    },
    native: {
      inputBody: 'The agent is waiting for your response.',
      turnDoneTitle: 'Agent finished'
    }
  },
  promptNotices: {
    sessionCommands: {
      busy: command =>
        `chat busy — the agent is still replying. Stop the current reply first (Stop button, or Ctrl+C in a terminal), then run /${command}.`,
      busyCompressing: command => `chat busy — a /compress is running; run /${command} when it finishes.`,
      titleSet: (title, queued) => `Chat title set: ${title}${queued ? ' (queued while the chat starts)' : ''}`,
      titleCleared: 'Chat title cleared.'
    },
    desktopCommands: {
      configFooter:
        'Change settings in Settings, or edit config.yaml with `hermes config edit` on the machine running Hermes.',
      historyHermes: 'Agent',
      saveLegacyRefused:
        'Not saved: Hermes on this machine cannot export a redacted, named or non-JSON copy. Update Hermes, or run /save with no options to write raw JSON on that machine.',
      savedOnHost: path => `Saved on the machine running Hermes: ${path}`,
      exportRemote:
        '/export packages an agent from the machine running Hermes, which is not this computer. Run `hermes profile export` on that machine.',
      importRemote:
        '/import installs an agent on the machine running Hermes, which is not this computer. Run `hermes profile import <archive>` on that machine.',
      checkpointsDisabled: 'Checkpoints are not enabled for this chat (set checkpoints.enabled in config.yaml).'
    }
  },
  sendDiagnostics: {
    privacyNotice:
      'This uploads a debug bundle to Nous-internal storage (not a public paste). It includes system info (OS, versions, provider, which API keys are configured — never the keys themselves) and full agent, Hermes service, and desktop logs (up to 512 KB each), which likely contain conversation content, tool outputs, and file paths. Secrets are redacted before upload. The bundle is viewable only by Nous staff and allowlisted Discord moderators, and auto-deletes after 14 days.'
  },
  keybinds: {
    categories: {
      profiles: 'Agents',
      session: 'Chat'
    },
    actions: {
      'nav.profiles': 'Open agents',
      'nav.cron': 'Open routines',
      'view.toggleProfileRail': 'Toggle agent rail',
      'profile.default': 'Switch to default agent',
      'profile.switch.1': 'Switch to agent 1',
      'profile.switch.2': 'Switch to agent 2',
      'profile.switch.3': 'Switch to agent 3',
      'profile.switch.4': 'Switch to agent 4',
      'profile.switch.5': 'Switch to agent 5',
      'profile.switch.6': 'Switch to agent 6',
      'profile.switch.7': 'Switch to agent 7',
      'profile.switch.8': 'Switch to agent 8',
      'profile.switch.9': 'Switch to agent 9',
      'profile.switch.10': 'Switch to agent 10',
      'profile.switch.11': 'Switch to agent 11',
      'profile.switch.12': 'Switch to agent 12',
      'profile.switch.13': 'Switch to agent 13',
      'profile.switch.14': 'Switch to agent 14',
      'profile.switch.15': 'Switch to agent 15',
      'profile.switch.16': 'Switch to agent 16',
      'profile.switch.17': 'Switch to agent 17',
      'profile.switch.18': 'Switch to agent 18',
      'profile.next': 'Next agent',
      'profile.prev': 'Previous agent',
      'profile.toggleAll': 'Toggle all-agents view',
      'profile.create': 'Create agent',
      'agents.new': 'New agent (Agents page)',
      'nav.artifacts': 'Open library',
      'session.new': 'New chat',
      'session.newTab': 'New chat tab',
      'session.next': 'Next chat',
      'session.prev': 'Previous chat',
      'session.slot.1': 'Switch to recent chat 1',
      'session.slot.2': 'Switch to recent chat 2',
      'session.slot.3': 'Switch to recent chat 3',
      'session.slot.4': 'Switch to recent chat 4',
      'session.slot.5': 'Switch to recent chat 5',
      'session.slot.6': 'Switch to recent chat 6',
      'session.slot.7': 'Switch to recent chat 7',
      'session.slot.8': 'Switch to recent chat 8',
      'session.slot.9': 'Switch to recent chat 9',
      'session.focusSearch': 'Search chats',
      'session.togglePin': 'Pin / unpin current chat',
      'session.archive': 'Archive current chat',
      'view.toggleSidebar': 'Toggle chats sidebar',
      'view.cycleSidebarGrouping': 'Cycle chat grouping'
    }
  },
  settings: {
    subpages: {
      gatewayConnection: 'Primary machine'
    },
    nav: {
      gateway: 'Machines',
      sessions: 'Chats'
    },
    plugins: {
      installModal: {
        profileLabel: 'Install for agent',
        desktopTargetFromPackage: 'Loaded into this app from the package above — same for every agent'
      }
    },
    vault: {
      otpHint:
        'The "setup key" the site shows when you enable 2FA. With it saved, the agent generates the codes itself.'
    },
    notifications: {
      kinds: {
        input: {
          description: 'The agent asked a question or needs a password or secret.'
        }
      }
    },
    searchPlaceholder: {
      gateway: 'Machine connection...',
      sessions: 'Search archived chats...'
    },
    appearance: {
      reactionsDesc: 'iMessage-style emoji tapbacks — react to messages, and the agent can react to yours.',
      tipsDesc:
        'Occasional hints from the app and your agent. Each tip appears once. Turns off automatically after your first 30 days; you can turn it back on.',
      toursDesc:
        'Let the agent spotlight each step as it guides you through the app. Turns off automatically after your first 30 days; you can turn it back on.',
      themeProfileNote: profile => `Saved for ${profile} — each agent keeps its own theme.`,
      pet: {
        intro:
          'Adopt an animated petdex mascot that floats over the app and reacts to what the agent is doing — running while tools execute, celebrating on success, sulking on errors.'
      },
      sessionDensityTitle: 'Chat List Density',
      sessionDensityDesc: 'Choose how much context appears beneath chat titles in the sidebar.'
    },
    uninstallSection: {
      options: {
        full: {
          description: 'Remove the app, the agent, and all user data — config, chats, routines, secrets, logs.'
        }
      }
    },
    memoryLayers: {
      noProviderDesc:
        'The agent uses only the built-in layer below. Choose an external provider under Settings › Memory & Context to add one.',
      builtinDesc:
        'MEMORY.md and USER.md are the agent’s own memory, separate from any external provider. Both load into chats by default; the memory and user profile toggles in Settings › Memory & Context turn each off. A provider adds to this layer; it never replaces it.',
      selectedDesc: current => `Not active yet: chats keep using ${current} until this choice is saved.`,
      providerDesc: name => `${name} adds its own recall to every chat, on top of the built-in layer.`
    },
    about: {
      channel: {
        heldBody: v =>
          `This build is newer than stable release ${v}. Your chats already use a Hermes runtime that ${v} cannot open, so Hermes stays on this build until a newer stable release ships.`
      }
    },
    connections: {
      title: 'Registered machines',
      intro:
        'Manage this device and every machine running Hermes it can reach through remote, SSH, or Cloud connections.',
      stagedNote:
        'Every machine stays connected. Agents, chats, messaging, and routines stay with their machine, and Chats can show all of them at once.',
      launchModeTitle: 'At startup, return to Chats on the last-used machine',
      launchModeDesc: 'When off, Chats opens on the primary machine.',
      searchPlaceholder: 'Search machines…',
      noSearchResults: 'No machines match your search.',
      kindRemote: 'Remote machine',
      kindRemoteDesc: 'A machine running Hermes reachable over HTTP(S) — LAN, Tailscale, or the internet.',
      urlTitle: 'Machine URL',
      headersTitle: 'Extra connection headers',
      headersDesc:
        'Sent with every HTTP and WebSocket request to this machine — for access proxies such as Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). Values are stored encrypted. Headers Hermes manages (Authorization, Cookie, Host…) are ignored.',
      duplicateUrl: (label: string) => `A connection to this machine URL already exists (“${label}”).`
    },
    managedUpdates: {
      intro:
        'Update Desktop-managed SSH installs transactionally: running chats finish, the remote checkout updates, and every agent is restored with a correlated receipt.',
      progress: 'Finishing running chats, updating the remote install, and restoring agents…',
      scopesRestored: (profiles: string) => `Restored agents: ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `Agent “${profile}” not restored: ${error}`
    },
    gateway: {
      loading: 'Loading machine settings...',
      unavailableTitle: 'Machine settings unavailable',
      webappHostDesc:
        "The Webapp always runs on the machine that serves it. The saved connections below are this machine's list, shared with the Hermes Desktop app; other machines are reached through this one.",
      webappOauthPaste: {
        description:
          'If the sign-in tab ended on an address that would not load (this browser is not on the machine running Hermes), paste that address here.'
      },
      webAccess: {
        pendingOn: 'Saved. The web app will be available after Hermes restarts.',
        pendingOff: 'Turned off. The web app stays available until Hermes restarts.',
        unavailableDetail: reason => `Hermes couldn't start the web app: ${reason}`,
        restartBusy: "Saved. A chat is running, so Hermes didn't restart. Choose Restart Hermes once it ends.",
        restartFailed: "Saved, but Hermes couldn't restart. Choose Restart Hermes to try again.",
        restartFailedBecause: reason => `Hermes couldn't restart: ${reason}`,
        startTimedOut: "Hermes didn't start the web app within 90 seconds.",
        stopTimedOut: "Hermes didn't stop the web app within 90 seconds.",
        startStages: {
          restarting: 'Restarting Hermes to start the web app.'
        },
        restartGateway: 'Restart Hermes'
      },
      title: 'Machine connection',
      intro:
        'Local by default. Use remote when this app should run Hermes elsewhere. Connect a machine to discover its agents.',
      localTitle: 'Local machine',
      remoteTitle: 'Remote machine',
      remoteAuthHint: 'Hosted machines use OAuth or a username and password; self-hosted ones may use a session token.',
      cloudSavedTitle: 'Saved Cloud machines',
      cloudSavedDesc:
        'Use a saved machine without changing your default. Sign in below to add instances. Manage names and sign-in in the saved connections list.',
      cloudUseSaved: 'Use machine',
      probing: 'Checking how this machine authenticates…',
      authSignedInPassword:
        'This machine uses a username and password. You are signed in; the session refreshes automatically.',
      authSignedInOauth: 'This machine uses OAuth. You are signed in; the session refreshes automatically.',
      authNeedsPassword: 'This machine uses a username and password. Sign in to authorize this desktop app.',
      authNeedsOauth: provider => `This machine uses OAuth. Sign in with ${provider} to authorize this desktop app.`,
      plainTextConfirmTitle: 'Store the access token in plain text?',
      keychainEncryptionDesc:
        'Off by default. When on, access tokens and sign-in credentials are encrypted with your system keychain (Keychain Access, GNOME Keyring, or Windows DPAPI) — your system may ask for permission or a password. When off, they are stored as plain files readable only by your user account.',
      diagnosticsDesc: 'Reveal desktop.log in your file manager — useful when Hermes fails to start.',
      incompleteTitle: 'Remote machine setup incomplete',
      restartingTitle: 'Machine connection restarting',
      savedTitle: 'Machine settings saved',
      reachableTitle: 'Remote machine reachable',
      signedOutMessage: 'Cleared the remote machine session.',
      failedLoad: 'Machine settings failed to load',
      testFailed: 'Remote machine test failed',
      applyFailed: 'Could not apply machine settings',
      saveFailed: 'Could not save machine settings',
      webappHostTitle: 'Hermes machine'
    },
    profileScope: {
      editsProfile: profile => `Changes on this page apply to the “${profile}” agent.`
    },
    model: {
      tasks: {
        profile_describer: {
          label: 'Agent describer',
          hint: 'Automatic agent descriptions'
        },
        title_generation: {
          hint: 'Chat titles'
        }
      },
      subagents: {
        routing: {
          modes: {
            profile_routes: {
              label: 'Agent routes',
              description: 'Subagents run as one of your agents, using its model, skills and keys.'
            }
          },
          pluginMissing: 'Model routes and Agent routes come from the Advanced Delegation plugin.',
          pluginDisabled: 'Advanced Delegation is installed but turned off for this agent.',
          blockedDisabled: 'Subagents will not start until Advanced Delegation is turned on for this agent.',
          profileRoutesTitle: 'Agents subagents can run as',
          profileRoutesHint: 'Each selected agent brings its own model, skills, instructions and keys.',
          profileEmpty: 'No other agents yet. Create one from the agent rail.',
          profileMissing: name => `${name} (agent no longer exists)`,
          inactiveProfiles: profile =>
            `Agent routes choose which agent each subagent runs as. Subagents that make no selection run as “${profile}”.`,
          inactiveProfilesNoDefault:
            "Agent routes choose which agent each subagent runs as. Subagents that make no selection use the main agent's model."
        }
      },
      appliesDesc: 'Applies to new chats. Use the model picker in the composer to hot-swap the active chat.',
      mainAppliedMessage: model => `New chats will use ${model}.`
    },
    billing: {
      errors: {
        remoteSpendingReconnect: who => `${who} Reconnect from Settings → Machines to re-authorize this device.`,
        sessionRevoked: {
          message: 'Your session was logged out. Sign in again from Settings → Machines.'
        },
        transport: {
          message: 'Billing request failed before reaching Hermes.'
        }
      }
    },
    toolsets: {
      browserRealProfile: {
        prompt: {
          body: 'Let the agent browse with a snapshot of your default browser profile, so sites open already signed in, both for the agent and in the in-app Browser tab.'
        },
        description:
          "Copies your default browser's logins and cookies into a managed snapshot the agent browses with, and signs the in-app Browser tab in with them too. Your live profile is never opened directly. Applies to new chats.",
        enabledMessage: 'New chats will browse with a snapshot of your default browser profile.',
        disabledMessage:
          'The profile snapshot will be deleted and the in-app browser signed out of it; new chats use a clean browser.'
      },
      modelSelectedMessage: model => `${model} applies to new chats.`,
      terminalBackend: {
        selectedMessage: backend => `Terminal commands now run via ${backend}. Applies to new chats.`,
        needsSetupConfirmDescription: detail =>
          `${detail} Chats that start after this change will have no terminal or file tools until setup is finished.`,
        needsSetupConfirmDescriptionGeneric:
          "This backend isn't set up yet. Chats that start after this change will have no terminal or file tools until setup is finished.",
        switchedToLocal: 'Terminal commands now run locally. Applies to new chats.'
      }
    },
    fieldDescriptions: {
      'display.personality': 'Default assistant style for new chats.',
      'memory.memoryEnabled': 'Save durable memories that can help future chats.'
    },
    sessions: {
      loading: 'Loading archived chats…',
      archivedTitle: 'Archived chats',
      defaultDirDesc:
        'New chats start in this folder unless you pick another. Leave it unset to use your home directory.',
      failedLoad: 'Could not load archived chats'
    },
    localModels: {
      sideloadAlreadyPresent: 'Already downloaded.'
    }
  },
  skills: {
    plugins: {
      agentBlurb:
        'Add tools, hooks, and providers to the selected agent. Changes take effect after you restart Hermes.',
      halfDesktopHint: 'this app, same for every agent',
      halfAgentIn: (profile: string) => `Agent: ${profile}`,
      defaultProfile: 'Default agent',
      installAgentHereNoOrigin:
        'The agent half is not installed for this agent, and this package was copied in by hand (no catalog entry or git remote), so it cannot be installed from here. Copy its folder into the agent’s directory or reinstall from Git.',
      empty: 'No plugins installed for this agent.',
      alreadyInstalled: (name: string) => `${name} is already installed for this agent.`,
      updated: (name: string) => `${name} updated to the current catalog pin. Restart Hermes to apply.`,
      autoUpdateTip: hours =>
        `Apply plugin updates automatically when Hermes checks (every ${hours} h). Updated code loads in new chats.`,
      uninstallConfirmBody: (name: string, profile: string) =>
        `This deletes the plugin's files from the ${profile} agent. Any desktop half it shipped is removed with it. Reinstall it from the catalog or from Git at any time.`,
      uninstalled: (name: string) => `${name} uninstalled. Restart Hermes to unload it.`,
      settingsForm: {
        secretStoredAs: (env: string) =>
          `Stored in the agent's .env as ${env}, never in config.yaml; leave blank to keep the current value.`
      }
    },
    remotePluginHint:
      'Open this agent on its machine to manage its plugins. Plugin installation uses the connected machine.',
    openAgentSettings: 'Agent settings',
    appliesToNewSessions: name => `${name} applies to new chats.`,
    bulkUpdated: count => `Updated ${count} ${count === 1 ? 'item' : 'items'} for new chats.`,
    changesApplyNewSessions: 'Changes apply to new chats.',
    openToolLibrary: 'Browse the tool catalog'
  },
  starmap: {
    emptyDesc: 'As your agent builds skills and memories for your work, they appear here.'
  },
  gatewayGroupOpen: {
    lookupFailed: 'Could not look up this group on the machine. Try again.'
  },
  commandCenter: {
    sectionEntries: {
      system: {
        detail: 'Machine status, logs, restart/update'
      },
      sessions: {
        title: 'Chats panel',
        detail: 'Search, pin, and manage chats'
      }
    },
    gatewayRunning: 'Messaging is running',
    gatewayStopped: 'Messaging is stopped',
    restartGateway: 'Restart Hermes',
    gatewayRestartFailed: 'Hermes restart failed.',
    sharedGatewayRestartTitle: 'Restart Hermes for all agents?',
    sharedGatewayRestartDescription: bots => `All agents on this device reconnect: ${bots}`,
    sharedGatewayRestarted: count => `Hermes restarted (${count} ${count === 1 ? 'agent' : 'agents'})`,
    searchPlaceholder: 'Search chats, views, and actions',
    goToSession: 'Go to chat',
    newSessionInProject: project => `New chat in ${project}`,
    sections: {
      sessions: 'Chats'
    },
    nav: {
      newChat: {
        title: 'New chat',
        detail: 'Start a fresh chat'
      },
      artifacts: {
        title: 'Library'
      }
    },
    providerSessions: 'Chats',
    pinSession: 'Pin chat',
    unpinSession: 'Unpin chat',
    exportSession: 'Export chat',
    deleteSession: 'Delete chat',
    noSessions: 'No chats yet.',
    hermesActiveSessions: (version, count) => `Hermes ${version} · Active chats ${count}`,
    statSessions: 'Chats',
    maintenance: {
      backupDesc: 'Zip config, memories, skills, and chats',
      resetUser: 'Reset user profile'
    }
  },
  messaging: {
    states: {
      gateway_stopped: 'Messaging is stopped'
    },
    hintPendingRestart: 'Restart Hermes from the status bar to apply this change.',
    sharedListenerUrl: 'Available at',
    hintGatewayStopped: 'Start Hermes from the status bar to connect.',
    gatewayStopped: 'Messaging is stopped',
    restartToApply: 'This change takes effect after you restart Hermes.',
    restartToReconnect: 'New credentials take effect after you restart Hermes.',
    appliedLive: 'Applied without restarting Hermes.',
    connectingLive: 'Connecting with the new credentials.',
    restartNeeded: 'Saved. Restart Hermes to apply your messaging settings.',
    relayLink: {
      managedNote: (platform: string) =>
        `Team Gateway carries messaging on this deployment, so Hermes turns the direct ${platform} connection off at every start. To run it alongside the relay, set GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true in the Hermes service's environment (a direct connection keeps the instance from scaling to zero and can deliver messages twice).`
    },
    telegramQr: {
      savedRestarting: 'Telegram saved; restarting Hermes…',
      savedRestartFailed: detail => `Telegram saved; Hermes could not restart${detail}`
    },
    fieldCopy: {
      DISCORD_HOME_CHANNEL: {
        help: 'Channel where the bot sends proactive messages (routine results, reminders).'
      },
      QQBOT_HOME_CHANNEL: {
        help: 'Default channel or group for routine results.'
      }
    }
  },
  webhooks: {
    disabledBody:
      'Webhooks work independently of chat channels. Enable them here to accept incoming HTTP events; chat channels are only needed when a subscription delivers to Telegram, Discord, Slack, or another channel.',
    restarting: 'Restarting Hermes...',
    restartNeeded: 'Webhooks are enabled. Restart Hermes to start receiving events.',
    restartGateway: 'Restart Hermes',
    restartFailed: (detail: string) => `Hermes could not restart${detail}`,
    enabledRestarting: 'Webhooks enabled; restarting Hermes...'
  },
  profiles: {
    close: 'Close agents',
    title: 'Agents',
    count: count => `${count} ${count === 1 ? 'agent' : 'agents'}`,
    search: 'Search agents...',
    loading: 'Loading agents...',
    newProfile: 'New agent',
    importProfile: 'Import agent…',
    exportProfile: 'Export agent…',
    imported: 'Agent imported',
    exported: 'Agent exported',
    failedImport: 'Failed to import agent',
    failedExport: 'Failed to export agent',
    exportNeedsHost: name =>
      `The browser can't save an agent archive. On the machine running Hermes, run: hermes profile export ${name}`,
    allProfiles: 'All agents',
    showAllProfiles: 'Show all agents',
    manageProfiles: 'Manage agents…',
    connectGateway: 'Manage machines…',
    fleet: {
      allOnGateway: 'All agents on this machine',
      gateway: gateway => `Agents on ${gateway}`,
      localDevice: 'This device (local backend — installs Hermes if missing, otherwise opens a fresh chat)',
      installDeviceDesc:
        'This will install Hermes locally, then open a fresh chat on this computer. Nothing is installed until you confirm.'
    },
    remoteOverride: {
      description: 'This agent’s chats will run on the remote Hermes you choose, instead of this computer.',
      collisionWarning: (label: string) =>
        `A machine named “${label}” already exists in Settings. This agent connection is separate and will not change it.`,
      confirmTitle: 'Connect this agent to a remote machine?',
      savedTitle: 'Agent connected',
      menuItem: 'Connect to a remote machine…',
      title: profile => `Connect ${profile} to a remote machine`,
      confirmNote: (profile, host) =>
        `New chats in ${profile} will run on ${host}. That computer will run commands and read files there, not on this one. Only connect to a machine you trust.`,
      authFailedTitle: 'Remote machine rejected the saved token'
    },
    defaultProfile: 'Default agent',
    defaultDescription: 'Used when Hermes opens and for new chats. Existing chats stay with their agents.',
    failedSetDefault: 'Could not set the default agent',
    noProfiles: 'No agents yet.',
    selectPrompt: 'Select an agent to view its details.',
    refresh: 'Refresh agents',
    refreshing: 'Refreshing agents',
    soulDesc: 'Instructions that guide this agent’s behavior.',
    soulMissing:
      'This agent has no SOUL.md yet. Add instructions below and save to create it. Personality presets in config.yaml are managed separately.',
    soulPlaceholder: mode => `Instructions for this agent.\nLeave blank to keep the ${mode} default.`,
    deleteTitle: 'Delete agent?',
    createDesc: 'Each agent has its own instructions, settings, skills, and chats.',
    cloneFromDesc: 'Copies settings, skills, and SOUL.md instructions from the selected agent.',
    cloneFromDefaultDesc: 'Copy settings, skills, and SOUL.md instructions from your default agent.',
    createAction: 'Create agent',
    renameTitle: 'Rename agent',
    renameDescPrefix: 'Renaming updates the agent’s directory and any wrapper scripts in ',
    displayNameDesc: 'Sets a display name shown across the app. The internal ID stays "default".',
    created: 'Agent created',
    renamed: 'Agent renamed',
    deleted: 'Agent deleted',
    failedLoad: 'Failed to load agents',
    failedDelete: 'Failed to delete agent',
    failedCreate: 'Failed to create agent',
    failedRename: 'Failed to rename agent',
    status: {
      unread: count => (count === 1 ? '1 unread chat' : `${count} unread chats`),
      needsInput: count => (count === 1 ? '1 chat needs your answer' : `${count} chats need your answer`),
      working: count => (count === 1 ? '1 chat running' : `${count} chats running`)
    }
  },
  cron: {
    close: 'Close routines',
    title: 'Routines',
    count: count => `${count} ${count === 1 ? 'routine' : 'routines'}`,
    search: 'Search routines...',
    loading: 'Loading routines...',
    editJob: 'Edit routine',
    newCron: 'New routine',
    emptyDescNew:
      'Run a prompt automatically at the times you choose and send the results to your preferred destination.',
    emptyTitleNew: 'No routines yet',
    actionsTitle: 'Routine actions',
    resume: 'Resume routine',
    pause: 'Pause routine',
    edit: 'Edit routine',
    deleteTitle: 'Delete routine?',
    deleteDescSuffix: ' permanently. No further runs will start.',
    resumed: 'Routine resumed',
    paused: 'Routine paused',
    triggered: 'Routine triggered',
    deleted: 'Routine deleted',
    created: 'Routine created',
    updated: 'Routine updated',
    failedLoad: 'Failed to load routines',
    failedUpdate: 'Failed to update routine',
    failedTrigger: 'Failed to trigger routine',
    failedDelete: 'Failed to delete routine',
    failedSave: 'Failed to save routine',
    editTitle: 'Edit routine',
    createTitle: 'New routine',
    createDesc:
      'Run a prompt automatically on a schedule. Use cron syntax or a natural phrase like "every 15 minutes".',
    scriptOnlyEditHint: 'This routine runs a script, not an AI prompt. Routine ID:',
    createAction: 'Create routine',
    tabs: {
      jobs: 'Routines'
    },
    blueprints: {
      dialogDesc: 'Fill in the details and set when it runs.',
      scheduled: 'Routine created from blueprint'
    }
  },
  sidebar: {
    filter: {
      profile: 'Agent'
    },
    gatewayGroups: {
      grouping: 'Machine & agent',
      aliasHint: 'Display name only; machine and agent names stay unchanged.'
    },
    profileRail: 'Agent rail',
    nav: {
      cron: 'Routines',
      'new-session': 'New chat',
      artifacts: 'Library'
    },
    cronJobs: 'Routines',
    storageCorrupt: {
      action: 'Quit Hermes for this agent, then inspect the file without changing it, or restore a snapshot:',
      title: 'Chat database is damaged',
      body: profiles =>
        `Hermes can't read all of the chat history for ${profiles}. Chats missing from this list were not deleted; the file they are stored in is damaged.`
    },
    projects: {
      createdInPreviousContext:
        "Project created on the previous connection or agent. Switch back to find it; IDEA.md wasn't written.",
      showAllSessions: 'Show all chats',
      moveFailed: 'Could not move chat',
      toggle: (label, open) => `${open ? 'Show' : 'Hide'} ${label} chats`,
      showAllCount: count => `Show all ${count} chats`
    },
    row: {
      ownedByProfile: profile => `Agent: ${profile}`,
      copyIdFailed: 'Could not copy chat ID',
      sessionActions: 'Chat actions',
      sessionRunning: 'Chat running',
      renameTitle: 'Rename chat',
      untitledPlaceholder: 'Untitled chat',
      deleteTitle: 'Delete chat?',
      deleted: 'Chat deleted'
    },
    searchAria: 'Search chats',
    searchPlaceholder: 'Search chats…',
    noMatch: query => `No chats match “${query}”.`,
    sessions: 'Chats',
    groupAriaGrouped: 'Show chats as a single list',
    groupAriaUngrouped: 'Group chats by workspace',
    showSessions: 'Show chats',
    groupTitleGrouped: 'Ungroup chats',
    projectEmpty: 'No chats yet',
    projectLoadFailed: 'Could not load chats',
    noSessions: 'No chats yet',
    noFilterMatches: 'No chats match these filters',
    newSessionIn: label => `New chat in ${label}`
  },
  composer: {
    voiceEngineChained: 'Speech-to-text + agent voice',
    voiceEngineLive: 'GPT-Live (full-duplex, delegates to the agent)',
    commandDescs: {
      '/export': 'Export an agent (settings, skills, theme) to a shareable archive',
      '/import': 'Import an agent archive as a new agent',
      '/goal': 'Set a standing goal the agent works on across turns until achieved',
      '/profile': 'Switch the active agent',
      '/platform': 'Pause, resume, or list a failing messaging connection',
      '/clear': 'start a new chat',
      '/resume': 'Resume a saved chat',
      '/topic': 'Enable or inspect Telegram DM topic chats',
      '/title': 'Rename the current chat',
      '/handoff': 'Hand off this chat to a messaging platform',
      '/bg': 'Run a prompt in a separate background chat',
      '/heartbeat': 'Set a recurring prompt that re-enters this chat when idle',
      '/loop': 'Re-run a prompt on a recurring interval in this chat',
      '/status': 'Show current chat status',
      '/model': 'Switch the model for this chat'
    },
    attachUrlDesc: 'The agent will fetch the page and include it as context for this turn.',
    queueLostDiscardTip:
      'Hermes restarted mid-turn and cannot finish this one. Discard it so the queued turns behind it run.',
    newSessionPlaceholders: [
      'What are we building?',
      'Give the agent a task',
      "What's on your mind?",
      'Describe what you need',
      'What should we tackle?',
      'Ask anything',
      'Start with a goal'
    ],
    queueDroppedBody:
      'This background queue entry was dropped because its chat could not be resumed after repeated attempts. Nothing else in the queue was affected.',
    browseAllSessions: 'Browse all chats…'
  },
  statusStack: {
    coding: {
      agentShip: 'Ask the agent to open PR'
    },
    control: {
      waitSession: target => `Waiting on chat ${target}`,
      loopDeferredNotice: 'An active goal currently controls the chat.',
      continuationBusy: 'Goal resumed — chat busy, stop the current reply first (Stop button or Esc) to continue',
      controlUnavailable: msg => `Chat controls unavailable: ${msg}`
    }
  },
  install: {
    setupChoiceDesc:
      'Connect this app to a machine already running Hermes, or install Hermes locally on this computer.',
    setupChoiceDescLocal: 'Install Hermes on this computer, or connect to a machine already running Hermes.',
    remoteSetupDesc: 'Enter your machine URL. Hermes Desktop will detect whether it needs a token or browser sign-in.',
    remoteUrlTitle: 'Machine URL',
    remoteUrlDesc: 'Use the base URL of Hermes on the machine, including https:// when remote.',
    probing: 'Detecting machine authentication...',
    authNeedsOauth: provider => `Sign in with ${provider} before testing this machine.`,
    enterUrlFirst: 'Enter a machine URL first.',
    tokenDesc: 'Paste the session token from the Hermes .env file on the remote machine.',
    incompleteSignInTest: 'Sign in before testing this machine that uses OAuth.',
    incompleteTokenTest: 'Enter a session token before testing this machine.',
    connectExistingDesc: 'Use a remote machine with a session token or browser sign-in. No local install will start.',
    remoteUrlPlaceholder: 'https://hermes.example.com'
  },
  onboarding: {
    headerTitle: "Let's set up your agent"
  },
  shell: {
    gatewayMenu: {
      gateway: 'Machine',
      reconnectGateway: 'Reconnect machine',
      machineSettings: 'Machines',
      restartTitle: 'Restart Hermes on this machine?'
    },
    statusbar: {
      gateway: 'Machine',
      gatewayTitle: 'Machine',
      cron: 'Routines',
      openCron: 'Open routines',
      toggleSessionTimer: 'Chat timer',
      cacheHitRateTitle: 'Prompt cache hit rate this chat — cached tokens cost less, so higher is cheaper'
    }
  },
  rightSidebar: {
    terminalGatewayRemoved: 'Removed machine',
    terminalGatewayRemovedBody: "The machine this terminal ran on was removed, so it can't reopen. Close this tab.",
    terminalGatewayNoShellBody: "This machine has no shell access, so this terminal can't reopen. Close this tab."
  },
  preview: {
    saveScopeChanged: 'Switch back to the original connection and agent to save this draft.',
    web: {
      restarting: 'The agent is restarting the server...',
      askRestart: 'Ask the agent to restart the server',
      lookingRestart: taskId => `The agent is looking for a preview server to restart (${taskId})`,
      restartingMessage: 'The agent is working in the background. Watch the preview console for progress.',
      blankPageBody: 'Type an address above to browse, or ask the agent to open a page.',
      finishedRestarting: message => `The agent finished restarting the preview server${message ? `: ${message}` : ''}`,
      restartFailedMessage: 'The agent could not restart the server.',
      stillWorking:
        'The agent is still working, but no restart result has arrived yet. The server command may be running in the foreground.'
    }
  },
  interfaceMode: {
    hint: 'Changes what is shown, not what the agent can do.',
    simple: {
      description: 'For talking to your agent. Sidebar and chat; no terminal, file or diff panes.'
    }
  },
  assistant: {
    thread: {
      loadingResponse: 'The agent is loading a response',
      errorLayers: {
        generic: "The agent couldn't finish this reply",
        runtime: 'The agent hit a problem'
      },
      errorLayerBodies: {
        generic: 'Something went wrong while the agent was replying. Retry, or copy the details if it keeps happening.'
      },
      errorCodes: {
        no_reply: {
          body: 'The agent finished this turn, but no reply reached this window. Retry to send it again.'
        },
        loop_error: {
          title: "The agent couldn't finish this step"
        },
        SESSION_OWNER_BUSY: {
          title: 'The agent is busy in this chat'
        },
        cron_session_read_only: {
          title: 'This chat belongs to a routine',
          body: 'This chat only runs its routine. Your message was not run. Start a new chat to send it.'
        },
        SESSION_OWNER_UNCONFIRMED: {
          body: 'Hermes did not confirm your message in time. If it was queued, it runs once the earlier turn finishes or is resolved. Check the chat before sending it again.'
        },
        launch_credentials_unavailable: {
          body: 'This chat started with an API key that has since changed or been removed, so your message was not sent. An open chat never switches to a different key on its own. Start a new chat to use your current key.'
        }
      },
      errorToastTitle: "The agent couldn't finish the reply",
      loadingSession: 'Loading chat',
      errorStartNewSession: 'Start new chat'
    },
    approval: {
      timedOutSystemLine:
        'Approval timed out — the command was not run. Ask the agent to try again, or raise the limit in Settings → Safety → Approval timeout.',
      alwaysDescription: pattern =>
        `This adds the “${pattern}” pattern to your permanent allowlist (~/.hermes/config.yaml). The agent won’t ask again for commands like this — in this chat or any future one.`,
      allowSession: 'Allow for this chat'
    },
    themePicker: {
      basicDetail: 'For talking to your agent.'
    },
    catalogInstall: {
      targetProfile: profile => `Installs for your ${profile} agent`
    },
    tool: {
      titles: {
        cronjob: {
          done: 'Routine',
          pending: 'Setting up routine',
          pendingAction: 'Setting up'
        },
        session_search_recall: {
          done: 'Searched chat history',
          pending: 'Searching chat history'
        }
      }
    },
    mcpSetup: {
      reloadFailed: 'Server saved, but reloading MCP tools failed — they load in the next chat'
    }
  },
  prompts: {
    sudoInstallDesc:
      'The agent needs your sudo password to install the Agent Screen packages (TigerVNC + Xfce) on its machine. It is sent only to that machine.',
    secretDesc: 'The agent needs a credential to continue.',
    vaultSaveDesc: origin =>
      `The agent reached a sign-in page at ${origin} and has no login for it. Enter it once here; it is encrypted on this machine and filled into the page without the model ever seeing the password.`,
    vaultCodeDesc: site =>
      `${site} is asking for a one-time code (text message, email or authenticator app). Enter it here and the agent types it into the page; the model never sees it.`,
    vaultCodeFootnote:
      'Tip: save the authenticator key with this login in Settings → Passwords & Logins and the agent enters codes for you.',
    sudoDesc:
      'Review the command before entering your sudo password. Your password is sent to the agent running it and cached for this chat.'
  },
  desktop: {
    profileStatus: current =>
      `Agent: ${current}. Use /profile <name> or the "New chat" picker to start a chat with another agent.`,
    unknownProfile: 'Unknown agent',
    noProfileNamed: (target, available) => `No agent named "${target}". Available: ${available}`,
    newChatsProfile: name => `New chats will use agent ${name}.`,
    setProfileFailed: 'Failed to set agent',
    resumeStrandedBody:
      'The connection to this chat failed and automatic retries gave up. Check that Hermes is running on the machine, then try again.',
    poolSlotTimeoutBody:
      "Too many agents are running at once for this computer's limit. Raise the limit in Settings → Advanced, or wait for one to finish and retry.",
    sessionUnavailable: 'Chat unavailable',
    createSessionFailed: 'Could not create a new chat',
    yoloSystem: active => `YOLO ${active ? 'on' : 'off'} for this chat`,
    resumeStrandedTitle: "Couldn't load this chat",
    sessionBusy: 'Chat busy',
    cwdStagedMessage: 'Restart the desktop backend to apply cwd changes to this active chat.',
    sessionExported: 'Chat exported',
    sessionExportFailed: 'Could not export chat',
    sessionOpenFailed: "Couldn't open this chat"
  },
  tips: {
    items: {
      skills: {
        text: 'Skills are folders of instructions the agent loads when the work calls for them.'
      },
      messaging: {
        title: 'Your agent away from your desk'
      },
      artifacts: {
        title: 'Everything your agent made',
        text: 'Images, files and links from every chat, collected in your library.'
      },
      cron: {
        text: 'Routines run a prompt hourly, nightly, or on a cron expression.'
      },
      profiles: {
        title: 'Agents are separate',
        text: 'Each agent has its own instructions, settings, memory, and chats.'
      },
      'command-palette': {
        text: 'Chats, settings, skills and commands all answer to the palette.'
      }
    }
  },
  sessionImport: {
    action: 'Import chat',
    search: 'Search loaded chats',
    scanError: 'Could not find chats',
    emptyHelp: 'Claude Code and Codex chats on this machine will appear here.',
    searchHelp: 'Try another title or folder, or load more chats.',
    more: 'Load more chats',
    chooseHelp: 'Choose a chat to read its history before bringing it into Hermes.'
  },
  titlebar: {
    searchTitle: 'Search chats, views, and actions',
    unreadSessions: count => (count === 1 ? '1 unread chat' : `${count} unread chats`)
  },
  skillDeepLink: {
    installDescription: 'This skill will be available in new chats. Only install sources you trust.'
  },
  artifacts: {
    search: 'Search library...',
    refresh: 'Refresh library',
    refreshing: 'Refreshing library',
    indexing: 'Indexing recent chat outputs',
    noArtifactsTitle: 'Nothing in your library yet',
    noArtifactsDesc: 'Generated images and file outputs will appear here as chats produce them.',
    failedLoad: 'Library failed to load',
    colSession: 'Chat',
    partialIndex: (failed, total) => `Skipped ${failed} of ${total} recent chats while indexing your library.`
  },
  artifactPreview: {
    missingTitle: 'Item unavailable',
    missingBody: 'This item is no longer in the library.'
  },
  updates: {
    applyStatus: {
      noReturn: 'Backend didn’t come back online. The update may not have completed — check the machine running Hermes.'
    }
  },
  freeTier: {
    busyBody: wait =>
      `Hermes couldn't finish signing you in because the Nous service is busy. Try again in ${wait}. Your chat is still here in the meantime.`,
    unreachableBody:
      "Hermes couldn't reach the Nous service to finish signing you in. Check your internet connection and try again. Your chat is still here.",
    setupFailed: {
      locked: "This chat can't continue without signing in. Sign in or create a free Nous account to keep going."
    }
  },
  zones: {
    closeRunningBody:
      'This chat is still working (or waiting on your input). Closing the tab hides it — the chat keeps its progress and can be reopened from the sidebar.',
    newSessionTab: 'New chat tab'
  }
}
