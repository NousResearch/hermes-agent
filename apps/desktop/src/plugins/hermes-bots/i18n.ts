/**
 * Plugin-scoped i18n for Bot Mode — bundles registered under the plugin id via
 * `ctx.i18n.register`, never touching core `en.ts`. Mirrors the kanban plugin:
 * `usePluginI18n` returns a stringly-typed `t(key, …)`, and `useBots()` binds it
 * to the message SHAPE so components keep typed `b.roster.search` access.
 *
 * Only strings Bot Mode OWNS live here. Generic verbs (Cancel, Delete, Remove,
 * Retry, Close, Loading…) and shared vocabulary core already ships in every
 * locale — weekday names, Daily/Hourly, Scheduled jobs — resolve against core
 * via `useI18n()` / `translateNow()`. Duplicating those here would be a
 * second, worse translation that drifts.
 *
 * Three kinds of literal deliberately stay hardcoded, and none of them is a
 * missed key:
 *
 *  - **Prompts sent to a model**, not shown as chrome: the room-picture image
 *    prompt and the scheduled-routine instruction. They are addressed to the
 *    model, which reads English best.
 *  - **Syntax and identifiers**: cron expressions and their examples, React
 *    keys, workspace ids.
 *  - **`'You'`**, the author marker on room-log entries. It is persisted into
 *    the log and compared as a sentinel (`group-activity.ts`), so it stays
 *    English where it is WRITTEN (`group-chat-parts.tsx`, `group-rounds.ts`);
 *    the places that RENDER the reader's own lines use `group.you` instead.
 *
 * Locales follow kanban: `en` / `ja` / `zh` / `zh-hant`. Arabic falls through
 * the resolution chain (active locale → this plugin's `en` → the key) the
 * same way a missing string in any locale does. Nouns match core: ボット /
 * 机器人 / 機器人, プロファイル / 配置档案 / 設定檔, ゲートウェイ / 网关 / 閘道.
 */

import { type PluginLocaleBundles, type PluginTranslate, usePluginI18n } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

import { getPluginCtx } from './shared'

type BotsMessages = {
  /** Left rail: the bot + group-chat roster. */
  editor: {
    fullConfigHint: string
    liveCapabilities: string
    editSoul: string
    remoteCapabilitiesHint: string
    skillsEnabled: (enabled: number, total: number) => string
    toolsetsEnabled: (enabled: number, total: number) => string
    mcpServers: string
    providerCustom: string
    modelCustom: string
    backToDropdowns: string
    inheritLaunch: string
    enterManually: string
    gatewayDefault: string
    modelNameExample: string
    modelSwitchFailed: string
    newDescription: string
    name: string
    title: string
    description: string
    createOn: string
    general: string
    capabilities: string
    skills: string
    tools: string
    cloneFrom: string
    freshProfile: string
    inheritedModel: string
    soul: string
    shareKeys: string
    shareKeysOn: (target: string) => string
    shareKeysHint: string
    needsModel: string
    configureModel: string
    createEmpty: string
    nameTakenHint: string
    nameFirstHint: string
    newerDesktop: string
    newerGateway: string
    emptySkillsHint: string
    defaultToolsHint: string
    catalog: string
    catalogInstalled: string
    mcpHint: string
    creating: string
    createBot: string
    auto: string
    autoHint: string
    unlock: string
    lockFace: string
    lockedHint: string
    unlockedHint: string
    noImageModel: string
    checkingImage: string
    chooseImage: string
    editDescription: (name: string, profile: string) => string
    nameTaken: (name: string) => string
    nameTakenOn: (name: string, target: string) => string
    currentConnection: (name: string) => string
    remoteHint: (target: string) => string
    cloneFromOn: (target: string) => string
    catalogHint: (source: string) => string
    sectionsFailed: (sections: string) => string
    updated: (name: string) => string
    created: (name: string) => string
    createdOn: (name: string, target: string) => string
  }
  roster: {
    search: string
    searchPlaceholder: string
    newBotOrGroup: string
    groupChats: string
    emptyTitle: string
    emptyDesc: string
    noMatchQuery: (query: string) => string
    noMatchQueryOn: (query: string, gateway: string) => string
    noMatchFiltersOn: (gateway: string) => string
    noMatchFilters: string
    clearFilters: string
    allHidden: string
    allHiddenDesc: string
    showHidden: string
    noHiddenMatch: string
    hiddenFromRoster: string
    pinned: string
    needsAttention: string
    needsInput: string
    /** The kind filter's three options, in menu order. */
    botsAndGroups: string
    botsOnly: string
    groupsOnly: string
    /** The activity filter's four options, in menu order. */
    anyActivity: string
    activeNow: string
    recentlyActive: string
    older: string
    /** How a row's owning gateway is doing — see `botSourceStatus`. */
    gatewayRemoved: string
    onDemand: string
    ready: string
    statusUnknown: string
    unavailable: string
    retryNow: string
    rosterUnavailable: (reason: string) => string
    waitingForGateway: string
  }
  /** User-made roster sections (folders the user files bots into). */
  sections: {
    newSection: string
    newTitle: string
    renameTitle: string
    nameLabel: string
    namePlaceholder: string
    create: string
    rename: string
    moveUp: string
    moveDown: string
    unassigned: string
    options: (name: string) => string
    headingTip: string
    emptyHint: string
    moveTo: string
    newSectionEllipsis: string
    removeFromSection: string
    deleted: (name: string, count: number) => string
    undo: string
  }
  /** Creating, editing and removing a bot. */
  bot: {
    newTitle: string
    editTitle: string
    editMenu: string
    helpPromptPlaceholder: string
    descriptionHint: string
    newChatWith: string
    /** Re-opens the forever-chat on purpose. A plain row click only returns to
     *  the tabs already open, so a closed Bot Chat needs an explicit ask. */
    openBotChat: string
    /** Screen-reader label for the row spinner while a cold bot chat opens. */
    openingChat: string
    /** Row context menu: pin/hide toggles, their toasts, and the groups entry. */
    pinToTop: string
    unpin: string
    pinnedToast: (name: string) => string
    unpinnedToast: (name: string) => string
    hide: string
    unhide: string
    hiddenToast: (name: string) => string
    unhiddenToast: (name: string) => string
    groupsMenu: (groups: string) => string
    manageGroups: string
    metadataLoadFailed: string
    loadFailed: string
    groupsLoadFailed: string
    thisDevice: string
    /** Roster badge tooltips per attention class; `attentionFallback` when the class is unknown. */
    attentionFallback: string
    attentionProviderAuth: string
    attentionQuota: string
    attentionMissingConfig: string
    attentionBlocked: string
    duplicate: string
    duplicateFailed: string
    deleteTitle: string
    removeFromAllGroups: string
    createFirstHint: string
    createFailed: string
    advanced: string
    advancedHint: string
    advancedFailed: string
    openAnotherChatUnsupported: string
    remoteConnectionsUnsupported: string
    /** Bot-open failure toasts (canonical-chat.ts notifyBotOpenFailure). The
     *  raw RPC/connection error travels in the toast `detail`, never here. */
    openNeedsUpdateTitle: string
    openNeedsUpdateMessage: (connectionLabel: string) => string
    openUnreachableTitle: string
    openUnreachableMessage: string
    openChatFailedTitle: (botName: string) => string
    openChatFailedMessage: string
    openGateways: string
    /** Stands under the bot's name in a chat it has not spoken in yet. */
    chatEmpty: string
    /** First line of a brand-new bot's forever-chat — see `kickoffText`. */
    kickoff: string
  }
  /** Avatar picker: shapes, blobs, pets, uploads, generation. */
  avatar: {
    classicShapes: string
    blobFromName: string
    unlockFollowsName: string
    randomize: string
    /** The picker's four tabs, in order. */
    tabBot: string
    tabGenerate: string
    upload: string
    tabPet: string
    removeImage: string
    removeBackToShape: string
    describePlaceholder: string
    describeHint: string
    matchTheName: string
    pickPet: string
    petLoadFailed: string
    imageTooLarge: string
    generationFailed: string
    savedLocally: string
    savedLocallyDescriptionFailed: string
    generate: string
    generating: string
  }
  /** Group chats: the room, its composer, threads and activity feed. */
  group: {
    newTitle: string
    manageDesc: string
    manageTitle: string
    settingsTitle: string
    settingsDesc: string
    nameLabel: string
    holdDetection: string
    holdDetectionHint: string
    compressHistory: string
    compressHistoryHint: (member: string) => string
    compressing: (member: string) => string
    compressDone: (member: string, compressed: number, detail: string) => string
    compressNothing: (member: string) => string
    compressFailed: (member: string, error: string) => string
    searchToAdd: string
    searchToAddPlaceholder: string
    removeFromSelection: string
    disbandTitle: string
    deleteTitle: string
    deleteAction: string
    composerPlaceholder: string
    slashCommandsUnsupported: string
    attachHint: string
    newThread: string
    reply: string
    replyInThread: string
    replyInThreadPlaceholder: string
    openThread: string
    collapseThread: string
    collapseThreadLabel: string
    activity: string
    noActivityYet: string
    showActivity: string
    hideActivity: string
    stop: string
    stopHint: string
    allHeldStatus: (count: number) => string
    heldMembersStatus: (members: string) => string
    holdReleaseHint: string
    needsYourInput: string
    noMembersToSend: (group: string) => string
    pictureGenerationFailed: string
    nameTaken: (name: string) => string
    memberCount: (count: number) => string
    /** The reader's own lines in a room: the transcript speaker and the roster preview. */
    you: string
    /** How many of a room's members are reachable right now. */
    availableCount: (available: number, total: number) => string
    settingsHint: (group: string) => string
    settingsLabel: (group: string) => string
    disbandHint: (group: string) => string
    disbandLabel: (group: string) => string
    disbandAction: string
    disbanding: string
    disbandDone: string
    disbanded: (group: string) => string
    /** Wraps the bolded group name, so the name can lead the sentence in
     *  languages that put it there — see core's cron.deleteDesc* pair. */
    disbandDescPrefix: string
    disbandDescSuffix: (count: number) => string
    stopped: (group: string) => string
    removeAttachment: string
    threadFallback: string
    replyCount: (replies: number) => string
    dropToThread: string
    dropToRoom: string
    waitingForAnswer: string
    memberThinking: (name: string) => string
    roomWorking: string
    messageRoom: (group: string) => string
    newThreadPlaceholder: (group: string) => string
    everyoneMeta: string
    commandApproval: string
    answerFailed: (handle: string, error: string) => string
    wantsToRunCommand: (handle: string) => string
    asks: (handle: string) => string
    answerTo: (member: string) => string
  }
  /** Skills hub + MCP setup surfaces embedded in the bot editor. */
  tools: {
    installHint: (name: string) => string
    installed: (name: string) => string
    installFailed: (name: string) => string
    searchHint: string
    resizeHint: string
    addServerFailed: string
    noTarget: string
    setKeyFailed: (name: string) => string
    configured: (name: string) => string
    authenticated: (name: string) => string
    testFailed: string
    completeSignIn: string
    needsSetup: (name: string) => string
    setUpDone: string
    saveTest: string
    authorizing: string
    working: string
    setupFailed: string
    signIn: string
    setUp: string
    skillsHub: string
    filterSkills: string
    searchHub: string
    noMcpServers: string
  }

  /** Bot Screen: the bot's headless desktop on the gateway host, live in a pane. */
  screen: {
    title: string
    menu: string
    unsupportedTitle: string
    unsupportedBody: string
    notInstalledTitle: string
    notInstalledBody: string
    installHint: string
    install: string
    installing: string
    installCancelled: string
    installFailed: string
    noPackageManager: string
    portalTitle: string
    portalOpen: string
    heroStopped: string
    heroNotInstalled: string
    heroConnecting: string
    heroStale: string
    heroSuppressed: string
    heroOpenLive: string
    heroInstall: string
    heroStart: string
    portalWatching: string
    portalYouControl: string
    portalOtherControls: string
    portalStopped: string
    portalNotInstalled: string
    portalUnsupported: string
    portalUnavailable: string
    unavailableTitle: string
    autoOpenMenu: string
    autoOpenOnToast: (name: string) => string
    autoOpenOffToast: (name: string) => string
    stoppedTitle: string
    stoppedBody: string
    start: string
    attaching: string
    streamLost: string
    reconnect: string
    takeOver: string
    handBack: string
    handBackForce: string
    handBackForceHint: string
    openNeedsUpdate: string
    youControl: string
    otherControls: string
    agentControls: string
    controlTaken: string
  }

  /** Bot-scoped scheduled jobs. Generic scheduling chrome (weekday names,
   *  Daily/Hourly, the job verbs) resolves against core's `cron` section. */
  cron: {
    untitled: string
    nameNul: string
    instructionNul: string
    minutesFromNow: string
    hoursFromNow: string
    daysFromNow: string
    stopAfter: string
    runsHint: string
    detailDescription: string
    status: string
    active: string
    paused: string
    schedule: string
    rawSchedule: string
    repeat: string
    nextRun: string
    overdueSince: string
    lastRun: string
    lastResult: string
    workdir: string
    succeeded: string
    failed: string
    deliveryFailed: string
    blockedConfig: string
    legacyUnsafe: string
    filterHint: string
    needsRosterFirst: string
    staleNotice: string
    readFailure: string
    createDesc: (bot: string) => string
    instruction: string
    whenToRun: string
    dayOfMonth: string
    sendResultsTo: string
    runHistoryOnly: string
    botChatTarget: (bot: string) => string
    continuity: string
    onceIn: (when: string) => string
    everyNDays: (days: number) => string
    everyNHours: (hours: number) => string
    everyNMinutes: (minutes: number) => string
    /** The frequency picker's eight options, in menu order. */
    freqOnce: string
    freqHourly: string
    freqDaily: string
    freqWeekdays: string
    freqWeekly: string
    freqMonthly: string
    freqInterval: string
    freqAdvanced: string
    unitMinutes: string
    unitHours: string
    unitDays: string
    /** One-line plain-language read-back of the picker's current state. */
    runsOnce: (count: number, unit: string) => string
    runsHourly: string
    runsDaily: (time: string) => string
    runsWeekdays: (time: string) => string
    runsWeekly: (day: string, time: string) => string
    runsMonthly: (day: string, time: string) => string
    runsInterval: (count: number, unit: string) => string
    runsRaw: string
    timesTotal: (count: number) => string
  }
}

const en: BotsMessages = {
  editor: {
    fullConfigHint: 'Full configuration needs a newer gateway (restart it after updating Hermes).',
    liveCapabilities: 'Capabilities (applies immediately — skills, tools, MCP)',
    editSoul: 'SOUL.md (persona + agent-messaging protocol)',
    remoteCapabilitiesHint:
      'Remote capabilities require a newer desktop. Model and SOUL changes remain staged until you save.',
    skillsEnabled: (enabled, total) => `Skills (${enabled}/${total} enabled)`,
    toolsetsEnabled: (enabled, total) => `Toolsets (${enabled}/${total} enabled — unchecking all restores the default)`,
    mcpServers: 'MCP servers',
    providerCustom: 'Provider (Custom)',
    modelCustom: 'Model (Custom)',
    backToDropdowns: '← Back to dropdowns',
    inheritLaunch: 'Inherit (launch profile)',
    enterManually: '✏️ Enter manually…',
    gatewayDefault: 'gateway default',
    modelNameExample: 'e.g. model name',
    modelSwitchFailed: 'Model switch failed',
    newDescription: 'A named teammate with its own memory, skills, and chat. It can message your other agents.',
    name: 'Name',
    title: 'Title',
    description: 'Description',
    createOn: 'Create on',
    general: 'General',
    capabilities: 'Capabilities',
    skills: 'Skills',
    tools: 'Tools',
    cloneFrom: 'Clone from profile',
    freshProfile: 'Fresh profile (bundled skills)',
    inheritedModel: 'inherited from launch profile',
    soul: 'SOUL.md (optional — replaces the generated persona)',
    shareKeys: 'Share keys & accounts with the main profile',
    shareKeysOn: target => `Share keys & accounts with the default profile on ${target}`,
    shareKeysHint:
      'Subscriptions, OAuth logins, and API keys stay shared (not copied), so token refreshes never invalidate each other. Uncheck for an isolated snapshot copy.',
    needsModel:
      'No model provider is ready for it yet, so it skipped its introduction. Pick a provider and model under Advanced.',
    configureModel: 'Configure model',
    createEmpty: 'Create empty (skip bundled skills)',
    nameTakenHint: 'That name is taken — pick another before configuring capabilities.',
    nameFirstHint: 'Name the bot first — a draft profile is created when you open this tab (discarded if you cancel).',
    newerDesktop: 'Skills need a newer Hermes Desktop.',
    newerGateway: 'Capability catalog needs a newer gateway (restart it after updating Hermes).',
    emptySkillsHint: '“Create empty” is checked — no bundled skills will be installed.',
    defaultToolsHint: 'Leaving all (or none) checked keeps the default toolset behavior.',
    catalog: 'catalog',
    catalogInstalled: 'catalog · installed',
    mcpHint:
      'Configured servers copy from the main profile; catalog entries are the bundled MCP menu. Entries needing API keys route through setup first (credentials follow the shared keys setting).',
    creating: 'Creating…',
    createBot: 'Create Bot',
    auto: 'Auto',
    autoHint: 'Auto — the name decides',
    unlock: 'Unlock',
    lockFace: 'Lock face',
    lockedHint: 'Face locked — renaming won’t change it.',
    unlockedHint: 'Face follows the name.',
    noImageModel:
      'No image model available. If you just enabled one (or updated Hermes), restart the gateway: Ctrl+K → "Restart gateway".',
    checkingImage: 'Checking image backend…',
    chooseImage: 'Choose an image…',
    editDescription: (name, profile) => `Appearance and role for ${name} (${profile}).`,
    nameTaken: name => `An agent named "${name}" already exists.`,
    nameTakenOn: (name, target) => `An agent named "${name}" already exists on ${target}.`,
    currentConnection: name => `${name} (current)`,
    remoteHint: target =>
      `The agent is created on ${target} and appears in the roster as a Connections bot. Chat routes to that machine.`,
    cloneFromOn: target => `Clone from profile (on ${target})`,
    catalogHint: source => `Catalog from ${source} — unchecked skills are disabled after creation.`,
    sectionsFailed: sections => `Some sections failed: ${sections}`,
    updated: name => `${name} updated`,
    created: name => `Bot "${name}" created`,
    createdOn: (name, target) => `Bot "${name}" created on ${target}`
  },
  roster: {
    search: 'Search bots and group chats',
    searchPlaceholder: 'Search bots and group chats…',
    newBotOrGroup: 'New bot or group chat',
    groupChats: 'Group chats',
    emptyTitle: 'No bots yet',
    emptyDesc: 'Create your first bot.',
    noMatchQuery: query => `No bots or group chats match “${query}”`,
    noMatchQueryOn: (query, gateway) => `No bots or group chats match “${query}” on ${gateway}`,
    noMatchFiltersOn: gateway => `No bots or group chats match these filters on ${gateway}`,
    noMatchFilters: 'No bots or group chats match these filters.',
    clearFilters: 'Clear filters',
    allHidden: 'All bots are hidden',
    allHiddenDesc: 'They keep working and retain their history.',
    showHidden: 'Show hidden bots',
    noHiddenMatch: 'No hidden bots match these filters.',
    hiddenFromRoster: 'Hidden from the roster',
    pinned: 'Pinned',
    needsAttention: 'needs attention',
    needsInput: 'Needs your input',
    botsAndGroups: 'Bots and group chats',
    botsOnly: 'Bots only',
    groupsOnly: 'Group chats only',
    anyActivity: 'Any activity',
    activeNow: 'Active now',
    recentlyActive: 'Recently active',
    older: 'Older',
    gatewayRemoved: 'Gateway removed',
    onDemand: 'On demand',
    ready: 'Ready',
    statusUnknown: 'Status unknown',
    unavailable: 'Unavailable',
    retryNow: 'Retry now',
    rosterUnavailable: reason =>
      `Roster unavailable: ${reason}. If your gateway predates profiles.list, update Hermes and restart the gateway.`,
    waitingForGateway:
      'Waiting for the gateway connection… (remote gateways can take a few seconds; retries automatically)'
  },
  sections: {
    newSection: 'New section',
    newTitle: 'New section',
    renameTitle: 'Rename section',
    nameLabel: 'Section name',
    namePlaceholder: 'e.g. Clients',
    create: 'Create',
    rename: 'Rename…',
    moveUp: 'Move up',
    moveDown: 'Move down',
    unassigned: 'Unassigned',
    options: name => `${name} section options`,
    headingTip: 'Drop bots here · double-click to rename',
    emptyHint: 'Drag bots here',
    moveTo: 'Move to section',
    newSectionEllipsis: 'New section…',
    removeFromSection: 'Remove from section',
    deleted: (name, count) =>
      count === 0
        ? `Deleted “${name}”`
        : `Deleted “${name}” — ${count} ${count === 1 ? 'bot' : 'bots'} moved to Unassigned`,
    undo: 'Undo'
  },
  bot: {
    newTitle: 'New bot',
    editTitle: 'Edit profile',
    editMenu: 'Edit…',
    helpPromptPlaceholder: 'What should this bot help with?',
    descriptionHint: 'Leave blank to generate from the bot’s name and description.',
    newChatWith: 'New chat with this bot',
    openBotChat: 'Open Bot Chat',
    openingChat: 'Opening chat…',
    pinToTop: 'Pin to top',
    unpin: 'Unpin',
    pinnedToast: name => `${name} pinned to top`,
    unpinnedToast: name => `${name} unpinned`,
    hide: 'Hide',
    unhide: 'Unhide',
    hiddenToast: name => `${name} hidden — use the eye button in the Bots header to see hidden bots`,
    unhiddenToast: name => `${name} is back in the roster`,
    groupsMenu: groups => `Groups: ${groups}…`,
    manageGroups: 'Manage groups…',
    metadataLoadFailed: 'Could not load bot metadata',
    loadFailed: 'Could not load bot',
    groupsLoadFailed: 'Could not load bot groups',
    thisDevice: 'This device',
    attentionFallback: 'Needs attention',
    attentionProviderAuth: 'Sign in again for this profile',
    attentionQuota: 'Quota or balance exhausted',
    attentionMissingConfig: 'Provider not configured — run hermes model',
    attentionBlocked: 'Bot is blocked — see its last message',
    duplicate: 'Duplicate',
    duplicateFailed: 'Duplicate failed',
    deleteTitle: 'Delete bot and profile?',
    removeFromAllGroups: 'Remove from all groups',
    createFirstHint: 'Open the Bots pane and hit “New Bot”.',
    createFailed: 'Could not create the profile yet',
    advanced: 'Advanced',
    advancedHint: 'Advanced — model, skills, toolsets, SOUL.md',
    advancedFailed: 'Advanced configuration failed',
    openAnotherChatUnsupported: 'Update Hermes Desktop to open another Bot chat.',
    remoteConnectionsUnsupported: 'Update Hermes Desktop to chat with bots on other connections.',
    openNeedsUpdateTitle: 'This bot lives on an older Hermes',
    openNeedsUpdateMessage: connectionLabel => `Update ${connectionLabel}, then try again.`,
    openUnreachableTitle: 'Hermes couldn’t reach the computer this bot runs on',
    openUnreachableMessage: 'Check it is online and try again.',
    openChatFailedTitle: botName => `Could not open ${botName}’s chat`,
    openChatFailedMessage: 'Try again.',
    openGateways: 'Open Gateways',
    chatEmpty: 'Say something to get started.',
    kickoff: 'Hey, tell me about yourself!'
  },
  avatar: {
    classicShapes: 'Classic shapes',
    blobFromName: 'Blob face — drawn from the bot’s name',
    unlockFollowsName: 'Unlock — the face follows the bot’s name again',
    randomize: 'Randomize',
    tabBot: 'Bot',
    tabGenerate: 'Generate',
    upload: 'Upload',
    tabPet: 'Pet',
    removeImage: 'Remove image — use shape',
    removeBackToShape: 'Remove — back to shape avatar',
    describePlaceholder: 'Describe your avatar…',
    describeHint: 'Leave blank to auto-generate from name/title/description + agent-messaging roster.',
    matchTheName: 'Match the name',
    pickPet: 'Pick a pet as this bot’s profile picture.',
    petLoadFailed: 'Could not load that pet — try another.',
    imageTooLarge: 'Image too large (max 15MB).',
    generationFailed: 'Avatar generation failed',
    savedLocally: 'Saved look locally; remote persistence failed',
    savedLocallyDescriptionFailed: 'Saved look locally; description update failed',
    generate: 'Generate',
    generating: 'Generating…'
  },
  group: {
    newTitle: 'New group chat',
    manageDesc: 'A bot can join multiple group chats. Memberships sync to every machine.',
    manageTitle: 'Manage groups',
    settingsTitle: 'Group settings',
    settingsDesc: 'Rename the group or set a room picture. Members and history are kept.',
    nameLabel: 'Group name',
    holdDetection: 'Detect stop directives',
    holdDetectionHint: 'Let room messages put addressed members on hold until they are mentioned again.',
    compressHistory: 'Compress history',
    compressHistoryHint: (member: string) =>
      `Compress ${member}'s hidden room history so the member stops failing with empty replies`,
    compressing: (member: string) => `Compressing ${member}'s room history…`,
    compressDone: (member: string, compressed: number, detail: string) =>
      `Compressed ${compressed} room session${compressed === 1 ? '' : 's'} for ${member}${detail ? ` — ${detail}` : ''}`,
    compressNothing: (member: string) => `Nothing to compress for ${member} — no room session yet`,
    compressFailed: (member: string, error: string) => `Could not compress ${member}'s room history: ${error}`,
    searchToAdd: 'Search bots to add',
    searchToAddPlaceholder: 'Search bots to add…',
    removeFromSelection: 'Remove from selection',
    disbandTitle: 'Disband group chat?',
    deleteTitle: 'Delete group chat?',
    deleteAction: 'Delete',
    composerPlaceholder: 'Say something — every bot in this group hears the room.',
    slashCommandsUnsupported:
      'Slash commands are not supported in group chats. Open an individual bot chat to use them.',
    attachHint: 'Attach files — every responding bot sees them',
    newThread: 'New Thread',
    reply: 'Reply',
    replyInThread: 'Reply in thread',
    replyInThreadPlaceholder: 'Reply in thread…',
    openThread: 'Open this thread',
    collapseThread: 'Collapse thread',
    collapseThreadLabel: 'Collapse this thread',
    activity: 'Activity',
    noActivityYet: 'No activity in this turn yet.',
    showActivity: 'Show room activity',
    hideActivity: 'Hide room activity',
    stop: 'Stop',
    stopHint: 'Stop this run — interrupts the member on turn and holds the rest',
    allHeldStatus: count => `All ${count} bots are paused`,
    heldMembersStatus: members => `Paused: ${members}`,
    holdReleaseHint: 'Mention a paused bot or send @all resume to release them.',
    needsYourInput: 'A bot in this group chat needs your input',
    noMembersToSend: group =>
      `${group} has no members to send to — add a bot, or reopen the room if members are still loading.`,
    pictureGenerationFailed: 'Group picture generation failed',
    nameTaken: name => `A group named “${name}” already exists.`,
    memberCount: count => `${count} bots`,
    you: 'You',
    availableCount: (available, total) => `${available} of ${total} available`,
    settingsHint: group => `Group settings — rename ${group} or set a room picture`,
    settingsLabel: group => `Group settings for ${group}`,
    disbandHint: group => `Disband the ${group} group chat`,
    disbandLabel: group => `Disband ${group}`,
    disbandAction: 'Disband',
    disbanding: 'Disbanding…',
    disbandDone: 'Disbanded',
    disbanded: group => `Disbanded “${group}”`,
    disbandDescPrefix: 'This removes the ',
    disbandDescSuffix: count =>
      ` grouping from its ${count} bots and clears the shared room log. The bots themselves and their per-group sessions are kept.`,
    stopped: group => `Stopped ${group} — remaining turns are held until you resume`,
    removeAttachment: 'Remove attachment',
    threadFallback: 'Thread',
    replyCount: replies => `${replies} ${replies === 1 ? 'reply' : 'replies'}`,
    dropToThread: 'Drop to attach to this thread reply',
    dropToRoom: 'Drop to attach — every responding bot sees it',
    waitingForAnswer: 'Waiting for your answer…',
    memberThinking: name => `${name} is thinking…`,
    roomWorking: 'The room is working…',
    messageRoom: group => `Message ${group}`,
    newThreadPlaceholder: group => `New thread in ${group}… (@name to direct, @everyone for all)`,
    everyoneMeta: 'Every bot in the room',
    commandApproval: 'command approval',
    answerFailed: (handle, error) => `Could not send the answer to @${handle}: ${error}`,
    wantsToRunCommand: handle => `@${handle} wants to run a command:`,
    asks: handle => `@${handle} asks:`,
    answerTo: member => `Answer @${member}`
  },
  tools: {
    installHint: name => `Install "${name}" and add it to the list above`,
    installed: name => `Skill "${name}" installed`,
    installFailed: name => `Installing "${name}" failed`,
    searchHint: 'Searching community + well-known sources — can take ~10s…',
    resizeHint: 'Drag the corner to resize.',
    addServerFailed: 'Could not add server',
    noTarget: 'No target profile',
    setKeyFailed: key => `Failed to set ${key}`,
    configured: name => `${name} configured`,
    authenticated: name => `${name} authenticated`,
    testFailed: 'Server test failed after setup',
    completeSignIn: 'Complete sign-in in your browser...',
    needsSetup: keys => `needs setup (${keys}) — restart the gateway to enable in-app setup`,
    setUpDone: 'set up ✓',
    saveTest: 'Save & test',
    authorizing: 'Authorizing…',
    working: 'Working…',
    setupFailed: 'Setup failed',
    signIn: 'Sign in…',
    setUp: 'Set up…',
    skillsHub: 'Hermes Skills Hub',
    filterSkills: 'Filter skills…',
    searchHub: 'Search the hub (community + well-known sources)…',
    noMcpServers: 'No MCP servers configured or in the catalog.'
  },
  screen: {
    title: 'Screen',
    menu: 'Open Screen',
    unsupportedTitle: 'No bot screen on this host',
    unsupportedBody: 'Bot screens run on Linux gateway hosts. This bot uses the host\u2019s own display.',
    notInstalledTitle: 'Screen packages missing',
    notInstalledBody: 'The gateway host needs TigerVNC and the Xfce core to give this bot a screen. Run on the host:',
    installHint: 'Runs on the gateway host as the user Hermes runs as; sudo is asked for once, through Hermes.',
    install: 'Install on host',
    installing: 'Installing…',
    installCancelled: 'Install cancelled: no sudo password was provided.',
    installFailed: 'Install failed. Read the log above, or run the command on the host yourself.',
    noPackageManager: 'No supported package manager (apt, dnf, pacman) was found on the gateway host.',
    portalTitle: 'Screen',
    portalOpen: 'Open',
    heroStopped: 'Screen is off',
    heroNotInstalled: 'Not installed on this host',
    heroConnecting: 'Checking the screen…',
    heroStale: 'Last seen — screen unreachable',
    heroSuppressed: 'Hidden while someone has control',
    heroOpenLive: 'Open live',
    heroInstall: 'Install',
    heroStart: 'Start',
    portalWatching: 'Live · bot in control',
    portalYouControl: 'Live · you are in control',
    portalOtherControls: 'Live · another viewer in control',
    portalStopped: 'Stopped',
    portalNotInstalled: 'Not installed on host',
    portalUnsupported: 'Not available on this host',
    portalUnavailable: 'Update the bot\u2019s Hermes to use Screen',
    unavailableTitle: 'Screen needs a newer Hermes',
    autoOpenMenu: 'Open Screen when the bot uses it',
    autoOpenOnToast: name => `${name}’s Screen opens when it starts using its desktop`,
    autoOpenOffToast: name => `${name}’s Screen stays closed until you open it`,
    stoppedTitle: 'Screen is off',
    stoppedBody: 'Start this bot\u2019s desktop to watch what it does and take over when it needs you.',
    start: 'Start screen',
    attaching: 'Connecting to the screen\u2026',
    streamLost: 'Screen stream ended',
    reconnect: 'Reconnect',
    takeOver: 'Take over',
    handBack: 'Hand back',
    handBackForce: 'Hand back (force)',
    handBackForceHint: 'Release a lease held by a viewer that is no longer here, e.g. after a reload.',
    openNeedsUpdate: 'Update Hermes Desktop to open bot screens.',
    youControl: 'You are in control',
    otherControls: 'Another viewer is in control',
    agentControls: 'Bot is in control',
    controlTaken: 'Another viewer took control. Watching only.'
  },
  cron: {
    untitled: 'Untitled job',
    nameNul: 'Job name cannot contain NUL (U+0000).',
    instructionNul: 'Job instruction cannot contain NUL (U+0000).',
    minutesFromNow: 'minutes from now',
    hoursFromNow: 'hours from now',
    daysFromNow: 'days from now',
    stopAfter: 'Stop after',
    runsHint: 'runs (blank = forever)',
    detailDescription: 'What this job runs, and when it runs next.',
    status: 'Status',
    active: 'Active',
    paused: 'Paused',
    schedule: 'Schedule',
    rawSchedule: 'Schedule (raw)',
    repeat: 'Repeat',
    nextRun: 'Next run',
    overdueSince: 'Overdue since',
    lastRun: 'Last run',
    lastResult: 'Last result',
    workdir: 'Working directory',
    succeeded: 'Succeeded',
    failed: 'Failed',
    deliveryFailed: 'Ran, but delivery failed',
    blockedConfig: 'Blocked by configuration (not run)',
    legacyUnsafe: 'Paused for security: delete and recreate this legacy job before running it again.',
    filterHint:
      'Scheduled jobs exist in this profile but none are tagged for this bot. Name a job "[bot:<name>] …" to show it here, or see them in Cron below.',
    needsRosterFirst: 'This bot has to appear in the roster first.',
    staleNotice: 'Could not refresh scheduled jobs. Showing the last list we had.',
    readFailure: 'The list may still be there — this was a read failure, not a delete.',
    createDesc: bot => `A recurring task ${bot} runs on a schedule. Runs land in its own chat history.`,
    instruction: 'Instruction',
    whenToRun: 'When to run',
    dayOfMonth: 'Day of month',
    sendResultsTo: 'Send results to',
    runHistoryOnly: 'Run history only',
    botChatTarget: bot => `${bot}’s chat (bot responds)`,
    continuity: 'Continuity: each run sees the previous run’s output (dedupe, continue where it left off)',
    onceIn: when => `Once (${when})`,
    everyNDays: days => `Every ${days} days`,
    everyNHours: hours => `Every ${hours}h`,
    everyNMinutes: minutes => `Every ${minutes}m`,
    freqOnce: 'Once, in…',
    freqHourly: 'Every hour',
    freqDaily: 'Every day',
    freqWeekdays: 'Weekdays',
    freqWeekly: 'Every week',
    freqMonthly: 'Every month',
    freqInterval: 'Interval',
    freqAdvanced: 'Advanced…',
    unitMinutes: 'minute(s)',
    unitHours: 'hour(s)',
    unitDays: 'day(s)',
    runsOnce: (count, unit) => `Runs once, ${count} ${unit} from now`,
    runsHourly: 'Runs at the top of every hour',
    runsDaily: time => `Runs every day at ${time}`,
    runsWeekdays: time => `Runs Monday–Friday at ${time}`,
    runsWeekly: (day, time) => `Runs every ${day} at ${time}`,
    runsMonthly: (day, time) => `Runs on day ${day} of each month at ${time}`,
    runsInterval: (count, unit) => `Runs every ${count} ${unit}`,
    runsRaw: 'Raw schedule — every Nm/Nh/Nd or 5-field cron',
    timesTotal: count => `, ${count} time(s) total`
  }
}

const ja: BotsMessages = {
  editor: {
    fullConfigHint: 'すべての設定を使うには新しいゲートウェイが必要です（Hermes 更新後に再起動してください）。',
    liveCapabilities: '機能（即時適用 — スキル、ツール、MCP）',
    editSoul: 'SOUL.md（人格 + エージェント間メッセージプロトコル）',
    remoteCapabilitiesHint:
      'リモート機能には新しいデスクトップアプリが必要です。モデルと SOUL の変更は保存するまで適用されません。',
    skillsEnabled: (enabled, total) => `スキル（${enabled}/${total} 有効）`,
    toolsetsEnabled: (enabled, total) => `ツールセット（${enabled}/${total} 有効 — すべて解除すると既定値に戻ります）`,
    mcpServers: 'MCP サーバー',
    providerCustom: 'プロバイダー（カスタム）',
    modelCustom: 'モデル（カスタム）',
    backToDropdowns: '← 選択リストに戻る',
    inheritLaunch: '継承（起動プロファイル）',
    enterManually: '✏️ 手動入力…',
    gatewayDefault: 'ゲートウェイの既定値',
    modelNameExample: '例：モデル名',
    modelSwitchFailed: 'モデルの切り替えに失敗しました',
    newDescription:
      '独自のメモリ、スキル、チャットを持つ名前付きの仲間です。他のエージェントとメッセージをやり取りできます。',
    name: '名前',
    title: '表示名',
    description: '説明',
    createOn: '作成先',
    general: '一般',
    capabilities: '機能',
    skills: 'スキル',
    tools: 'ツール',
    cloneFrom: '複製元のプロファイル',
    freshProfile: '新規プロファイル（同梱スキル）',
    inheritedModel: '起動時のプロファイルから継承',
    soul: 'SOUL.md（任意 — 生成された人格を置き換えます）',
    shareKeys: 'メインプロファイルとキー・アカウントを共有',
    shareKeysOn: target => `${target} の default プロファイルとキー・アカウントを共有`,
    shareKeysHint:
      'サブスクリプション、OAuth ログイン、API キーをコピーせず共有するため、トークン更新で互いに無効になりません。チェックを外すと独立したスナップショットをコピーします。',
    needsModel:
      'まだ使用できるモデルプロバイダーがないため、自己紹介をスキップしました。「詳細設定」でプロバイダーとモデルを選んでください。',
    configureModel: 'モデルを設定',
    createEmpty: '空のプロファイルを作成（同梱スキルを除外）',
    nameTakenHint: 'その名前は使用済みです。機能を設定する前に別の名前を選んでください。',
    nameFirstHint:
      '先にボットに名前を付けてください。このタブを開くと下書きプロファイルが作成されます（キャンセルすると破棄されます）。',
    newerDesktop: 'スキルには新しい Hermes Desktop が必要です。',
    newerGateway: '機能カタログには新しいゲートウェイが必要です（Hermes 更新後に再起動してください）。',
    emptySkillsHint: '「空のプロファイルを作成」が選択されているため、同梱スキルはインストールされません。',
    defaultToolsHint: 'すべて選択するか、何も選択しない場合は、既定のツールセット動作を維持します。',
    catalog: 'カタログ',
    catalogInstalled: 'カタログ · インストール済み',
    mcpHint:
      '設定済みサーバーはメインプロファイルからコピーされます。カタログは同梱 MCP メニューです。API キーが必要な項目は先に設定を行います（認証情報はキー共有設定に従います）。',
    creating: '作成中…',
    createBot: 'ボットを作成',
    auto: '自動',
    autoHint: '自動 — 名前から決定',
    unlock: 'ロック解除',
    lockFace: '顔を固定',
    lockedHint: '顔を固定しました。名前を変えても変化しません。',
    unlockedHint: '顔は名前に合わせて変わります。',
    noImageModel:
      '画像モデルがありません。有効にした直後や Hermes 更新後の場合は、Ctrl+K →「ゲートウェイを再起動」で再起動してください。',
    checkingImage: '画像バックエンドを確認中…',
    chooseImage: '画像を選択…',
    editDescription: (name, profile) => `${name}（${profile}）の外観と役割。`,
    nameTaken: name => `「${name}」というエージェントはすでに存在します。`,
    nameTakenOn: (name, target) => `${target} には「${name}」というエージェントがすでに存在します。`,
    currentConnection: name => `${name}（現在）`,
    remoteHint: target =>
      `エージェントは ${target} に作成され、接続先のボットとして一覧に表示されます。チャットはそのマシンに送られます。`,
    cloneFromOn: target => `複製元のプロファイル（${target} 上）`,
    catalogHint: source => `${source} のカタログです。未選択のスキルは作成後に無効になります。`,
    sectionsFailed: sections => `一部の設定に失敗しました: ${sections}`,
    updated: name => `${name} を更新しました`,
    created: name => `ボット「${name}」を作成しました`,
    createdOn: (name, target) => `${target} にボット「${name}」を作成しました`
  },
  roster: {
    search: 'ボットとグループチャットを検索',
    searchPlaceholder: 'ボットとグループチャットを検索…',
    newBotOrGroup: '新しいボットまたはグループチャット',
    groupChats: 'グループチャット',
    emptyTitle: 'ボットはまだありません',
    emptyDesc: '最初のボットを作成しましょう。',
    noMatchQuery: query => `「${query}」に一致するボットやグループチャットはありません`,
    noMatchQueryOn: (query, gateway) => `${gateway} に「${query}」に一致するボットやグループチャットはありません`,
    noMatchFiltersOn: gateway => `${gateway} にこれらのフィルタに一致するボットやグループチャットはありません`,
    noMatchFilters: 'これらのフィルタに一致するボットやグループチャットはありません。',
    clearFilters: 'フィルタをクリア',
    allHidden: 'すべてのボットが非表示です',
    allHiddenDesc: '非表示でも動作を続け、履歴も残ります。',
    showHidden: '非表示のボットを表示',
    noHiddenMatch: 'これらのフィルタに一致する非表示ボットはありません。',
    hiddenFromRoster: '名簿から非表示',
    pinned: 'ピン留め',
    needsAttention: '要対応',
    needsInput: '入力が必要です',
    botsAndGroups: 'ボットとグループチャット',
    botsOnly: 'ボットのみ',
    groupsOnly: 'グループチャットのみ',
    anyActivity: 'すべてのアクティビティ',
    activeNow: '現在アクティブ',
    recentlyActive: '最近アクティブ',
    older: '以前',
    gatewayRemoved: 'ゲートウェイが削除されました',
    onDemand: 'オンデマンド',
    ready: '準備完了',
    statusUnknown: '状態不明',
    unavailable: '利用できません',
    retryNow: '今すぐ再試行',
    rosterUnavailable: reason =>
      `名簿を取得できません: ${reason}。ゲートウェイが profiles.list より前の場合は、Hermes を更新してゲートウェイを再起動してください。`,
    waitingForGateway: 'ゲートウェイ接続を待っています…（リモートは数秒かかることがあります。自動で再試行します）'
  },
  sections: {
    newSection: '新しいセクション',
    newTitle: '新しいセクション',
    renameTitle: 'セクション名を変更',
    nameLabel: 'セクション名',
    namePlaceholder: '例: クライアント',
    create: '作成',
    rename: '名前を変更…',
    moveUp: '上へ移動',
    moveDown: '下へ移動',
    unassigned: '未分類',
    options: name => `${name} セクションのオプション`,
    headingTip: 'ここにボットをドロップ · ダブルクリックで名前を変更',
    emptyHint: 'ここにボットをドラッグ',
    moveTo: 'セクションへ移動',
    newSectionEllipsis: '新しいセクション…',
    removeFromSection: 'セクションから外す',
    deleted: (name, count) =>
      count === 0
        ? `「${name}」を削除しました`
        : `「${name}」を削除しました — ${count} 件のボットを未分類に移動しました`,
    undo: '元に戻す'
  },
  bot: {
    newTitle: '新しいボット',
    editTitle: 'プロファイルを編集',
    editMenu: '編集…',
    helpPromptPlaceholder: 'このボットは何を手伝いますか？',
    descriptionHint: '空欄のままにすると、ボットの名前と説明から生成します。',
    newChatWith: 'このボットと新しいチャット',
    openBotChat: 'ボットチャットを開く',
    openingChat: 'チャットを開いています…',
    pinToTop: '先頭にピン留め',
    unpin: 'ピン留めを解除',
    pinnedToast: name => `${name}を先頭にピン留めしました`,
    unpinnedToast: name => `${name}のピン留めを解除しました`,
    hide: '非表示',
    unhide: '再表示',
    hiddenToast: name => `${name}を非表示にしました — Botsヘッダーの目のボタンで非表示のボットを表示できます`,
    unhiddenToast: name => `${name}が一覧に戻りました`,
    groupsMenu: groups => `グループ: ${groups}…`,
    manageGroups: 'グループを管理…',
    metadataLoadFailed: 'ボットのメタデータを読み込めませんでした',
    loadFailed: 'ボットを読み込めませんでした',
    groupsLoadFailed: 'ボットのグループを読み込めませんでした',
    thisDevice: 'このデバイス',
    attentionFallback: '要対応',
    attentionProviderAuth: 'このプロファイルで再度サインインしてください',
    attentionQuota: 'クォータまたは残高が不足しています',
    attentionMissingConfig: 'プロバイダーが未設定です — hermes model を実行してください',
    attentionBlocked: 'ボットがブロックされています — 最後のメッセージを確認してください',
    duplicate: '複製',
    duplicateFailed: '複製に失敗しました',
    deleteTitle: 'ボットとプロファイルを削除しますか？',
    removeFromAllGroups: 'すべてのグループから外す',
    createFirstHint: 'ボットパネルを開いて「新しいボット」を押してください。',
    createFailed: 'プロファイルをまだ作成できませんでした',
    advanced: '詳細設定',
    advancedHint: '詳細設定 — モデル、スキル、ツールセット、SOUL.md',
    advancedFailed: '詳細設定に失敗しました',
    openAnotherChatUnsupported: '別のボットチャットを開くには Hermes Desktop を更新してください。',
    remoteConnectionsUnsupported: '他の接続上のボットとチャットするには Hermes Desktop を更新してください。',
    openNeedsUpdateTitle: 'このボットは古い Hermes 上で動いています',
    openNeedsUpdateMessage: connectionLabel => `${connectionLabel} を更新してから、もう一度お試しください。`,
    openUnreachableTitle: 'このボットが動いているコンピューターに Hermes が接続できませんでした',
    openUnreachableMessage: 'オンラインか確認して、もう一度お試しください。',
    openChatFailedTitle: botName => `${botName} のチャットを開けませんでした`,
    openChatFailedMessage: 'もう一度お試しください。',
    openGateways: 'ゲートウェイを開く',
    chatEmpty: '何か書いて始めましょう。',
    kickoff: 'こんにちは、自己紹介をしてください！'
  },
  avatar: {
    classicShapes: 'クラシックシェイプ',
    blobFromName: 'ブロブ顔 — ボットの名前から描画',
    unlockFollowsName: 'ロック解除 — 顔がボットの名前に再び追従します',
    randomize: 'ランダム',
    tabBot: 'ボット',
    tabGenerate: '生成',
    upload: 'アップロード',
    tabPet: 'ペット',
    removeImage: '画像を削除してシェイプを使う',
    removeBackToShape: '削除 — シェイプアバターに戻す',
    describePlaceholder: 'アバターを説明…',
    describeHint: '空欄のままにすると、名前・タイトル・説明と agent-messaging の名簿から自動生成します。',
    matchTheName: '名前に合わせる',
    pickPet: 'このボットのプロフィール画像としてペットを選びます。',
    petLoadFailed: 'そのペットを読み込めませんでした。別のペットを試してください。',
    imageTooLarge: '画像が大きすぎます（最大 15MB）。',
    generationFailed: 'アバターの生成に失敗しました',
    savedLocally: '見た目はローカルに保存されましたが、リモートへの保存に失敗しました',
    savedLocallyDescriptionFailed: '見た目はローカルに保存されましたが、説明の更新に失敗しました',
    generate: '生成',
    generating: '生成中…'
  },
  group: {
    newTitle: '新しいグループチャット',
    manageDesc: 'ボットは複数のグループチャットに参加できます。メンバーシップはすべてのマシンに同期されます。',
    manageTitle: 'グループを管理',
    settingsTitle: 'グループ設定',
    settingsDesc: 'グループ名の変更や部屋の画像の設定ができます。メンバーと履歴は保持されます。',
    nameLabel: 'グループ名',
    holdDetection: '停止指示を検出',
    holdDetectionHint: 'ルームのメッセージで、再びメンションされるまで対象メンバーを保留にします。',
    compressHistory: '履歴を圧縮',
    compressHistoryHint: (member: string) =>
      `${member} の非表示のルーム履歴を圧縮し、空の応答で失敗しなくなるようにします`,
    compressing: (member: string) => `${member} のルーム履歴を圧縮中…`,
    compressDone: (member: string, compressed: number, detail: string) =>
      `${member} のルームセッション ${compressed} 件を圧縮しました${detail ? ` — ${detail}` : ''}`,
    compressNothing: (member: string) => `${member} に圧縮する履歴はありません — ルームセッションがまだありません`,
    compressFailed: (member: string, error: string) => `${member} のルーム履歴を圧縮できませんでした: ${error}`,
    searchToAdd: '追加するボットを検索',
    searchToAddPlaceholder: '追加するボットを検索…',
    removeFromSelection: '選択から外す',
    disbandTitle: 'グループチャットを解散しますか？',
    deleteTitle: 'グループチャットを削除しますか？',
    deleteAction: '削除',
    composerPlaceholder: '何か書いてください — このグループのすべてのボットが部屋の内容を受け取ります。',
    slashCommandsUnsupported:
      'グループチャットではスラッシュコマンドを使用できません。個別のボットチャットを開いて使用してください。',
    attachHint: 'ファイルを添付 — 応答するすべてのボットが見ます',
    newThread: '新しいスレッド',
    reply: '返信',
    replyInThread: 'スレッドで返信',
    replyInThreadPlaceholder: 'スレッドで返信…',
    openThread: 'このスレッドを開く',
    collapseThread: 'スレッドを折りたたむ',
    collapseThreadLabel: 'このスレッドを折りたたむ',
    activity: 'アクティビティ',
    noActivityYet: 'このターンのアクティビティはまだありません。',
    showActivity: '部屋のアクティビティを表示',
    hideActivity: '部屋のアクティビティを隠す',
    stop: '停止',
    stopHint: 'この実行を停止 — ターン中のメンバーを中断し、残りを保留します',
    allHeldStatus: count => `すべてのボット（${count}体）が一時停止中`,
    heldMembersStatus: members => `一時停止中: ${members}`,
    holdReleaseHint: '一時停止中のボットにメンションするか、@all resume を送信して再開します。',
    needsYourInput: 'このグループチャットのボットが入力を待っています',
    noMembersToSend: group =>
      `${group} に送信先のメンバーがいません。ボットを追加するか、メンバーの読み込み中であればルームを開き直してください。`,
    pictureGenerationFailed: 'グループ画像の生成に失敗しました',
    nameTaken: name => `「${name}」という名前のグループはすでに存在します。`,
    memberCount: count => `ボット${count}体`,
    you: 'あなた',
    availableCount: (available, total) => `${total}体中${available}体が利用可能`,
    settingsHint: group => `グループ設定 — ${group}の名前変更やルーム画像の設定`,
    settingsLabel: group => `${group}のグループ設定`,
    disbandHint: group => `${group}グループチャットを解散`,
    disbandLabel: group => `${group}を解散`,
    disbandAction: '解散',
    disbanding: '解散中…',
    disbandDone: '解散しました',
    disbanded: group => `「${group}」を解散しました`,
    disbandDescPrefix: '',
    disbandDescSuffix: count =>
      `のグループ分けをボット${count}体から解除し、共有ルームログを消去します。ボット自体と各グループのセッションは保持されます。`,
    stopped: group => `${group}を停止しました — 残りのターンは再開するまで保留されます`,
    removeAttachment: '添付を削除',
    threadFallback: 'スレッド',
    replyCount: replies => `返信${replies}件`,
    dropToThread: 'ドロップしてこのスレッド返信に添付',
    dropToRoom: 'ドロップして添付 — 応答するすべてのボットが見られます',
    waitingForAnswer: 'あなたの回答を待っています…',
    memberThinking: name => `${name}が考えています…`,
    roomWorking: 'ルームが作業中です…',
    messageRoom: group => `${group}にメッセージ`,
    newThreadPlaceholder: group => `${group}で新しいスレッド…（@名前で個別、@everyoneで全員）`,
    everyoneMeta: 'ルーム内のすべてのボット',
    commandApproval: 'コマンドの承認',
    answerFailed: (handle, error) => `@${handle}に回答を送信できませんでした: ${error}`,
    wantsToRunCommand: handle => `@${handle}がコマンドを実行しようとしています:`,
    asks: handle => `@${handle}からの質問:`,
    answerTo: member => `@${member}に回答`
  },
  tools: {
    installHint: name => `「${name}」をインストールして上の一覧に追加`,
    installed: name => `スキル「${name}」をインストールしました`,
    installFailed: name => `「${name}」のインストールに失敗しました`,
    searchHint: 'コミュニティと主要なソースを検索中 — 約 10 秒かかる場合があります…',
    resizeHint: '角をドラッグしてサイズを変更できます。',
    addServerFailed: 'サーバーを追加できませんでした',
    noTarget: '対象プロファイルがありません',
    setKeyFailed: key => `${key} の設定に失敗しました`,
    configured: name => `${name} を設定しました`,
    authenticated: name => `${name} の認証が完了しました`,
    testFailed: '設定後のサーバーテストに失敗しました',
    completeSignIn: 'ブラウザーでサインインを完了してください…',
    needsSetup: keys => `設定が必要（${keys}）— アプリ内で設定するにはゲートウェイを再起動してください`,
    setUpDone: '設定済み ✓',
    saveTest: '保存してテスト',
    authorizing: '認証中…',
    working: '処理中…',
    setupFailed: '設定に失敗しました',
    signIn: 'サインイン…',
    setUp: '設定…',
    skillsHub: 'Hermes スキルハブ',
    filterSkills: 'スキルを絞り込み…',
    searchHub: 'ハブを検索（コミュニティと既知のソース）…',
    noMcpServers: '設定済みまたはカタログ内の MCP サーバーはありません。'
  },
  screen: {
    title: '画面',
    menu: '画面を開く',
    unsupportedTitle: 'このホストにはボット画面がありません',
    unsupportedBody:
      'ボット画面は Linux のゲートウェイホストで動作します。このボットはホスト自身のディスプレイを使います。',
    notInstalledTitle: '画面パッケージが不足しています',
    notInstalledBody:
      'このボットに画面を与えるには、ゲートウェイホストに TigerVNC と Xfce コアが必要です。ホストで実行:',
    installHint:
      'Hermes を実行しているユーザーとしてゲートウェイホスト上で実行されます。sudo は Hermes 経由で一度だけ求められます。',
    install: 'ホストにインストール',
    installing: 'インストール中…',
    installCancelled: 'インストールを中止しました: sudo パスワードが入力されませんでした。',
    installFailed: 'インストールに失敗しました。上のログを確認するか、ホストでコマンドを直接実行してください。',
    noPackageManager: 'ゲートウェイホストに対応するパッケージマネージャー (apt, dnf, pacman) が見つかりません。',
    portalTitle: 'スクリーン',
    portalOpen: '開く',
    heroStopped: '画面は停止中',
    heroNotInstalled: 'このホストには未インストール',
    heroConnecting: '画面を確認中…',
    heroStale: '最終表示 — 画面に接続できません',
    heroSuppressed: '他の人が操作中は非表示',
    heroOpenLive: 'ライブで開く',
    heroInstall: 'インストール',
    heroStart: '開始',
    portalWatching: 'ライブ · ボットが操作中',
    portalYouControl: 'ライブ · あなたが操作中',
    portalOtherControls: 'ライブ · 別のビューアーが操作中',
    portalStopped: '停止中',
    portalNotInstalled: 'ホストに未インストール',
    portalUnsupported: 'このホストでは利用できません',
    portalUnavailable: 'Screen を使うにはボットの Hermes を更新してください',
    unavailableTitle: 'Screen には新しい Hermes が必要です',
    autoOpenMenu: 'ボットが画面を使い始めたら Screen を開く',
    autoOpenOnToast: name => `${name} がデスクトップを使い始めると Screen が開きます`,
    autoOpenOffToast: name => `${name} の Screen は手動で開くまで閉じたままです`,
    stoppedTitle: '画面はオフです',
    stoppedBody: 'このボットのデスクトップを起動すると、動作を見守り、必要なときに操作を引き継げます。',
    start: '画面を起動',
    attaching: '画面に接続中…',
    streamLost: '画面ストリームが終了しました',
    reconnect: '再接続',
    takeOver: '引き継ぐ',
    handBack: '戻す',
    handBackForce: '強制的に戻す',
    handBackForceHint: 'もう存在しないビューア（再読み込み後など）が保持しているリースを解放します。',
    openNeedsUpdate: 'ボットの画面を開くには Hermes Desktop を更新してください。',
    youControl: 'あなたが操作中',
    otherControls: '別のビューアが操作中',
    agentControls: 'ボットが操作中',
    controlTaken: '別のビューアが操作を引き継ぎました。閲覧のみ。'
  },
  cron: {
    untitled: '無題のジョブ',
    nameNul: 'ジョブ名に NUL (U+0000) は使用できません。',
    instructionNul: 'ジョブの指示に NUL (U+0000) は使用できません。',
    minutesFromNow: '分後',
    hoursFromNow: '時間後',
    daysFromNow: '日後',
    stopAfter: '実行上限',
    runsHint: '回（空欄で無制限）',
    detailDescription: 'このジョブの実行内容と次回の実行日時。',
    status: '状態',
    active: '有効',
    paused: '一時停止',
    schedule: 'スケジュール',
    rawSchedule: 'スケジュール（元の値）',
    repeat: '繰り返し',
    nextRun: '次回実行',
    overdueSince: '実行予定超過',
    lastRun: '前回実行',
    lastResult: '前回の結果',
    workdir: '作業ディレクトリ',
    succeeded: '成功',
    failed: '失敗',
    deliveryFailed: '実行済みですが、配信に失敗しました',
    blockedConfig: '設定によりブロック（未実行）',
    legacyUnsafe: '安全のため一時停止中です。この旧形式のジョブを削除して作り直してから実行してください。',
    filterHint:
      'このプロファイルには定期実行ジョブがありますが、このボット向けのタグが付いたものはありません。ジョブ名を「[bot:<名前>] …」にするとここに表示されます。下のCronでも確認できます。',
    needsRosterFirst: 'このボットは先に名簿に表示される必要があります。',
    staleNotice: '定期実行ジョブを更新できませんでした。最後に取得したリストを表示しています。',
    readFailure: 'リストはまだ存在している可能性があります — これは読み取りの失敗で、削除ではありません。',
    createDesc: bot => `${bot}がスケジュールに沿って実行する定期タスクです。実行結果は専用のチャット履歴に残ります。`,
    instruction: '指示',
    whenToRun: '実行するタイミング',
    dayOfMonth: '日付',
    sendResultsTo: '結果の送信先',
    runHistoryOnly: '実行履歴のみ',
    botChatTarget: bot => `${bot}のチャット（ボットが応答）`,
    continuity: '継続: 各実行が前回の出力を参照します（重複を避け、続きから実行）',
    onceIn: when => `1回のみ（${when}）`,
    everyNDays: days => `${days}日ごと`,
    everyNHours: hours => `${hours}時間ごと`,
    everyNMinutes: minutes => `${minutes}分ごと`,
    freqOnce: '1回のみ、…後',
    freqHourly: '毎時',
    freqDaily: '毎日',
    freqWeekdays: '平日',
    freqWeekly: '毎週',
    freqMonthly: '毎月',
    freqInterval: '間隔',
    freqAdvanced: '詳細…',
    unitMinutes: '分',
    unitHours: '時間',
    unitDays: '日',
    runsOnce: (count, unit) => `今から${count}${unit}後に1回実行します`,
    runsHourly: '毎時0分に実行します',
    runsDaily: time => `毎日${time}に実行します`,
    runsWeekdays: time => `月曜〜金曜の${time}に実行します`,
    runsWeekly: (day, time) => `毎週${day}の${time}に実行します`,
    runsMonthly: (day, time) => `毎月${day}日の${time}に実行します`,
    runsInterval: (count, unit) => `${count}${unit}ごとに実行します`,
    runsRaw: '生のスケジュール — Nm/Nh/Nd または5フィールドのcron',
    timesTotal: count => `、合計${count}回`
  }
}

const zh: BotsMessages = {
  editor: {
    fullConfigHint: '完整配置需要更新网关（更新 Hermes 后请重启网关）。',
    liveCapabilities: '功能（立即生效 — 技能、工具、MCP）',
    editSoul: 'SOUL.md（人格 + 智能体消息协议）',
    remoteCapabilitiesHint: '远程功能需要更新桌面应用。模型和 SOUL 的更改会在保存后生效。',
    skillsEnabled: (enabled, total) => `技能（已启用 ${enabled}/${total}）`,
    toolsetsEnabled: (enabled, total) => `工具集（已启用 ${enabled}/${total} — 全部取消勾选可恢复默认值）`,
    mcpServers: 'MCP 服务器',
    providerCustom: '提供商（自定义）',
    modelCustom: '模型（自定义）',
    backToDropdowns: '← 返回下拉选项',
    inheritLaunch: '继承（启动配置文件）',
    enterManually: '✏️ 手动输入…',
    gatewayDefault: '网关默认值',
    modelNameExample: '例如：模型名称',
    modelSwitchFailed: '模型切换失败',
    newDescription: '拥有独立记忆、技能和聊天的具名队友，可以与你的其他智能体互发消息。',
    name: '名称',
    title: '显示名称',
    description: '描述',
    createOn: '创建位置',
    general: '常规',
    capabilities: '能力',
    skills: '技能',
    tools: '工具',
    cloneFrom: '从配置档案克隆',
    freshProfile: '新配置档案（内置技能）',
    inheritedModel: '继承启动时的配置档案',
    soul: 'SOUL.md（可选，将替换生成的人格）',
    shareKeys: '与主配置档案共享密钥和账户',
    shareKeysOn: target => `与 ${target} 上的 default 配置档案共享密钥和账户`,
    shareKeysHint:
      '订阅、OAuth 登录和 API 密钥保持共享而非复制，令牌刷新不会使另一方失效。取消勾选则创建隔离的快照副本。',
    needsModel: '它还没有可用的模型提供商，因此跳过了自我介绍。请在“高级”中选择提供商和模型。',
    configureModel: '配置模型',
    createEmpty: '创建空配置（跳过内置技能）',
    nameTakenHint: '此名称已被占用，请先选择其他名称再配置能力。',
    nameFirstHint: '请先为机器人命名，打开此标签页时将创建草稿配置档案（取消时会丢弃）。',
    newerDesktop: '技能功能需要更新的 Hermes Desktop。',
    newerGateway: '能力目录需要更新的网关（更新 Hermes 后请重启网关）。',
    emptySkillsHint: '已勾选“创建空配置”，不会安装内置技能。',
    defaultToolsHint: '全部勾选或全部不选将保留默认工具集行为。',
    catalog: '目录',
    catalogInstalled: '目录 · 已安装',
    mcpHint:
      '已配置的服务器从主配置档案复制；目录条目来自内置 MCP 菜单。需要 API 密钥的条目先进行设置（凭据遵循共享密钥设置）。',
    creating: '正在创建…',
    createBot: '创建机器人',
    auto: '自动',
    autoHint: '自动 — 由名称决定',
    unlock: '解锁',
    lockFace: '锁定外观',
    lockedHint: '外观已锁定，重命名不会改变外观。',
    unlockedHint: '外观随名称变化。',
    noImageModel: '没有可用的图像模型。如果刚启用模型或更新了 Hermes，请重启网关：Ctrl+K →“重启网关”。',
    checkingImage: '正在检查图像后端…',
    chooseImage: '选择图像…',
    editDescription: (name, profile) => `${name}（${profile}）的外观和职责。`,
    nameTaken: name => `名为“${name}”的智能体已存在。`,
    nameTakenOn: (name, target) => `${target} 上已存在名为“${name}”的智能体。`,
    currentConnection: name => `${name}（当前）`,
    remoteHint: target => `智能体将在 ${target} 上创建，并作为连接机器人显示在名册中。聊天将路由到该机器。`,
    cloneFromOn: target => `从配置档案克隆（位于 ${target}）`,
    catalogHint: source => `目录来自 ${source}，未勾选的技能将在创建后禁用。`,
    sectionsFailed: sections => `部分设置失败：${sections}`,
    updated: name => `已更新 ${name}`,
    created: name => `已创建机器人“${name}”`,
    createdOn: (name, target) => `已在 ${target} 上创建机器人“${name}”`
  },
  roster: {
    search: '搜索机器人和群聊',
    searchPlaceholder: '搜索机器人和群聊…',
    newBotOrGroup: '新建机器人或群聊',
    groupChats: '群聊',
    emptyTitle: '还没有机器人',
    emptyDesc: '创建你的第一个机器人。',
    noMatchQuery: query => `没有机器人或群聊匹配“${query}”`,
    noMatchQueryOn: (query, gateway) => `${gateway} 上没有机器人或群聊匹配“${query}”`,
    noMatchFiltersOn: gateway => `${gateway} 上没有机器人或群聊匹配这些筛选条件`,
    noMatchFilters: '没有机器人或群聊匹配这些筛选条件。',
    clearFilters: '清除筛选',
    allHidden: '所有机器人都已隐藏',
    allHiddenDesc: '它们会继续运行，并保留各自的历史。',
    showHidden: '显示已隐藏的机器人',
    noHiddenMatch: '没有已隐藏的机器人匹配这些筛选条件。',
    hiddenFromRoster: '已从名单中隐藏',
    pinned: '已置顶',
    needsAttention: '需要处理',
    needsInput: '需要你输入',
    botsAndGroups: '机器人和群聊',
    botsOnly: '仅机器人',
    groupsOnly: '仅群聊',
    anyActivity: '任何活动',
    activeNow: '正在活动',
    recentlyActive: '最近活跃',
    older: '更早',
    gatewayRemoved: '网关已移除',
    onDemand: '按需',
    ready: '就绪',
    statusUnknown: '状态未知',
    unavailable: '不可用',
    retryNow: '立即重试',
    rosterUnavailable: reason => `无法获取名单：${reason}。如果网关早于 profiles.list，请更新 Hermes 并重启网关。`,
    waitingForGateway: '正在等待网关连接…（远程网关可能需要几秒；会自动重试）'
  },
  sections: {
    newSection: '新建分区',
    newTitle: '新建分区',
    renameTitle: '重命名分区',
    nameLabel: '分区名称',
    namePlaceholder: '例如：客户',
    create: '创建',
    rename: '重命名…',
    moveUp: '上移',
    moveDown: '下移',
    unassigned: '未分类',
    options: name => `${name} 分区选项`,
    headingTip: '将机器人拖放到此处 · 双击重命名',
    emptyHint: '将机器人拖到此处',
    moveTo: '移动到分区',
    newSectionEllipsis: '新建分区…',
    removeFromSection: '移出分区',
    deleted: (name, count) => (count === 0 ? `已删除“${name}”` : `已删除“${name}” — ${count} 个机器人已移至未分类`),
    undo: '撤销'
  },
  bot: {
    newTitle: '新建机器人',
    editTitle: '编辑配置档案',
    editMenu: '编辑…',
    helpPromptPlaceholder: '这个机器人应该帮你做什么？',
    descriptionHint: '留空则根据机器人的名称和描述生成。',
    newChatWith: '与此机器人开新聊天',
    openBotChat: '打开机器人聊天',
    openingChat: '正在打开聊天…',
    pinToTop: '置顶',
    unpin: '取消置顶',
    pinnedToast: name => `已将 ${name} 置顶`,
    unpinnedToast: name => `已取消置顶 ${name}`,
    hide: '隐藏',
    unhide: '取消隐藏',
    hiddenToast: name => `已隐藏 ${name} — 点击机器人标题栏的眼睛按钮可查看隐藏的机器人`,
    unhiddenToast: name => `${name} 已回到列表`,
    groupsMenu: groups => `群聊：${groups}…`,
    manageGroups: '管理群聊…',
    metadataLoadFailed: '无法加载机器人元数据',
    loadFailed: '无法加载机器人',
    groupsLoadFailed: '无法加载机器人的群聊',
    thisDevice: '本设备',
    attentionFallback: '需要处理',
    attentionProviderAuth: '请为此配置档案重新登录',
    attentionQuota: '配额或余额已用尽',
    attentionMissingConfig: '未配置提供商 — 请运行 hermes model',
    attentionBlocked: '机器人已被阻止 — 请查看其最后一条消息',
    duplicate: '复制',
    duplicateFailed: '复制失败',
    deleteTitle: '删除机器人和配置档案？',
    removeFromAllGroups: '从所有群组中移除',
    createFirstHint: '打开机器人面板，点击“新建机器人”。',
    createFailed: '暂时无法创建配置档案',
    advanced: '高级',
    advancedHint: '高级 — 模型、技能、工具集、SOUL.md',
    advancedFailed: '高级配置失败',
    openAnotherChatUnsupported: '请更新 Hermes Desktop 以打开另一个机器人聊天。',
    remoteConnectionsUnsupported: '请更新 Hermes Desktop 以与其他连接上的机器人聊天。',
    openNeedsUpdateTitle: '这个机器人运行在较旧的 Hermes 上',
    openNeedsUpdateMessage: connectionLabel => `请更新 ${connectionLabel}，然后重试。`,
    openUnreachableTitle: 'Hermes 无法连接到运行这个机器人的电脑',
    openUnreachableMessage: '请确认它在线后重试。',
    openChatFailedTitle: botName => `无法打开 ${botName} 的聊天`,
    openChatFailedMessage: '请重试。',
    openGateways: '打开网关',
    chatEmpty: '说点什么开始吧。',
    kickoff: '你好，介绍一下你自己吧！'
  },
  avatar: {
    classicShapes: '经典形状',
    blobFromName: '斑点脸 — 根据机器人名称绘制',
    unlockFollowsName: '解锁 — 面孔再次跟随机器人名称',
    randomize: '随机',
    tabBot: '机器人',
    tabGenerate: '生成',
    upload: '上传',
    tabPet: '宠物',
    removeImage: '移除图片，改用形状',
    removeBackToShape: '移除 — 回到形状头像',
    describePlaceholder: '描述你的头像…',
    describeHint: '留空则根据名称/标题/描述和 agent-messaging 名册自动生成。',
    matchTheName: '匹配名称',
    pickPet: '选择一只宠物作为此机器人的头像。',
    petLoadFailed: '无法加载该宠物 — 请换一只试试。',
    imageTooLarge: '图片过大（最大 15MB）。',
    generationFailed: '头像生成失败',
    savedLocally: '外观已保存在本地；远程持久化失败',
    savedLocallyDescriptionFailed: '外观已保存在本地；描述更新失败',
    generate: '生成',
    generating: '生成中…'
  },
  group: {
    newTitle: '新建群聊',
    manageDesc: '一个机器人可以加入多个群聊。成员关系会同步到每台设备。',
    manageTitle: '管理群组',
    settingsTitle: '群组设置',
    settingsDesc: '重命名群组或设置房间图片。成员和历史都会保留。',
    nameLabel: '群组名称',
    holdDetection: '检测停止指令',
    holdDetectionHint: '允许房间消息将指定成员保持暂停，直到再次提及该成员。',
    compressHistory: '压缩历史',
    compressHistoryHint: (member: string) => `压缩 ${member} 隐藏的房间历史，避免该成员因空回复而失败`,
    compressing: (member: string) => `正在压缩 ${member} 的房间历史…`,
    compressDone: (member: string, compressed: number, detail: string) =>
      `已压缩 ${member} 的 ${compressed} 个房间会话${detail ? ` — ${detail}` : ''}`,
    compressNothing: (member: string) => `${member} 没有可压缩的历史 — 还没有房间会话`,
    compressFailed: (member: string, error: string) => `无法压缩 ${member} 的房间历史: ${error}`,
    searchToAdd: '搜索要添加的机器人',
    searchToAddPlaceholder: '搜索要添加的机器人…',
    removeFromSelection: '从选择中移除',
    disbandTitle: '解散群聊？',
    deleteTitle: '删除群聊？',
    deleteAction: '删除',
    composerPlaceholder: '说点什么 — 这个群里的每个机器人都会听到。',
    slashCommandsUnsupported: '群聊不支持斜杠命令。请打开单个机器人的聊天来使用。',
    attachHint: '附加文件 — 每个回应的机器人都能看到',
    newThread: '新帖子',
    reply: '回复',
    replyInThread: '在帖子中回复',
    replyInThreadPlaceholder: '在帖子中回复…',
    openThread: '打开此帖子',
    collapseThread: '收起帖子',
    collapseThreadLabel: '收起此帖子',
    activity: '活动',
    noActivityYet: '本回合还没有活动。',
    showActivity: '显示房间活动',
    hideActivity: '隐藏房间活动',
    stop: '停止',
    stopHint: '停止本次运行 — 中断当前回合的成员，并暂停其余成员',
    allHeldStatus: count => `全部 ${count} 个机器人已暂停`,
    heldMembersStatus: members => `已暂停：${members}`,
    holdReleaseHint: '提及已暂停的机器人，或发送 @all resume 以恢复它们。',
    needsYourInput: '此群聊中有机器人需要你输入',
    noMembersToSend: group => `${group} 没有可发送的成员——请添加机器人，如果成员仍在加载，请重新打开该群聊。`,
    pictureGenerationFailed: '群组图片生成失败',
    nameTaken: name => `已存在名为“${name}”的群聊。`,
    memberCount: count => `${count} 个机器人`,
    you: '你',
    availableCount: (available, total) => `${total} 个中 ${available} 个可用`,
    settingsHint: group => `群聊设置 — 重命名 ${group} 或设置房间图片`,
    settingsLabel: group => `${group} 的群聊设置`,
    disbandHint: group => `解散 ${group} 群聊`,
    disbandLabel: group => `解散 ${group}`,
    disbandAction: '解散',
    disbanding: '正在解散…',
    disbandDone: '已解散',
    disbanded: group => `已解散“${group}”`,
    disbandDescPrefix: '',
    disbandDescSuffix: count =>
      ` 的分组将从 ${count} 个机器人中移除，并清空共享房间日志。机器人本身及其各群聊会话都会保留。`,
    stopped: group => `已停止 ${group} — 其余轮次将保留到你恢复为止`,
    removeAttachment: '移除附件',
    threadFallback: '讨论串',
    replyCount: replies => `${replies} 条回复`,
    dropToThread: '拖放以附加到此讨论串回复',
    dropToRoom: '拖放以附加 — 每个回应的机器人都能看到',
    waitingForAnswer: '等待你的回答…',
    memberThinking: name => `${name} 正在思考…`,
    roomWorking: '房间正在处理…',
    messageRoom: group => `发消息给 ${group}`,
    newThreadPlaceholder: group => `在 ${group} 中开启新讨论串…（@名称指定，@everyone 全体）`,
    everyoneMeta: '房间里的所有机器人',
    commandApproval: '命令批准',
    answerFailed: (handle, error) => `无法将回答发送给 @${handle}：${error}`,
    wantsToRunCommand: handle => `@${handle} 想执行一个命令：`,
    asks: handle => `@${handle} 的提问：`,
    answerTo: member => `回答 @${member}`
  },
  tools: {
    installHint: name => `安装“${name}”并添加到上方列表`,
    installed: name => `技能“${name}”已安装`,
    installFailed: name => `安装“${name}”失败`,
    searchHint: '正在搜索社区和常用来源 — 可能需要约 10 秒…',
    resizeHint: '拖动角落可调整大小。',
    addServerFailed: '无法添加服务器',
    noTarget: '没有目标配置文件',
    setKeyFailed: key => `无法设置 ${key}`,
    configured: name => `${name} 已配置`,
    authenticated: name => `${name} 已验证身份`,
    testFailed: '配置后的服务器测试失败',
    completeSignIn: '请在浏览器中完成登录…',
    needsSetup: keys => `需要设置（${keys}）— 重启网关以启用应用内设置`,
    setUpDone: '已设置 ✓',
    saveTest: '保存并测试',
    authorizing: '正在授权…',
    working: '正在处理…',
    setupFailed: '设置失败',
    signIn: '登录…',
    setUp: '设置…',
    skillsHub: 'Hermes 技能中心',
    filterSkills: '筛选技能…',
    searchHub: '搜索技能中心（社区和常见来源）…',
    noMcpServers: '未配置 MCP 服务器，目录中也没有。'
  },
  screen: {
    title: '屏幕',
    menu: '打开屏幕',
    unsupportedTitle: '此主机没有机器人屏幕',
    unsupportedBody: '机器人屏幕在 Linux 网关主机上运行。此机器人使用主机自身的显示器。',
    notInstalledTitle: '缺少屏幕软件包',
    notInstalledBody: '网关主机需要 TigerVNC 和 Xfce 核心组件才能为此机器人提供屏幕。在主机上运行:',
    installHint: '在网关主机上以运行 Hermes 的用户身份执行；sudo 只会通过 Hermes 询问一次。',
    install: '安装到主机',
    installing: '正在安装…',
    installCancelled: '安装已取消：未提供 sudo 密码。',
    installFailed: '安装失败。请查看上方日志，或在主机上手动运行该命令。',
    noPackageManager: '网关主机上未找到受支持的包管理器（apt、dnf、pacman）。',
    portalTitle: '屏幕',
    portalOpen: '打开',
    heroStopped: '屏幕已关闭',
    heroNotInstalled: '此主机未安装',
    heroConnecting: '正在检查屏幕…',
    heroStale: '最后画面 — 屏幕无法访问',
    heroSuppressed: '有人控制时隐藏',
    heroOpenLive: '实时打开',
    heroInstall: '安装',
    heroStart: '启动',
    portalWatching: '直播 · 机器人控制中',
    portalYouControl: '直播 · 你在控制',
    portalOtherControls: '直播 · 其他查看者控制中',
    portalStopped: '已停止',
    portalNotInstalled: '主机未安装',
    portalUnsupported: '此主机不可用',
    portalUnavailable: '更新机器人的 Hermes 以使用屏幕',
    unavailableTitle: '屏幕需要更新版的 Hermes',
    autoOpenMenu: '机器人使用屏幕时自动打开',
    autoOpenOnToast: name => `${name} 开始使用桌面时会自动打开屏幕`,
    autoOpenOffToast: name => `${name} 的屏幕将保持关闭，直到你手动打开`,
    stoppedTitle: '屏幕已关闭',
    stoppedBody: '启动此机器人的桌面，观看它的操作，并在需要时接管。',
    start: '启动屏幕',
    attaching: '正在连接屏幕…',
    streamLost: '屏幕流已结束',
    reconnect: '重新连接',
    takeOver: '接管',
    handBack: '交还',
    handBackForce: '强制交还',
    handBackForceHint: '释放已不在场的查看者（例如重新加载后）持有的控制权。',
    openNeedsUpdate: '更新 Hermes Desktop 以打开机器人屏幕。',
    youControl: '你正在控制',
    otherControls: '另一位查看者正在控制',
    agentControls: '机器人正在控制',
    controlTaken: '另一位查看者已接管控制。仅可观看。'
  },
  cron: {
    untitled: '未命名任务',
    nameNul: '任务名称不能包含 NUL (U+0000)。',
    instructionNul: '任务指令不能包含 NUL (U+0000)。',
    minutesFromNow: '分钟后',
    hoursFromNow: '小时后',
    daysFromNow: '天后',
    stopAfter: '运行上限',
    runsHint: '次（留空则持续运行）',
    detailDescription: '此任务的内容和下次运行时间。',
    status: '状态',
    active: '运行中',
    paused: '已暂停',
    schedule: '计划',
    rawSchedule: '计划（原始值）',
    repeat: '重复',
    nextRun: '下次运行',
    overdueSince: '逾期起始时间',
    lastRun: '上次运行',
    lastResult: '上次结果',
    workdir: '工作目录',
    succeeded: '成功',
    failed: '失败',
    deliveryFailed: '已运行，但发送失败',
    blockedConfig: '配置阻止了运行（未执行）',
    legacyUnsafe: '为安全起见已暂停：请删除并重新创建此旧任务，然后再运行。',
    filterHint:
      '此配置档案中有定时任务，但没有一个标记给这个机器人。将任务命名为“[bot:<名称>] …”即可显示在这里，也可以在下方的 Cron 中查看。',
    needsRosterFirst: '这个机器人需要先出现在名册中。',
    staleNotice: '无法刷新定时任务。显示的是上一次获取的列表。',
    readFailure: '列表可能仍然存在 — 这是一次读取失败，不是删除。',
    createDesc: bot => `由 ${bot} 按计划运行的重复任务。运行结果会保存在它自己的聊天记录中。`,
    instruction: '指令',
    whenToRun: '运行时间',
    dayOfMonth: '每月日期',
    sendResultsTo: '结果发送到',
    runHistoryOnly: '仅运行历史',
    botChatTarget: bot => `${bot} 的聊天（机器人会回应）`,
    continuity: '连续性：每次运行都能看到上次的输出（去重，从上次的地方继续）',
    onceIn: when => `一次（${when}）`,
    everyNDays: days => `每 ${days} 天`,
    everyNHours: hours => `每 ${hours} 小时`,
    everyNMinutes: minutes => `每 ${minutes} 分钟`,
    freqOnce: '一次，在…之后',
    freqHourly: '每小时',
    freqDaily: '每天',
    freqWeekdays: '工作日',
    freqWeekly: '每周',
    freqMonthly: '每月',
    freqInterval: '间隔',
    freqAdvanced: '高级…',
    unitMinutes: '分钟',
    unitHours: '小时',
    unitDays: '天',
    runsOnce: (count, unit) => `从现在起 ${count} ${unit}后运行一次`,
    runsHourly: '每小时整点运行',
    runsDaily: time => `每天 ${time} 运行`,
    runsWeekdays: time => `周一至周五 ${time} 运行`,
    runsWeekly: (day, time) => `每${day} ${time} 运行`,
    runsMonthly: (day, time) => `每月 ${day} 日 ${time} 运行`,
    runsInterval: (count, unit) => `每 ${count} ${unit}运行`,
    runsRaw: '原始计划 — every Nm/Nh/Nd 或 5 段 cron',
    timesTotal: count => `，共 ${count} 次`
  }
}

const zhHant: BotsMessages = {
  editor: {
    fullConfigHint: '完整設定需要更新閘道（更新 Hermes 後請重新啟動閘道）。',
    liveCapabilities: '功能（立即生效 — 技能、工具、MCP）',
    editSoul: 'SOUL.md（人格 + 智慧代理訊息協定）',
    remoteCapabilitiesHint: '遠端功能需要更新桌面應用程式。模型和 SOUL 的變更會在儲存後生效。',
    skillsEnabled: (enabled, total) => `技能（已啟用 ${enabled}/${total}）`,
    toolsetsEnabled: (enabled, total) => `工具集（已啟用 ${enabled}/${total} — 全部取消勾選可還原預設值）`,
    mcpServers: 'MCP 伺服器',
    providerCustom: '供應商（自訂）',
    modelCustom: '模型（自訂）',
    backToDropdowns: '← 返回下拉選項',
    inheritLaunch: '繼承（啟動設定檔）',
    enterManually: '✏️ 手動輸入…',
    gatewayDefault: '閘道預設值',
    modelNameExample: '例如：模型名稱',
    modelSwitchFailed: '模型切換失敗',
    newDescription: '擁有獨立記憶、技能和聊天的具名隊友，可以與你的其他智慧代理互傳訊息。',
    name: '名稱',
    title: '顯示名稱',
    description: '描述',
    createOn: '建立位置',
    general: '一般',
    capabilities: '功能',
    skills: '技能',
    tools: '工具',
    cloneFrom: '從設定檔複製',
    freshProfile: '新設定檔（內建技能）',
    inheritedModel: '繼承啟動時的設定檔',
    soul: 'SOUL.md（選填，將取代產生的人格）',
    shareKeys: '與主要設定檔共用金鑰和帳戶',
    shareKeysOn: target => `與 ${target} 上的 default 設定檔共用金鑰和帳戶`,
    shareKeysHint:
      '訂閱、OAuth 登入和 API 金鑰保持共用而非複製，權杖更新不會使另一方失效。取消勾選則建立隔離的快照副本。',
    needsModel: '它還沒有可用的模型提供者，因此略過了自我介紹。請在「進階」中選擇提供者和模型。',
    configureModel: '設定模型',
    createEmpty: '建立空白設定（略過內建技能）',
    nameTakenHint: '此名稱已被使用，請先選擇其他名稱再設定功能。',
    nameFirstHint: '請先為機器人命名，開啟此分頁時將建立草稿設定檔（取消時會捨棄）。',
    newerDesktop: '技能功能需要更新的 Hermes Desktop。',
    newerGateway: '功能目錄需要更新的閘道（更新 Hermes 後請重新啟動閘道）。',
    emptySkillsHint: '已勾選「建立空白設定」，不會安裝內建技能。',
    defaultToolsHint: '全部勾選或全部不選將保留預設工具集行為。',
    catalog: '目錄',
    catalogInstalled: '目錄 · 已安裝',
    mcpHint:
      '已設定的伺服器從主要設定檔複製；目錄項目來自內建 MCP 選單。需要 API 金鑰的項目先進行設定（憑證遵循共用金鑰設定）。',
    creating: '正在建立…',
    createBot: '建立機器人',
    auto: '自動',
    autoHint: '自動 — 由名稱決定',
    unlock: '解鎖',
    lockFace: '鎖定外觀',
    lockedHint: '外觀已鎖定，重新命名不會改變外觀。',
    unlockedHint: '外觀隨名稱變化。',
    noImageModel: '沒有可用的影像模型。如果剛啟用模型或更新了 Hermes，請重新啟動閘道：Ctrl+K →「重新啟動閘道」。',
    checkingImage: '正在檢查影像後端…',
    chooseImage: '選擇影像…',
    editDescription: (name, profile) => `${name}（${profile}）的外觀和職責。`,
    nameTaken: name => `名為「${name}」的智慧代理已存在。`,
    nameTakenOn: (name, target) => `${target} 上已存在名為「${name}」的智慧代理。`,
    currentConnection: name => `${name}（目前）`,
    remoteHint: target => `智慧代理將在 ${target} 上建立，並作為連線機器人顯示在名冊中。聊天將路由到該機器。`,
    cloneFromOn: target => `從設定檔複製（位於 ${target}）`,
    catalogHint: source => `目錄來自 ${source}，未勾選的技能將在建立後停用。`,
    sectionsFailed: sections => `部分設定失敗：${sections}`,
    updated: name => `已更新 ${name}`,
    created: name => `已建立機器人「${name}」`,
    createdOn: (name, target) => `已在 ${target} 上建立機器人「${name}」`
  },
  roster: {
    search: '搜尋機器人和群組聊天',
    searchPlaceholder: '搜尋機器人和群組聊天…',
    newBotOrGroup: '新增機器人或群組聊天',
    groupChats: '群組聊天',
    emptyTitle: '還沒有機器人',
    emptyDesc: '建立你的第一個機器人。',
    noMatchQuery: query => `沒有機器人或群組聊天符合「${query}」`,
    noMatchQueryOn: (query, gateway) => `${gateway} 上沒有機器人或群組聊天符合「${query}」`,
    noMatchFiltersOn: gateway => `${gateway} 上沒有機器人或群組聊天符合這些篩選條件`,
    noMatchFilters: '沒有機器人或群組聊天符合這些篩選條件。',
    clearFilters: '清除篩選',
    allHidden: '所有機器人都已隱藏',
    allHiddenDesc: '它們會繼續運作，並保留各自的歷史。',
    showHidden: '顯示已隱藏的機器人',
    noHiddenMatch: '沒有已隱藏的機器人符合這些篩選條件。',
    hiddenFromRoster: '已從名單中隱藏',
    pinned: '已釘選',
    needsAttention: '需要處理',
    needsInput: '需要您的輸入',
    botsAndGroups: '機器人和群組聊天',
    botsOnly: '僅機器人',
    groupsOnly: '僅群組聊天',
    anyActivity: '任何活動',
    activeNow: '目前活躍',
    recentlyActive: '最近活躍',
    older: '更早',
    gatewayRemoved: '閘道已移除',
    onDemand: '隨需',
    ready: '就緒',
    statusUnknown: '狀態未知',
    unavailable: '不可用',
    retryNow: '立即重試',
    rosterUnavailable: reason => `無法取得名單：${reason}。如果閘道早於 profiles.list，請更新 Hermes 並重新啟動閘道。`,
    waitingForGateway: '正在等待閘道連線…（遠端閘道可能需要幾秒；會自動重試）'
  },
  sections: {
    newSection: '新增分區',
    newTitle: '新增分區',
    renameTitle: '重新命名分區',
    nameLabel: '分區名稱',
    namePlaceholder: '例如：客戶',
    create: '建立',
    rename: '重新命名…',
    moveUp: '上移',
    moveDown: '下移',
    unassigned: '未分類',
    options: name => `${name} 分區選項`,
    headingTip: '將機器人拖放到此處 · 雙擊重新命名',
    emptyHint: '將機器人拖到此處',
    moveTo: '移動到分區',
    newSectionEllipsis: '新增分區…',
    removeFromSection: '移出分區',
    deleted: (name, count) => (count === 0 ? `已刪除「${name}」` : `已刪除「${name}」— ${count} 個機器人已移至未分類`),
    undo: '復原'
  },
  bot: {
    newTitle: '新增機器人',
    editTitle: '編輯設定檔',
    editMenu: '編輯…',
    helpPromptPlaceholder: '這個機器人應該幫你做什麼？',
    descriptionHint: '留空則依機器人的名稱和描述產生。',
    newChatWith: '與此機器人開新聊天',
    openBotChat: '開啟機器人聊天',
    openingChat: '正在開啟聊天…',
    pinToTop: '釘選到頂端',
    unpin: '取消釘選',
    pinnedToast: name => `已將 ${name} 釘選到頂端`,
    unpinnedToast: name => `已取消釘選 ${name}`,
    hide: '隱藏',
    unhide: '取消隱藏',
    hiddenToast: name => `已隱藏 ${name} — 點擊機器人標題列的眼睛按鈕可查看隱藏的機器人`,
    unhiddenToast: name => `${name} 已回到名單`,
    groupsMenu: groups => `群組：${groups}…`,
    manageGroups: '管理群組…',
    metadataLoadFailed: '無法載入機器人中繼資料',
    loadFailed: '無法載入機器人',
    groupsLoadFailed: '無法載入機器人的群組',
    thisDevice: '本裝置',
    attentionFallback: '需要處理',
    attentionProviderAuth: '請為此設定檔重新登入',
    attentionQuota: '配額或餘額已用盡',
    attentionMissingConfig: '未設定供應商 — 請執行 hermes model',
    attentionBlocked: '機器人已被封鎖 — 請查看其最後一則訊息',
    duplicate: '複製',
    duplicateFailed: '複製失敗',
    deleteTitle: '刪除機器人和設定檔？',
    removeFromAllGroups: '從所有群組中移除',
    createFirstHint: '開啟機器人面板，點「新增機器人」。',
    createFailed: '暫時無法建立設定檔',
    advanced: '進階',
    advancedHint: '進階 — 模型、技能、工具集、SOUL.md',
    advancedFailed: '進階設定失敗',
    openAnotherChatUnsupported: '請更新 Hermes Desktop 以開啟另一個機器人聊天。',
    remoteConnectionsUnsupported: '請更新 Hermes Desktop 以與其他連線上的機器人聊天。',
    openNeedsUpdateTitle: '這個機器人運行在較舊的 Hermes 上',
    openNeedsUpdateMessage: connectionLabel => `請更新 ${connectionLabel}，然後再試一次。`,
    openUnreachableTitle: 'Hermes 無法連線到運行這個機器人的電腦',
    openUnreachableMessage: '請確認它在線上後再試一次。',
    openChatFailedTitle: botName => `無法開啟 ${botName} 的聊天`,
    openChatFailedMessage: '請再試一次。',
    openGateways: '開啟閘道',
    chatEmpty: '說點什麼開始吧。',
    kickoff: '你好，介紹一下你自己吧！'
  },
  avatar: {
    classicShapes: '經典形狀',
    blobFromName: '斑點臉 — 依機器人名稱繪製',
    unlockFollowsName: '解鎖 — 面孔再次跟隨機器人名稱',
    randomize: '隨機',
    tabBot: '機器人',
    tabGenerate: '生成',
    upload: '上傳',
    tabPet: '寵物',
    removeImage: '移除圖片，改用形狀',
    removeBackToShape: '移除 — 回到形狀頭像',
    describePlaceholder: '描述你的頭像…',
    describeHint: '留空則依名稱／標題／描述與 agent-messaging 名冊自動產生。',
    matchTheName: '符合名稱',
    pickPet: '選擇一隻寵物作為此機器人的頭像。',
    petLoadFailed: '無法載入該寵物 — 請換一隻試試。',
    imageTooLarge: '圖片過大（最大 15MB）。',
    generationFailed: '頭像產生失敗',
    savedLocally: '外觀已儲存在本機；遠端持久化失敗',
    savedLocallyDescriptionFailed: '外觀已儲存在本機；描述更新失敗',
    generate: '生成',
    generating: '生成中…'
  },
  group: {
    newTitle: '新增群組聊天',
    manageDesc: '一個機器人可以加入多個群組聊天。成員關係會同步到每台裝置。',
    manageTitle: '管理群組',
    settingsTitle: '群組設定',
    settingsDesc: '重新命名群組或設定房間圖片。成員和歷史都會保留。',
    nameLabel: '群組名稱',
    holdDetection: '偵測停止指令',
    holdDetectionHint: '允許房間訊息暫停指定成員，直到再次提及該成員。',
    compressHistory: '壓縮歷史',
    compressHistoryHint: (member: string) => `壓縮 ${member} 隱藏的房間歷史，避免該成員因空回覆而失敗`,
    compressing: (member: string) => `正在壓縮 ${member} 的房間歷史…`,
    compressDone: (member: string, compressed: number, detail: string) =>
      `已壓縮 ${member} 的 ${compressed} 個房間會話${detail ? ` — ${detail}` : ''}`,
    compressNothing: (member: string) => `${member} 沒有可壓縮的歷史 — 還沒有房間會話`,
    compressFailed: (member: string, error: string) => `無法壓縮 ${member} 的房間歷史: ${error}`,
    searchToAdd: '搜尋要加入的機器人',
    searchToAddPlaceholder: '搜尋要加入的機器人…',
    removeFromSelection: '從選取中移除',
    disbandTitle: '解散群組聊天？',
    deleteTitle: '刪除群組聊天？',
    deleteAction: '刪除',
    composerPlaceholder: '說點什麼 — 這個群組裡的每個機器人都會聽到。',
    slashCommandsUnsupported: '群組聊天不支援斜線命令。請開啟個別機器人的聊天來使用。',
    attachHint: '附加檔案 — 每個回應的機器人都能看到',
    newThread: '新討論串',
    reply: '回覆',
    replyInThread: '在討論串中回覆',
    replyInThreadPlaceholder: '在討論串中回覆…',
    openThread: '開啟此討論串',
    collapseThread: '收合討論串',
    collapseThreadLabel: '收合此討論串',
    activity: '活動',
    noActivityYet: '本回合還沒有活動。',
    showActivity: '顯示房間活動',
    hideActivity: '隱藏房間活動',
    stop: '停止',
    stopHint: '停止本次執行 — 中斷目前回合的成員，並暫停其餘成員',
    allHeldStatus: count => `全部 ${count} 個機器人已暫停`,
    heldMembersStatus: members => `已暫停：${members}`,
    holdReleaseHint: '提及已暫停的機器人，或傳送 @all resume 以恢復它們。',
    needsYourInput: '此群組聊天中有機器人需要您的輸入',
    noMembersToSend: group => `${group} 沒有可傳送的成員——請新增機器人，若成員仍在載入中，請重新開啟該群組聊天。`,
    pictureGenerationFailed: '群組圖片產生失敗',
    nameTaken: name => `已存在名為「${name}」的群組聊天。`,
    memberCount: count => `${count} 個機器人`,
    you: '您',
    availableCount: (available, total) => `${total} 個中 ${available} 個可用`,
    settingsHint: group => `群組設定 — 重新命名 ${group} 或設定房間圖片`,
    settingsLabel: group => `${group} 的群組設定`,
    disbandHint: group => `解散 ${group} 群組聊天`,
    disbandLabel: group => `解散 ${group}`,
    disbandAction: '解散',
    disbanding: '正在解散…',
    disbandDone: '已解散',
    disbanded: group => `已解散「${group}」`,
    disbandDescPrefix: '',
    disbandDescSuffix: count =>
      ` 的分組將從 ${count} 個機器人中移除，並清空共享房間日誌。機器人本身及其各群組工作階段都會保留。`,
    stopped: group => `已停止 ${group} — 其餘回合將保留到你恢復為止`,
    removeAttachment: '移除附件',
    threadFallback: '討論串',
    replyCount: replies => `${replies} 則回覆`,
    dropToThread: '拖放以附加到此討論串回覆',
    dropToRoom: '拖放以附加 — 每個回應的機器人都能看到',
    waitingForAnswer: '等待你的回答…',
    memberThinking: name => `${name} 正在思考…`,
    roomWorking: '房間正在處理…',
    messageRoom: group => `傳訊息給 ${group}`,
    newThreadPlaceholder: group => `在 ${group} 中開啟新討論串…（@名稱指定，@everyone 全體）`,
    everyoneMeta: '房間裡的所有機器人',
    commandApproval: '命令核准',
    answerFailed: (handle, error) => `無法將回答傳送給 @${handle}：${error}`,
    wantsToRunCommand: handle => `@${handle} 想執行一個命令：`,
    asks: handle => `@${handle} 的提問：`,
    answerTo: member => `回覆 @${member}`
  },
  tools: {
    installHint: name => `安裝「${name}」並新增至上方清單`,
    installed: name => `技能「${name}」已安裝`,
    installFailed: name => `安裝「${name}」失敗`,
    searchHint: '正在搜尋社群和常用來源 — 可能需要約 10 秒…',
    resizeHint: '拖曳角落可調整大小。',
    addServerFailed: '無法新增伺服器',
    noTarget: '沒有目標設定檔',
    setKeyFailed: key => `無法設定 ${key}`,
    configured: name => `${name} 已設定`,
    authenticated: name => `${name} 已驗證身分`,
    testFailed: '設定後的伺服器測試失敗',
    completeSignIn: '請在瀏覽器中完成登入…',
    needsSetup: keys => `需要設定（${keys}）— 重新啟動閘道以啟用應用程式內設定`,
    setUpDone: '已設定 ✓',
    saveTest: '儲存並測試',
    authorizing: '正在授權…',
    working: '正在處理…',
    setupFailed: '設定失敗',
    signIn: '登入…',
    setUp: '設定…',
    skillsHub: 'Hermes 技能中心',
    filterSkills: '篩選技能…',
    searchHub: '搜尋技能中心（社群和常見來源）…',
    noMcpServers: '未設定 MCP 伺服器，目錄中也沒有。'
  },
  screen: {
    title: '螢幕',
    menu: '開啟螢幕',
    unsupportedTitle: '此主機沒有機器人螢幕',
    unsupportedBody: '機器人螢幕在 Linux 閘道主機上執行。此機器人使用主機自身的顯示器。',
    notInstalledTitle: '缺少螢幕套件',
    notInstalledBody: '閘道主機需要 TigerVNC 與 Xfce 核心元件才能為此機器人提供螢幕。在主機上執行:',
    installHint: '在閘道主機上以執行 Hermes 的使用者身分執行；sudo 只會透過 Hermes 詢問一次。',
    install: '安裝到主機',
    installing: '安裝中…',
    installCancelled: '安裝已取消：未提供 sudo 密碼。',
    installFailed: '安裝失敗。請查看上方日誌，或在主機上手動執行該指令。',
    noPackageManager: '閘道主機上找不到受支援的套件管理器（apt、dnf、pacman）。',
    portalTitle: '螢幕',
    portalOpen: '開啟',
    heroStopped: '螢幕已關閉',
    heroNotInstalled: '此主機未安裝',
    heroConnecting: '正在檢查螢幕…',
    heroStale: '最後畫面 — 螢幕無法連線',
    heroSuppressed: '有人控制時隱藏',
    heroOpenLive: '即時開啟',
    heroInstall: '安裝',
    heroStart: '啟動',
    portalWatching: '直播 · 機器人控制中',
    portalYouControl: '直播 · 您在控制',
    portalOtherControls: '直播 · 其他檢視者控制中',
    portalStopped: '已停止',
    portalNotInstalled: '主機未安裝',
    portalUnsupported: '此主機不可用',
    portalUnavailable: '更新機器人的 Hermes 以使用螢幕',
    unavailableTitle: '螢幕需要較新版的 Hermes',
    autoOpenMenu: '機器人使用螢幕時自動開啟',
    autoOpenOnToast: name => `${name} 開始使用桌面時會自動開啟螢幕`,
    autoOpenOffToast: name => `${name} 的螢幕將保持關閉，直到你手動開啟`,
    stoppedTitle: '螢幕已關閉',
    stoppedBody: '啟動此機器人的桌面，觀看它的操作，並在需要時接手。',
    start: '啟動螢幕',
    attaching: '正在連線至螢幕…',
    streamLost: '螢幕串流已結束',
    reconnect: '重新連線',
    takeOver: '接手',
    handBack: '交還',
    handBackForce: '強制交還',
    handBackForceHint: '釋放已不在場的檢視者（例如重新載入後）持有的控制權。',
    openNeedsUpdate: '更新 Hermes Desktop 以開啟機器人螢幕。',
    youControl: '你正在控制',
    otherControls: '另一位檢視者正在控制',
    agentControls: '機器人正在控制',
    controlTaken: '另一位檢視者已接手控制。僅可觀看。'
  },
  cron: {
    untitled: '未命名工作',
    nameNul: '工作名稱不能包含 NUL (U+0000)。',
    instructionNul: '工作指令不能包含 NUL (U+0000)。',
    minutesFromNow: '分鐘後',
    hoursFromNow: '小時後',
    daysFromNow: '天後',
    stopAfter: '執行上限',
    runsHint: '次（留空則持續執行）',
    detailDescription: '此工作的內容和下次執行時間。',
    status: '狀態',
    active: '啟用中',
    paused: '已暫停',
    schedule: '排程',
    rawSchedule: '排程（原始值）',
    repeat: '重複',
    nextRun: '下次執行',
    overdueSince: '逾期起始時間',
    lastRun: '上次執行',
    lastResult: '上次結果',
    workdir: '工作目錄',
    succeeded: '成功',
    failed: '失敗',
    deliveryFailed: '已執行，但傳送失敗',
    blockedConfig: '設定阻止了執行（未執行）',
    legacyUnsafe: '基於安全考量已暫停：請刪除並重新建立此舊工作，然後再執行。',
    filterHint:
      '此設定檔中有排程工作，但沒有任何一個標記給這個機器人。將工作命名為「[bot:<名稱>] …」即可顯示在這裡，也可以在下方的 Cron 中查看。',
    needsRosterFirst: '這個機器人需要先出現在名冊中。',
    staleNotice: '無法重新整理排程工作。顯示的是上一次取得的清單。',
    readFailure: '清單可能仍然存在 — 這是一次讀取失敗，不是刪除。',
    createDesc: bot => `由 ${bot} 按排程執行的重複工作。執行結果會保存在它自己的聊天紀錄中。`,
    instruction: '指示',
    whenToRun: '執行時間',
    dayOfMonth: '每月日期',
    sendResultsTo: '結果傳送到',
    runHistoryOnly: '僅執行紀錄',
    botChatTarget: bot => `${bot} 的聊天（機器人會回應）`,
    continuity: '連續性：每次執行都能看到上次的輸出（去重，從上次的地方繼續）',
    onceIn: when => `一次（${when}）`,
    everyNDays: days => `每 ${days} 天`,
    everyNHours: hours => `每 ${hours} 小時`,
    everyNMinutes: minutes => `每 ${minutes} 分鐘`,
    freqOnce: '一次，在…之後',
    freqHourly: '每小時',
    freqDaily: '每天',
    freqWeekdays: '工作日',
    freqWeekly: '每週',
    freqMonthly: '每月',
    freqInterval: '間隔',
    freqAdvanced: '進階…',
    unitMinutes: '分鐘',
    unitHours: '小時',
    unitDays: '天',
    runsOnce: (count, unit) => `從現在起 ${count} ${unit}後執行一次`,
    runsHourly: '每小時整點執行',
    runsDaily: time => `每天 ${time} 執行`,
    runsWeekdays: time => `週一至週五 ${time} 執行`,
    runsWeekly: (day, time) => `每${day} ${time} 執行`,
    runsMonthly: (day, time) => `每月 ${day} 日 ${time} 執行`,
    runsInterval: (count, unit) => `每 ${count} ${unit}執行`,
    runsRaw: '原始排程 — every Nm/Nh/Nd 或 5 段 cron',
    timesTotal: count => `，共 ${count} 次`
  }
}

/** Registered via `ctx.i18n.register` at plugin load (disposer tracked). */
export const BOTS_LOCALES: PluginLocaleBundles = { en, ja, zh, 'zh-hant': zhHant }

// Bind the message SHAPE to a plugin translator: string leaves resolve now,
// function leaves forward their args through t(path, …).
type Bound<T> = {
  [K in keyof T]: T[K] extends (...args: infer A) => string
    ? (...args: A) => string
    : T[K] extends object
      ? Bound<T[K]>
      : string
}

function bind<T extends object>(t: PluginTranslate, template: T, prefix = ''): Bound<T> {
  const out = {} as Record<string, unknown>

  for (const [key, value] of Object.entries(template)) {
    const path = prefix ? `${prefix}.${key}` : key
    out[key] =
      typeof value === 'function'
        ? (...args: unknown[]) => t(path, ...args)
        : value && typeof value === 'object'
          ? bind(t, value as object, path)
          : t(path)
  }

  return out as Bound<T>
}

export type BotsText = Bound<BotsMessages>

/** The Bot Mode strings for the active locale — one hook every component reads. */
export function useBots(): BotsText {
  const t = usePluginI18n('hermes-bots')

  return useMemo(() => bind(t, en), [t])
}

/** Resolve a dotted path against the English bundle — the floor for a read
 *  that beats `ctx.i18n` into existence, so an unresolved key never ships as
 *  the literal `cron.runsHourly`. */
function english(key: string, ...args: unknown[]): string {
  const leaf = key.split('.').reduce<unknown>((node, part) => (node as Record<string, unknown>)?.[part], en)

  return typeof leaf === 'function' ? (leaf as (...a: unknown[]) => string)(...args) : String(leaf ?? key)
}

let bound: { text: BotsText; translate: PluginTranslate } | null = null

/** `useBots` for the module-level functions a hook can't reach — the schedule
 *  summarizers and label helpers that render inside components but aren't
 *  components. Non-reactive on its own; every caller is invoked during a
 *  render that a core `useI18n()` already subscribes to, so a locale switch
 *  still repaints. Cached on translator identity: `bind` walks the whole tree,
 *  and these run per row. */
export function botsText(): BotsText {
  const translate = getPluginCtx()?.i18n?.t ?? english

  if (bound?.translate !== translate) {
    bound = { text: bind(translate, en), translate }
  }

  return bound.text
}
