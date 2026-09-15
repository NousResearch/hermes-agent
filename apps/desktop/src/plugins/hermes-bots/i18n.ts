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
 *    the log and compared as a sentinel (`group-activity.ts`), so translating
 *    it in place would break both. Localizing it needs the marker and its
 *    rendering split apart — worth doing, not doable as a rename.
 *
 * Locales: `en` / `ko` / `ja` / `zh` / `zh-hant`. Arabic falls through
 * the resolution chain (active locale → this plugin's `en` → the key) the
 * same way a missing string in any locale does. Nouns match core: ボット /
 * 机器人 / 機器人, プロファイル / 配置档案 / 設定檔, ゲートウェイ / 网关 / 閘道.
 */

import { type PluginLocaleBundles, type PluginTranslate, usePluginI18n } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

import { getPluginCtx } from './shared'

type BotsMessages = {
  /** Left rail: the bot + group-chat roster. */
  roster: {
    activityToastsOn: string
    activityToastsOff: string
    filterRoster: string
    activeFilters: (count: number) => string
    filterRosterActive: (count: number) => string
    allGateways: string
    hidden: string
    gatewayError: string
    thisDevice: string
    attentionAuth: string
    attentionQuota: string
    attentionConfig: string
    attentionBlocked: string

    title: string
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
    pinToTop: string
    pinnedToTop: (name: string) => string
    unpinned: (name: string) => string
    unhide: string
    shownInRoster: (name: string) => string
    hiddenFromRoster: (name: string) => string
    metadataFailed: string
    loadFailed: string
    groupsFailed: string
    groupsMenu: (groups: string) => string
    manageGroups: string
    duplicating: (name: string) => string
    duplicated: (name: string, source: string) => string

    draftDiscarded: (name: string) => string
    draftCleanupFailed: (name: string) => string
    createError: string
    created: (name: string) => string
    createdOn: (name: string, target: string) => string
    createDesc: string
    nameTaken: (name: string) => string
    nameTakenOn: (name: string, target: string) => string
    createOn: string
    currentConnection: (name: string) => string
    remoteCreateHint: (target: string) => string
    titleLabel: string
    titlePlaceholder: string
    descriptionLabel: string
    generalTab: string
    cloneFrom: string
    cloneFromOn: (target: string) => string
    freshProfile: string
    inheritedModel: string
    soulLabel: string
    shareAuth: string
    shareAuthHint: string
    createEmpty: string
    capabilitiesNameTaken: string
    capabilitiesNeedName: string
    skillsUnsupported: string
    catalogUnsupported: string
    emptySkillsHint: string
    catalogFrom: (source: string) => string
    defaultToolsetsHint: string
    catalogInstalled: string
    catalog: string
    mcpCatalogHint: string
    createAction: string
    newTitle: string
    editTitle: string
    editMenu: string
    helpPromptPlaceholder: string
    descriptionHint: string
    newChatWith: string
    /** Re-opens the forever-chat on purpose. A plain row click only returns to
     *  the tabs already open, so a closed Bot Chat needs an explicit ask. */
    openBotChat: string
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
    /** Stands under the bot's name in a chat it has not spoken in yet. */
    chatEmpty: string
    /** First line of a brand-new bot's forever-chat — see `kickoffText`. */
    kickoff: string
  }
  /** Avatar picker: shapes, blobs, pets, uploads, generation. */
  avatar: {
    auto: string
    autoHint: string
    lockFace: string
    lockFaceHint: string
    unlock: string
    faceFollowsName: string
    faceLocked: string
    imageModelUnavailable: (restartAction: string) => string
    checkingImageBackend: string
    chooseImage: string
    blobKinds: {
      round: string
      organic: string
      boxy: string
      capsule: string
      nub: string
      cloud: string
      droplet: string
      hexagon: string
      sun: string
      triangle: string
    }
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
    openChat: string
    availableMembers: (available: number, total: number) => string

    memberAdded: (name: string, group: string) => string
    memberRemoved: (name: string, group: string) => string
    newGroupPlaceholder: string
    nameExample: string
    createAndJoin: string
    createdWithBots: (name: string, count: number) => string
    createDesc: (max: number) => string
    memberInGroups: (handle: string, groups: string) => string
    noBotMatches: (query: string) => string
    createBotFirst: string
    atLeastTwo: string
    createAction: (count: string) => string
    newTitle: string
    manageDesc: string
    manageTitle: string
    settingsTitle: string
    settingsDesc: string
    nameLabel: string
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
    pictureGenerationFailed: string
    nameTaken: (name: string) => string
    memberCount: (count: number) => string
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
    skillsHub: string
    filterSkills: string
    searchHub: string
    noMcpServers: string
  }

  /** Bot-scoped scheduled jobs. Generic scheduling chrome (weekday names,
   *  Daily/Hourly, the job verbs) resolves against core's `cron` section. */
  cron: {
    untitledJob: string
    nameNulError: string
    instructionNulError: string
    resultSucceeded: string
    resultFailed: string
    resultDeliveryFailed: string
    resultBlockedConfig: string
    detailStatus: string
    detailActive: string
    detailPaused: string
    detailSchedule: string
    detailRawSchedule: string
    detailRepeat: string
    detailNextRun: string
    detailLastRun: string
    detailLastResult: string
    detailWorkdir: string
    detailDesc: string
    legacyPaused: string

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
    unitFromNow: (unit: string) => string
    stopAfterPrefix: string
    stopAfterSuffix: string
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
  model: {
    customProvider: string
    customModel: string
    providerExample: string
    modelExample: string
    backToDropdowns: string
    inheritLaunchProfile: string
    enterManually: string
    gatewayDefault: string
    nameExample: string
  }
}

const en: BotsMessages = {
  model: {
    customProvider: 'Provider (Custom)',
    customModel: 'Model (Custom)',
    providerExample: 'e.g. omnirouter, inferx, 9router',
    modelExample: 'e.g. antigravity/gemini-3.6-flash-high',
    backToDropdowns: '← Back to dropdowns',
    inheritLaunchProfile: 'Use profile model settings',
    enterManually: '✏️ Enter manually…',
    gatewayDefault: 'profile model settings',
    nameExample: 'e.g. model name'
  },
  roster: {
    activityToastsOn: 'Activity toasts on — click to silence',
    activityToastsOff: 'Activity toasts off — click to enable',
    filterRoster: 'Filter roster',
    activeFilters: count => `Filters (${count} active)`,
    filterRosterActive: count => `Filter roster, ${count} active`,
    allGateways: 'All gateways',
    hidden: 'Hidden',
    gatewayError: 'gateway error',
    thisDevice: 'This device',
    attentionAuth: 'Sign in again for this profile',
    attentionQuota: 'Quota or balance exhausted',
    attentionConfig: 'Provider not configured — run hermes model',
    attentionBlocked: 'Bot is blocked — see its last message',

    title: 'Bots',
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
    pinToTop: 'Pin to top',
    pinnedToTop: name => `${name} pinned to top`,
    unpinned: name => `${name} unpinned`,
    unhide: 'Unhide',
    shownInRoster: name => `${name} is back in the roster`,
    hiddenFromRoster: name => `${name} hidden — show hidden bots in the roster to find it again`,
    metadataFailed: 'Could not load bot metadata',
    loadFailed: 'Could not load bot',
    groupsFailed: 'Could not load bot groups',
    groupsMenu: groups => `Groups: ${groups}…`,
    manageGroups: 'Manage groups…',
    duplicating: name => `Duplicating ${name}…`,
    duplicated: (name, source) => `Created ${name} — full copy of ${source}`,

    draftDiscarded: name => `Draft agent "${name}" discarded`,
    draftCleanupFailed: name => `Could not clean up draft profile "${name}"`,
    createError: 'Could not create the bot.',
    created: name => `Bot "${name}" created`,
    createdOn: (name, target) => `Bot "${name}" created on ${target}`,
    createDesc: 'A named teammate with its own memory, skills, and chat. It can message your other agents.',
    nameTaken: name => `An agent named "${name}" already exists.`,
    nameTakenOn: (name, target) => `An agent named "${name}" already exists on ${target}.`,
    createOn: 'Create on',
    currentConnection: name => `${name} (current)`,
    remoteCreateHint: target =>
      `The agent is created on ${target} and appears in the roster as a Connections bot. Chat routes to that machine.`,
    titleLabel: 'Title',
    titlePlaceholder: 'Inbox Triage',
    descriptionLabel: 'Description',
    generalTab: 'General',
    cloneFrom: 'Clone from profile',
    cloneFromOn: target => `Clone from profile (on ${target})`,
    freshProfile: 'Fresh profile (bundled skills)',
    inheritedModel: 'profile model settings',
    soulLabel: 'SOUL.md (optional — replaces the generated persona)',
    shareAuth: 'Share keys & accounts with the main profile',
    shareAuthHint:
      'Subscriptions, OAuth logins, and API keys stay shared (not copied), so token refreshes never invalidate each other. Uncheck for an isolated snapshot copy.',
    createEmpty: 'Create empty (skip bundled skills)',
    capabilitiesNameTaken: 'That name is taken — pick another before configuring capabilities.',
    capabilitiesNeedName:
      'Name the bot first — a draft profile is created when you open this tab (discarded if you cancel).',
    skillsUnsupported: 'Skills need a newer Hermes Desktop.',
    catalogUnsupported: 'Capability catalog needs a newer gateway (restart it after updating Hermes).',
    emptySkillsHint: '“Create empty” is checked — no bundled skills will be installed.',
    catalogFrom: source => `Catalog from ${source} — unchecked skills are disabled after creation.`,
    defaultToolsetsHint: 'Leaving all (or none) checked keeps the default toolset behavior.',
    catalogInstalled: 'catalog · installed',
    catalog: 'catalog',
    mcpCatalogHint:
      'Configured servers copy from the main profile; catalog entries are the bundled MCP menu. Entries needing API keys route through setup first (credentials follow the shared keys setting).',
    createAction: 'Create Bot',
    newTitle: 'New bot',
    editTitle: 'Edit profile',
    editMenu: 'Edit…',
    helpPromptPlaceholder: 'What should this bot help with?',
    descriptionHint: 'Leave blank to generate from the bot’s name and description.',
    newChatWith: 'New chat with this bot',
    openBotChat: 'Open Bot Chat',
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
    chatEmpty: 'Say something to get started.',
    kickoff: 'Hey, tell me about yourself!'
  },
  avatar: {
    auto: 'Auto',
    autoHint: 'Auto — the name decides',
    lockFace: 'Lock face',
    lockFaceHint: 'Keep this exact face even if the name changes',
    unlock: 'Unlock',
    faceFollowsName: 'Face follows the name.',
    faceLocked: 'Face locked — renaming won’t change it.',
    imageModelUnavailable: restartAction =>
      `No image model available. If you just enabled one (or updated Hermes), restart the gateway: Ctrl+K → "${restartAction}".`,
    checkingImageBackend: 'Checking image backend…',
    chooseImage: 'Choose an image…',
    blobKinds: {
      round: 'round',
      organic: 'organic',
      boxy: 'boxy',
      capsule: 'capsule',
      nub: 'nub',
      cloud: 'cloud',
      droplet: 'droplet',
      hexagon: 'hexagon',
      sun: 'sun',
      triangle: 'triangle'
    },
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
    openChat: 'Open Group Chat',
    availableMembers: (available, total) => `${available} of ${total} available`,

    memberAdded: (name, group) => `${name} added to “${group}”`,
    memberRemoved: (name, group) => `${name} removed from “${group}”`,
    newGroupPlaceholder: 'New group…',
    nameExample: 'Group name (e.g. Research)',
    createAndJoin: 'Create & join',
    createdWithBots: (name, count) => `“${name}” created with ${count} bots`,
    createDesc: max =>
      `Pick 2–${max} bots. Local memberships sync through each Bot profile; cross-machine members stay scoped to this room.`,
    memberInGroups: (handle, groups) => `@${handle} · in ${groups}`,
    noBotMatches: query => `No bots match “${query}”`,
    createBotFirst: 'No bots yet — create one first.',
    atLeastTwo: 'Pick at least 2 bots',
    createAction: count => `Create Group${count}`,
    newTitle: 'New group chat',
    manageDesc: 'A bot can join multiple group chats. Memberships sync to every machine.',
    manageTitle: 'Manage groups',
    settingsTitle: 'Group settings',
    settingsDesc: 'Rename the group or set a room picture. Members and history are kept.',
    nameLabel: 'Group name',
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
    pictureGenerationFailed: 'Group picture generation failed',
    nameTaken: name => `A group named “${name}” already exists.`,
    memberCount: count => `${count} bots`,
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
    skillsHub: 'Hermes Skills Hub',
    filterSkills: 'Filter skills…',
    searchHub: 'Search the hub (community + well-known sources)…',
    noMcpServers: 'No MCP servers configured or in the catalog.'
  },
  cron: {
    untitledJob: 'Untitled job',
    nameNulError: 'Job name cannot contain NUL (U+0000).',
    instructionNulError: 'Job instruction cannot contain NUL (U+0000).',
    resultSucceeded: 'Succeeded',
    resultFailed: 'Failed',
    resultDeliveryFailed: 'Ran, but delivery failed',
    resultBlockedConfig: 'Blocked by configuration (not run)',
    detailStatus: 'Status',
    detailActive: 'Active',
    detailPaused: 'Paused',
    detailSchedule: 'Schedule',
    detailRawSchedule: 'Schedule (raw)',
    detailRepeat: 'Repeat',
    detailNextRun: 'Next run',
    detailLastRun: 'Last run',
    detailLastResult: 'Last result',
    detailWorkdir: 'Working directory',
    detailDesc: 'What this job runs, and when it runs next.',
    legacyPaused: 'Paused for security: delete and recreate this legacy job before running it again.',

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
    unitFromNow: unit => `${unit} from now`,
    stopAfterPrefix: 'Stop after',
    stopAfterSuffix: 'runs (blank = forever)',
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

const ko: BotsMessages = {
  model: {
    customProvider: '공급자 (직접 입력)',
    customModel: '모델 (직접 입력)',
    providerExample: '예: omnirouter, inferx, 9router',
    modelExample: '예: antigravity/gemini-3.6-flash-high',
    backToDropdowns: '← 목록에서 선택',
    inheritLaunchProfile: '프로필 모델 설정 사용',
    enterManually: '✏️ 직접 입력…',
    gatewayDefault: '프로필 모델 설정 사용',
    nameExample: '예: 모델 이름'
  },
  roster: {
    activityToastsOn: '활동 알림 켜짐 — 클릭하여 끄기',
    activityToastsOff: '활동 알림 꺼짐 — 클릭하여 켜기',
    filterRoster: '봇 목록 필터',
    activeFilters: count => `필터 (${count}개 적용 중)`,
    filterRosterActive: count => `봇 목록 필터, ${count}개 적용 중`,
    allGateways: '모든 게이트웨이',
    hidden: '숨김',
    gatewayError: '게이트웨이 오류',
    thisDevice: '이 기기',
    attentionAuth: '이 프로필에 다시 로그인하세요',
    attentionQuota: '사용 한도 또는 잔액을 모두 소진했습니다',
    attentionConfig: '공급자가 설정되지 않았습니다 — hermes model을 실행하세요',
    attentionBlocked: '봇이 진행하지 못하고 있습니다 — 마지막 메시지를 확인하세요',

    title: '봇',
    search: '봇과 그룹 대화 검색',
    searchPlaceholder: '봇과 그룹 대화 검색…',
    newBotOrGroup: '새 봇 또는 그룹 대화',
    groupChats: '그룹 대화',
    emptyTitle: '아직 봇이 없습니다',
    emptyDesc: '첫 번째 봇을 만들어 보세요.',
    noMatchQuery: query => `“${query}”에 해당하는 봇이나 그룹 대화가 없습니다`,
    noMatchQueryOn: (query, gateway) => `${gateway}에 “${query}”에 해당하는 봇이나 그룹 대화가 없습니다`,
    noMatchFiltersOn: gateway => `${gateway}에 이 필터와 일치하는 봇이나 그룹 대화가 없습니다`,
    noMatchFilters: '이 필터와 일치하는 봇이나 그룹 대화가 없습니다.',
    clearFilters: '필터 초기화',
    allHidden: '모든 봇이 숨겨져 있습니다',
    allHiddenDesc: '숨겨진 봇도 계속 작동하며 기록이 유지됩니다.',
    showHidden: '숨겨진 봇 표시',
    noHiddenMatch: '이 필터와 일치하는 숨겨진 봇이 없습니다.',
    hiddenFromRoster: '목록에서 숨김',
    pinned: '고정됨',
    needsAttention: '확인 필요',
    needsInput: '입력이 필요합니다',
    botsAndGroups: '봇과 그룹 대화',
    botsOnly: '봇만',
    groupsOnly: '그룹 대화만',
    anyActivity: '모든 활동',
    activeNow: '현재 활동 중',
    recentlyActive: '최근 활동',
    older: '이전 활동',
    gatewayRemoved: '게이트웨이가 삭제되었습니다',
    onDemand: '필요할 때 실행',
    ready: '준비 완료',
    statusUnknown: '상태를 알 수 없음',
    unavailable: '사용할 수 없음',
    retryNow: '지금 다시 시도',
    rosterUnavailable: reason =>
      `목록을 불러올 수 없습니다: ${reason}. 게이트웨이가 profiles.list를 지원하지 않는 구버전이라면 Hermes를 업데이트한 뒤 게이트웨이를 다시 시작하세요.`,
    waitingForGateway:
      '게이트웨이 연결을 기다리는 중… (원격 게이트웨이는 몇 초 걸릴 수 있으며 자동으로 다시 시도합니다)'
  },
  sections: {
    newSection: '새 섹션',
    newTitle: '새 섹션',
    renameTitle: '섹션 이름 변경',
    nameLabel: '섹션 이름',
    namePlaceholder: '예: 고객사',
    create: '만들기',
    rename: '이름 변경…',
    moveUp: '위로 이동',
    moveDown: '아래로 이동',
    unassigned: '미분류',
    options: name => `${name} 섹션 옵션`,
    headingTip: '여기에 봇 놓기 · 두 번 클릭하여 이름 변경',
    emptyHint: '봇을 여기로 드래그하세요',
    moveTo: '섹션으로 이동',
    newSectionEllipsis: '새 섹션…',
    removeFromSection: '섹션에서 제외',
    deleted: (name, count) =>
      count === 0 ? `“${name}” 삭제됨` : `“${name}” 삭제됨 — 봇 ${count}개를 미분류로 이동했습니다`,
    undo: '실행 취소'
  },
  bot: {
    pinToTop: '맨 위에 고정',
    pinnedToTop: name => `맨 위에 고정했습니다: ${name}`,
    unpinned: name => `고정을 해제했습니다: ${name}`,
    unhide: '숨김 해제',
    shownInRoster: name => `목록에 다시 표시했습니다: ${name}`,
    hiddenFromRoster: name => `봇을 숨겼습니다: ${name}. 숨겨진 봇 목록에서 다시 찾을 수 있습니다.`,
    metadataFailed: '봇 정보를 불러오지 못했습니다',
    loadFailed: '봇을 불러오지 못했습니다',
    groupsFailed: '봇의 그룹을 불러오지 못했습니다',
    groupsMenu: groups => `그룹: ${groups}…`,
    manageGroups: '그룹 관리…',
    duplicating: name => `봇 복제 중: ${name}…`,
    duplicated: (name, source) => `${source}의 전체 복사본을 만들었습니다: ${name}`,

    draftDiscarded: name => `에이전트 초안을 버렸습니다: "${name}"`,
    draftCleanupFailed: name => `프로필 초안을 정리하지 못했습니다: "${name}"`,
    createError: '봇을 만들지 못했습니다.',
    created: name => `봇을 만들었습니다: "${name}"`,
    createdOn: (name, target) => `${target}에 봇을 만들었습니다: "${name}"`,
    createDesc: '자신만의 메모리, 스킬, 대화를 갖춘 동료입니다. 다른 에이전트와 메시지를 주고받을 수 있습니다.',
    nameTaken: name => `이름이 "${name}"인 에이전트가 이미 있습니다.`,
    nameTakenOn: (name, target) => `${target}에 이름이 "${name}"인 에이전트가 이미 있습니다.`,
    createOn: '생성할 연결',
    currentConnection: name => `${name} (현재 연결)`,
    remoteCreateHint: target =>
      `에이전트는 ${target}에 생성되어 연결된 봇으로 목록에 표시됩니다. 대화는 해당 컴퓨터에서 처리합니다.`,
    titleLabel: '표시 제목',
    titlePlaceholder: '받은 메일 분류',
    descriptionLabel: '설명',
    generalTab: '일반',
    cloneFrom: '복제할 원본 프로필',
    cloneFromOn: target => `복제할 원본 프로필 (${target})`,
    freshProfile: '새 프로필 (기본 제공 스킬 포함)',
    inheritedModel: '프로필 모델 설정 사용',
    soulLabel: 'SOUL.md (선택 사항 — 자동 생성된 페르소나를 대체)',
    shareAuth: '기본 프로필과 키 및 계정 공유',
    shareAuthHint:
      '구독, OAuth 로그인, API 키를 복사하지 않고 공유하므로 토큰 갱신이 서로의 인증을 무효화하지 않습니다. 선택을 해제하면 현재 상태를 독립된 사본으로 복사합니다.',
    createEmpty: '빈 프로필 만들기 (기본 제공 스킬 제외)',
    capabilitiesNameTaken: '이미 사용 중인 이름입니다. 스킬과 도구를 설정하기 전에 다른 이름을 선택하세요.',
    capabilitiesNeedName: '먼저 봇의 이름을 입력하세요. 이 탭을 열면 프로필 초안이 생성되며, 취소하면 삭제됩니다.',
    skillsUnsupported: '스킬을 사용하려면 Hermes Desktop을 업데이트하세요.',
    catalogUnsupported:
      '스킬과 도구 목록을 사용하려면 게이트웨이를 업데이트해야 합니다. Hermes를 업데이트한 뒤 게이트웨이를 다시 시작하세요.',
    emptySkillsHint: '“빈 프로필 만들기”가 선택되어 기본 제공 스킬을 설치하지 않습니다.',
    catalogFrom: source => `${source}의 목록입니다. 선택하지 않은 스킬은 생성 후 비활성화됩니다.`,
    defaultToolsetsHint: '모두 선택하거나 모두 해제하면 기본 도구 세트 동작을 유지합니다.',
    catalogInstalled: '카탈로그 · 설치됨',
    catalog: '카탈로그',
    mcpCatalogHint:
      '설정된 서버는 기본 프로필에서 복사하며, 카탈로그 항목은 기본 제공 MCP 목록입니다. API 키가 필요한 항목은 먼저 설정을 진행합니다. 인증 정보는 키 공유 설정을 따릅니다.',
    createAction: '봇 만들기',
    newTitle: '새 봇',
    editTitle: '프로필 편집',
    editMenu: '편집…',
    helpPromptPlaceholder: '이 봇이 어떤 일을 도와주면 좋을까요?',
    descriptionHint: '비워 두면 봇의 이름과 설명을 바탕으로 생성합니다.',
    newChatWith: '이 봇과 새 대화',
    openBotChat: '봇 대화 열기',
    duplicate: '복제',
    duplicateFailed: '복제하지 못했습니다',
    deleteTitle: '봇과 프로필을 삭제할까요?',
    removeFromAllGroups: '모든 그룹에서 제외',
    createFirstHint: '봇 패널에서 “새 봇”을 누르세요.',
    createFailed: '아직 프로필을 만들지 못했습니다',
    advanced: '고급',
    advancedHint: '고급 — 모델, 스킬, 도구 세트, SOUL.md',
    advancedFailed: '고급 설정을 적용하지 못했습니다',
    openAnotherChatUnsupported: '봇과 다른 대화를 열려면 Hermes Desktop을 업데이트하세요.',
    remoteConnectionsUnsupported: '다른 연결에 있는 봇과 대화하려면 Hermes Desktop을 업데이트하세요.',
    chatEmpty: '메시지를 보내 대화를 시작해 보세요.',
    // Sent to the model by kickoffText; keep the existing prompt unchanged.
    kickoff: en.bot.kickoff
  },
  avatar: {
    auto: '자동',
    autoHint: '이름에 따라 자동 선택',
    lockFace: '얼굴 고정',
    lockFaceHint: '이름을 바꿔도 이 얼굴을 유지합니다',
    unlock: '고정 해제',
    faceFollowsName: '이름에 따라 얼굴이 바뀝니다.',
    faceLocked: '얼굴을 고정했습니다. 이름을 바꿔도 얼굴은 유지됩니다.',
    imageModelUnavailable: restartAction =>
      `사용 가능한 이미지 모델이 없습니다. 방금 이미지 모델을 활성화했거나 Hermes를 업데이트했다면 Ctrl+K → "${restartAction}"에서 게이트웨이를 다시 시작하세요.`,
    checkingImageBackend: '이미지 백엔드 확인 중…',
    chooseImage: '이미지 선택…',
    blobKinds: {
      round: '둥근 모양',
      organic: '자유로운 모양',
      boxy: '각진 모양',
      capsule: '캡슐',
      nub: '돌기',
      cloud: '구름',
      droplet: '물방울',
      hexagon: '육각형',
      sun: '태양',
      triangle: '삼각형'
    },
    classicShapes: '기본 도형',
    blobFromName: '블롭 얼굴 — 봇 이름으로 생성',
    unlockFollowsName: '고정 해제 — 봇 이름에 따라 얼굴이 다시 바뀝니다',
    randomize: '무작위 선택',
    tabBot: '봇',
    tabGenerate: '생성',
    upload: '업로드',
    tabPet: '펫',
    removeImage: '이미지 제거 — 도형 사용',
    removeBackToShape: '제거 — 도형 아바타로 돌아가기',
    describePlaceholder: '원하는 아바타를 설명하세요…',
    describeHint: '비워 두면 이름, 제목, 설명과 에이전트 메시징 목록을 바탕으로 자동 생성합니다.',
    matchTheName: '이름에 맞추기',
    pickPet: '이 봇의 프로필 사진으로 사용할 펫을 선택하세요.',
    petLoadFailed: '펫을 불러오지 못했습니다. 다른 펫을 선택해 보세요.',
    imageTooLarge: '이미지가 너무 큽니다 (최대 15MB).',
    generationFailed: '아바타를 생성하지 못했습니다',
    savedLocally: '아바타를 로컬에 저장했지만 원격에 저장하지 못했습니다',
    savedLocallyDescriptionFailed: '아바타를 로컬에 저장했지만 설명을 업데이트하지 못했습니다',
    generate: '생성',
    generating: '생성 중…'
  },
  group: {
    openChat: '그룹 대화 열기',
    availableMembers: (available, total) => `${total}개 중 ${available}개 사용 가능`,

    memberAdded: (name, group) => `${name}을(를) “${group}”에 추가했습니다`,
    memberRemoved: (name, group) => `${name}을(를) “${group}”에서 제외했습니다`,
    newGroupPlaceholder: '새 그룹…',
    nameExample: '그룹 이름 (예: 연구)',
    createAndJoin: '만들고 참여',
    createdWithBots: (name, count) => `봇 ${count}개로 “${name}” 그룹을 만들었습니다`,
    createDesc: max =>
      `봇을 2~${max}개 선택하세요. 로컬 봇의 그룹 참여 정보는 각 봇의 프로필을 통해 동기화됩니다. 다른 컴퓨터에서 연결한 봇의 참여 정보는 이 대화방에만 적용됩니다.`,
    memberInGroups: (handle, groups) => `@${handle} · 참여 그룹: ${groups}`,
    noBotMatches: query => `“${query}”에 해당하는 봇이 없습니다`,
    createBotFirst: '아직 봇이 없습니다. 먼저 봇을 만드세요.',
    atLeastTwo: '봇을 2개 이상 선택하세요',
    createAction: count => `그룹 만들기${count}`,
    newTitle: '새 그룹 대화',
    manageDesc: '하나의 봇이 여러 그룹 대화에 참여할 수 있습니다. 참여 정보는 모든 기기에 동기화됩니다.',
    manageTitle: '그룹 관리',
    settingsTitle: '그룹 설정',
    settingsDesc: '그룹 이름을 바꾸거나 대화방 사진을 설정하세요. 참여 봇과 기록은 유지됩니다.',
    nameLabel: '그룹 이름',
    searchToAdd: '추가할 봇 검색',
    searchToAddPlaceholder: '추가할 봇 검색…',
    removeFromSelection: '선택에서 제외',
    disbandTitle: '그룹 대화를 해체할까요?',
    deleteTitle: '그룹 대화를 삭제할까요?',
    deleteAction: '삭제',
    composerPlaceholder: '메시지를 보내세요. 이 그룹의 모든 봇에게 전달됩니다.',
    slashCommandsUnsupported: '그룹 대화에서는 슬래시 명령을 지원하지 않습니다. 개별 봇 대화에서 사용하세요.',
    attachHint: '파일 첨부 — 응답하는 모든 봇이 파일을 볼 수 있습니다',
    newThread: '새 스레드',
    reply: '답글',
    replyInThread: '스레드에 답글',
    replyInThreadPlaceholder: '스레드에 답글…',
    openThread: '이 스레드 열기',
    collapseThread: '스레드 접기',
    collapseThreadLabel: '이 스레드 접기',
    activity: '활동',
    noActivityYet: '이번 턴에는 아직 활동이 없습니다.',
    showActivity: '대화방 활동 표시',
    hideActivity: '대화방 활동 숨기기',
    stop: '중지',
    stopHint: '이번 실행 중지 — 현재 응답 중인 봇을 멈추고 나머지 봇은 대기시킵니다',
    allHeldStatus: count => `봇 ${count}개가 모두 일시 중지되었습니다`,
    heldMembersStatus: members => `일시 중지됨: ${members}`,
    holdReleaseHint: '일시 중지된 봇을 멘션하거나 @all resume을 보내면 다시 실행합니다.',
    needsYourInput: '이 그룹 대화의 봇이 입력을 기다리고 있습니다',
    pictureGenerationFailed: '그룹 사진을 생성하지 못했습니다',
    nameTaken: name => `“${name}” 그룹이 이미 있습니다.`,
    memberCount: count => `봇 ${count}개`,
    settingsHint: group => `그룹 설정 — ${group} 이름 변경 또는 대화방 사진 설정`,
    settingsLabel: group => `${group} 그룹 설정`,
    disbandHint: group => `${group} 그룹 대화 해체`,
    disbandLabel: group => `${group} 해체`,
    disbandAction: '해체',
    disbanding: '해체 중…',
    disbandDone: '해체됨',
    disbanded: group => `“${group}” 해체됨`,
    disbandDescPrefix: '',
    disbandDescSuffix: count =>
      ` 그룹에 속한 봇 ${count}개의 그룹 참여 정보를 제거하고 공유 대화 기록을 지웁니다. 봇 자체와 각 봇의 그룹별 세션은 유지됩니다.`,
    stopped: group => `${group} 중지됨 — 재개할 때까지 나머지 봇은 대기합니다`,
    removeAttachment: '첨부 파일 제거',
    threadFallback: '스레드',
    replyCount: replies => `답글 ${replies}개`,
    dropToThread: '여기에 놓아 스레드 답글에 첨부',
    dropToRoom: '여기에 놓아 첨부 — 응답하는 모든 봇이 볼 수 있습니다',
    waitingForAnswer: '답변을 기다리는 중…',
    memberThinking: name => `${name} 생각 중…`,
    roomWorking: '대화방에서 작업 중…',
    messageRoom: group => `${group}에 메시지 보내기`,
    newThreadPlaceholder: group => `${group}의 새 스레드… (특정 봇은 @name, 모든 봇은 @everyone)`,
    everyoneMeta: '대화방의 모든 봇',
    commandApproval: '명령 승인',
    answerFailed: (handle, error) => `@${handle}에게 답변을 보내지 못했습니다: ${error}`,
    wantsToRunCommand: handle => `@${handle}의 명령 실행 요청:`,
    asks: handle => `@${handle}의 질문:`,
    answerTo: member => `@${member}에게 답변`
  },
  tools: {
    skillsHub: 'Hermes 스킬 허브',
    filterSkills: '스킬 필터링…',
    searchHub: '허브 검색 (커뮤니티 및 알려진 소스)…',
    noMcpServers: '설정되었거나 카탈로그에 등록된 MCP 서버가 없습니다.'
  },
  cron: {
    untitledJob: '이름 없는 작업',
    nameNulError: '작업 이름에는 NUL 문자(U+0000)를 넣을 수 없습니다.',
    instructionNulError: '작업 지시에는 NUL 문자(U+0000)를 넣을 수 없습니다.',
    resultSucceeded: '성공',
    resultFailed: '실패',
    resultDeliveryFailed: '실행했지만 결과 전달 실패',
    resultBlockedConfig: '설정 문제로 차단됨 (실행하지 않음)',
    detailStatus: '상태',
    detailActive: '활성',
    detailPaused: '일시 중지',
    detailSchedule: '일정',
    detailRawSchedule: '일정 (원문)',
    detailRepeat: '반복',
    detailNextRun: '다음 실행',
    detailLastRun: '최근 실행',
    detailLastResult: '최근 실행 결과',
    detailWorkdir: '작업 디렉터리',
    detailDesc: '작업의 실행 내용과 다음 실행 시각을 확인합니다.',
    legacyPaused: '보안을 위해 일시 중지했습니다. 다시 실행하려면 이 이전 형식의 작업을 삭제한 뒤 새로 만드세요.',

    filterHint:
      '이 프로필에 예약 작업이 있지만 이 봇의 태그가 붙은 작업은 없습니다. 여기에 표시하려면 작업 이름을 "[bot:<name>] …" 형식으로 지정하세요. 아래 예약 작업에서도 확인할 수 있습니다.',
    needsRosterFirst: '먼저 이 봇이 목록에 표시되어야 합니다.',
    staleNotice: '예약 작업을 새로 불러오지 못했습니다. 마지막으로 불러온 목록을 표시합니다.',
    readFailure: '목록을 읽지 못한 것이며 삭제된 것은 아닙니다. 기존 목록이 남아 있을 수 있습니다.',
    createDesc: bot => `${bot}의 예약된 반복 작업입니다. 실행 결과는 해당 작업의 대화 기록에 남습니다.`,
    instruction: '지시 사항',
    whenToRun: '실행 시점',
    dayOfMonth: '매월 실행일',
    sendResultsTo: '결과 전송 대상',
    runHistoryOnly: '실행 기록에만 저장',
    botChatTarget: bot => `${bot}의 대화 (봇이 응답)`,
    continuity: '작업 연속성: 매번 이전 실행 결과를 참고합니다 (중복 방지, 이전 작업 이어서 수행)',
    onceIn: when => `한 번 (${when})`,
    everyNDays: days => `${days}일마다`,
    everyNHours: hours => `${hours}시간마다`,
    everyNMinutes: minutes => `${minutes}분마다`,
    freqOnce: '일정 시간 후 한 번…',
    freqHourly: '매시간',
    freqDaily: '매일',
    freqWeekdays: '평일',
    freqWeekly: '매주',
    freqMonthly: '매월',
    freqInterval: '일정 간격',
    freqAdvanced: '고급…',
    unitMinutes: '분',
    unitHours: '시간',
    unitDays: '일',
    unitFromNow: unit => `${unit} 후`,
    stopAfterPrefix: '총',
    stopAfterSuffix: '회 실행 후 중지 (비워 두면 무제한)',
    runsOnce: (count, unit) => `지금부터 ${count}${unit} 후 한 번 실행`,
    runsHourly: '매시간 정각에 실행',
    runsDaily: time => `매일 ${time}에 실행`,
    runsWeekdays: time => `월요일부터 금요일까지 ${time}에 실행`,
    runsWeekly: (day, time) => `매주 ${day} ${time}에 실행`,
    runsMonthly: (day, time) => `매월 ${day}일 ${time}에 실행`,
    runsInterval: (count, unit) => `${count}${unit}마다 실행`,
    runsRaw: '직접 입력 — every Nm/Nh/Nd 또는 5개 필드의 cron 표현식',
    timesTotal: count => `, 총 ${count}회`
  }
}

const ja: BotsMessages = {
  model: {
    customProvider: 'プロバイダー（手動入力）',
    customModel: 'モデル（手動入力）',
    providerExample: '例: omnirouter, inferx, 9router',
    modelExample: '例: antigravity/gemini-3.6-flash-high',
    backToDropdowns: '← リスト選択に戻る',
    inheritLaunchProfile: 'プロファイルのモデル設定を使用',
    enterManually: '✏️ 手動入力…',
    gatewayDefault: 'プロファイルのモデル設定',
    nameExample: '例: モデル名'
  },
  roster: {
    activityToastsOn: 'アクティビティ通知オン — クリックでオフ',
    activityToastsOff: 'アクティビティ通知オフ — クリックでオン',
    filterRoster: '一覧を絞り込む',
    activeFilters: count => `フィルター（${count}個適用中）`,
    filterRosterActive: count => `一覧を絞り込む、${count}個適用中`,
    allGateways: 'すべてのゲートウェイ',
    hidden: '非表示',
    gatewayError: 'ゲートウェイエラー',
    thisDevice: 'このデバイス',
    attentionAuth: 'このプロファイルで再ログインしてください',
    attentionQuota: '利用枠または残高を使い切りました',
    attentionConfig: 'プロバイダー未設定 — hermes model を実行してください',
    attentionBlocked: 'ボットが進行できません — 最後のメッセージを確認してください',

    title: 'ボット',
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
    pinToTop: '先頭に固定',
    pinnedToTop: name => `${name} を先頭に固定しました`,
    unpinned: name => `${name} の固定を解除しました`,
    unhide: '再表示',
    shownInRoster: name => `${name} を一覧に再表示しました`,
    hiddenFromRoster: name => `${name} を非表示にしました — 一覧で非表示のボットを表示すると再び見つかります`,
    metadataFailed: 'ボットのメタデータを読み込めませんでした',
    loadFailed: 'ボットを読み込めませんでした',
    groupsFailed: 'ボットのグループを読み込めませんでした',
    groupsMenu: groups => `グループ: ${groups}…`,
    manageGroups: 'グループを管理…',
    duplicating: name => `${name} を複製中…`,
    duplicated: (name, source) => `${source} の完全なコピー ${name} を作成しました`,

    draftDiscarded: name => `エージェントの下書き「${name}」を破棄しました`,
    draftCleanupFailed: name => `下書きプロファイル「${name}」を削除できませんでした`,
    createError: 'ボットを作成できませんでした。',
    created: name => `ボット「${name}」を作成しました`,
    createdOn: (name, target) => `${target} にボット「${name}」を作成しました`,
    createDesc:
      '独自のメモリ、スキル、チャットを持つ、名前付きのチームメイトです。他のエージェントとメッセージをやり取りできます。',
    nameTaken: name => `「${name}」という名前のエージェントは既に存在します。`,
    nameTakenOn: (name, target) => `${target} には「${name}」という名前のエージェントが既に存在します。`,
    createOn: '作成先',
    currentConnection: name => `${name}（現在の接続）`,
    remoteCreateHint: target =>
      `エージェントは ${target} に作成され、接続先のボットとして一覧に表示されます。チャットはそのマシンに送られます。`,
    titleLabel: '表示タイトル',
    titlePlaceholder: '受信トレイの整理',
    descriptionLabel: '説明',
    generalTab: '一般',
    cloneFrom: '複製元のプロファイル',
    cloneFromOn: target => `複製元のプロファイル（${target}）`,
    freshProfile: '新規プロファイル（同梱スキルを含む）',
    inheritedModel: 'プロファイルのモデル設定',
    soulLabel: 'SOUL.md（任意 — 自動生成のペルソナを置き換えます）',
    shareAuth: 'メインプロファイルとキー・アカウントを共有',
    shareAuthHint:
      'サブスクリプション、OAuth ログイン、API キーはコピーせず共有するため、トークン更新が互いの認証を無効にしません。オフにすると現在の状態を独立したコピーとして保存します。',
    createEmpty: '空で作成（同梱スキルを除外）',
    capabilitiesNameTaken: 'その名前は使用中です。機能を設定する前に別の名前を選んでください。',
    capabilitiesNeedName:
      '先にボットに名前を付けてください。このタブを開くと下書きプロファイルが作成され、キャンセルすると破棄されます。',
    skillsUnsupported: 'スキルを使うには Hermes Desktop を更新してください。',
    catalogUnsupported:
      '機能カタログには新しいゲートウェイが必要です。Hermes を更新してゲートウェイを再起動してください。',
    emptySkillsHint: '「空で作成」が選択されているため、同梱スキルはインストールされません。',
    catalogFrom: source => `${source} のカタログです。未選択のスキルは作成後に無効になります。`,
    defaultToolsetsHint: 'すべて選択、またはすべて未選択の場合は、既定のツールセット動作を維持します。',
    catalogInstalled: 'カタログ · インストール済み',
    catalog: 'カタログ',
    mcpCatalogHint:
      '設定済みサーバーはメインプロファイルからコピーされ、カタログは同梱の MCP メニューです。API キーが必要な項目は先に設定します。認証情報はキー共有設定に従います。',
    createAction: 'ボットを作成',
    newTitle: '新しいボット',
    editTitle: 'プロファイルを編集',
    editMenu: '編集…',
    helpPromptPlaceholder: 'このボットは何を手伝いますか？',
    descriptionHint: '空欄のままにすると、ボットの名前と説明から生成します。',
    newChatWith: 'このボットと新しいチャット',
    openBotChat: 'ボットチャットを開く',
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
    chatEmpty: '何か書いて始めましょう。',
    kickoff: 'こんにちは、自己紹介をしてください！'
  },
  avatar: {
    auto: '自動',
    autoHint: '名前に応じて自動で選択',
    lockFace: '顔を固定',
    lockFaceHint: '名前を変えてもこの顔を維持します',
    unlock: '固定を解除',
    faceFollowsName: '名前に応じて顔が変わります。',
    faceLocked: '顔を固定しました。名前を変えても顔は変わりません。',
    imageModelUnavailable: restartAction =>
      `利用できる画像モデルがありません。画像モデルを有効にした直後、または Hermes を更新した直後なら、Ctrl+K →「${restartAction}」でゲートウェイを再起動してください。`,
    checkingImageBackend: '画像バックエンドを確認中…',
    chooseImage: '画像を選択…',
    blobKinds: {
      round: '丸型',
      organic: '不定形',
      boxy: '角型',
      capsule: 'カプセル',
      nub: '突起',
      cloud: '雲',
      droplet: 'しずく',
      hexagon: '六角形',
      sun: '太陽',
      triangle: '三角形'
    },
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
    openChat: 'グループチャットを開く',
    availableMembers: (available, total) => `${total}体中${available}体が利用可能`,

    memberAdded: (name, group) => `${name} を「${group}」に追加しました`,
    memberRemoved: (name, group) => `${name} を「${group}」から外しました`,
    newGroupPlaceholder: '新しいグループ…',
    nameExample: 'グループ名（例: 調査）',
    createAndJoin: '作成して参加',
    createdWithBots: (name, count) => `ボット ${count} 体で「${name}」を作成しました`,
    createDesc: max =>
      `ボットを 2〜${max} 体選んでください。ローカルの参加情報は各ボットのプロファイルで同期され、別マシンのメンバーはこのルームにのみ参加します。`,
    memberInGroups: (handle, groups) => `@${handle} · 参加先: ${groups}`,
    noBotMatches: query => `「${query}」に一致するボットはありません`,
    createBotFirst: 'ボットがまだありません。先に作成してください。',
    atLeastTwo: 'ボットを 2 体以上選んでください',
    createAction: count => `グループを作成${count}`,
    newTitle: '新しいグループチャット',
    manageDesc: 'ボットは複数のグループチャットに参加できます。メンバーシップはすべてのマシンに同期されます。',
    manageTitle: 'グループを管理',
    settingsTitle: 'グループ設定',
    settingsDesc: 'グループ名の変更や部屋の画像の設定ができます。メンバーと履歴は保持されます。',
    nameLabel: 'グループ名',
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
    pictureGenerationFailed: 'グループ画像の生成に失敗しました',
    nameTaken: name => `「${name}」という名前のグループはすでに存在します。`,
    memberCount: count => `ボット${count}体`,
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
    skillsHub: 'Hermes スキルハブ',
    filterSkills: 'スキルを絞り込み…',
    searchHub: 'ハブを検索（コミュニティと既知のソース）…',
    noMcpServers: '設定済みまたはカタログ内の MCP サーバーはありません。'
  },
  cron: {
    untitledJob: '無題のジョブ',
    nameNulError: 'ジョブ名に NUL（U+0000）は使用できません。',
    instructionNulError: 'ジョブの指示に NUL（U+0000）は使用できません。',
    resultSucceeded: '成功',
    resultFailed: '失敗',
    resultDeliveryFailed: '実行済み、結果の配信に失敗',
    resultBlockedConfig: '設定によりブロック（未実行）',
    detailStatus: '状態',
    detailActive: '有効',
    detailPaused: '一時停止',
    detailSchedule: 'スケジュール',
    detailRawSchedule: 'スケジュール（元の形式）',
    detailRepeat: '繰り返し',
    detailNextRun: '次回の実行',
    detailLastRun: '前回の実行',
    detailLastResult: '前回の結果',
    detailWorkdir: '作業ディレクトリ',
    detailDesc: 'このジョブの実行内容と次回の実行時刻。',
    legacyPaused:
      'セキュリティのため一時停止しました。再実行するには、この旧形式のジョブを削除して作り直してください。',

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
    unitFromNow: unit => `${unit}後`,
    stopAfterPrefix: '',
    stopAfterSuffix: '回実行したら停止（空欄なら無制限）',
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
  model: {
    customProvider: '提供商（手动输入）',
    customModel: '模型（手动输入）',
    providerExample: '例如 omnirouter、inferx、9router',
    modelExample: '例如 antigravity/gemini-3.6-flash-high',
    backToDropdowns: '← 返回下拉选择',
    inheritLaunchProfile: '使用配置档案的模型设置',
    enterManually: '✏️ 手动输入…',
    gatewayDefault: '配置档案的模型设置',
    nameExample: '例如模型名称'
  },
  roster: {
    activityToastsOn: '活动通知已开启 — 点击关闭',
    activityToastsOff: '活动通知已关闭 — 点击开启',
    filterRoster: '筛选列表',
    activeFilters: count => `筛选条件（已启用 ${count} 个）`,
    filterRosterActive: count => `筛选列表，已启用 ${count} 个条件`,
    allGateways: '所有网关',
    hidden: '已隐藏',
    gatewayError: '网关错误',
    thisDevice: '此设备',
    attentionAuth: '请为此配置档案重新登录',
    attentionQuota: '配额或余额已用尽',
    attentionConfig: '尚未配置提供商 — 请运行 hermes model',
    attentionBlocked: '机器人无法继续 — 请查看最后一条消息',

    title: '机器人',
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
    pinToTop: '置顶',
    pinnedToTop: name => `已将 ${name} 置顶`,
    unpinned: name => `已取消 ${name} 的置顶`,
    unhide: '取消隐藏',
    shownInRoster: name => `已在列表中重新显示 ${name}`,
    hiddenFromRoster: name => `已隐藏 ${name} — 在列表中显示隐藏的机器人即可再次找到`,
    metadataFailed: '无法加载机器人元数据',
    loadFailed: '无法加载机器人',
    groupsFailed: '无法加载机器人的群组',
    groupsMenu: groups => `群组：${groups}…`,
    manageGroups: '管理群组…',
    duplicating: name => `正在复制 ${name}…`,
    duplicated: (name, source) => `已创建 ${name} — ${source} 的完整副本`,

    draftDiscarded: name => `已丢弃智能体草稿“${name}”`,
    draftCleanupFailed: name => `无法清理草稿配置档案“${name}”`,
    createError: '无法创建机器人。',
    created: name => `已创建机器人“${name}”`,
    createdOn: (name, target) => `已在 ${target} 上创建机器人“${name}”`,
    createDesc: '拥有独立记忆、技能和聊天的具名伙伴，可以与其他智能体互发消息。',
    nameTaken: name => `名为“${name}”的智能体已存在。`,
    nameTakenOn: (name, target) => `${target} 上已存在名为“${name}”的智能体。`,
    createOn: '创建位置',
    currentConnection: name => `${name}（当前连接）`,
    remoteCreateHint: target => `智能体将在 ${target} 上创建，并作为连接中的机器人显示在列表中。聊天将由该计算机处理。`,
    titleLabel: '显示标题',
    titlePlaceholder: '收件箱分类',
    descriptionLabel: '描述',
    generalTab: '常规',
    cloneFrom: '从配置档案克隆',
    cloneFromOn: target => `从配置档案克隆（位于 ${target}）`,
    freshProfile: '新配置档案（含内置技能）',
    inheritedModel: '配置档案的模型设置',
    soulLabel: 'SOUL.md（可选 — 替换自动生成的角色设定）',
    shareAuth: '与主配置档案共享密钥和账户',
    shareAuthHint:
      '订阅、OAuth 登录和 API 密钥保持共享而非复制，因此令牌刷新不会使彼此失效。取消勾选可创建当前状态的独立副本。',
    createEmpty: '创建空配置（不含内置技能）',
    capabilitiesNameTaken: '该名称已被使用，请先选择其他名称再配置功能。',
    capabilitiesNeedName: '请先为机器人命名。打开此标签页时会创建草稿配置档案，取消时将删除。',
    skillsUnsupported: '使用技能需要更新 Hermes Desktop。',
    catalogUnsupported: '功能目录需要新版网关。请更新 Hermes 后重启网关。',
    emptySkillsHint: '已勾选“创建空配置”，不会安装内置技能。',
    catalogFrom: source => `目录来自 ${source}，未勾选的技能将在创建后禁用。`,
    defaultToolsetsHint: '全选或全不选将保留默认工具集行为。',
    catalogInstalled: '目录 · 已安装',
    catalog: '目录',
    mcpCatalogHint:
      '已配置的服务器从主配置档案复制，目录项为内置 MCP 菜单。需要 API 密钥的项目会先进入设置，凭据遵循密钥共享设置。',
    createAction: '创建机器人',
    newTitle: '新建机器人',
    editTitle: '编辑配置档案',
    editMenu: '编辑…',
    helpPromptPlaceholder: '这个机器人应该帮你做什么？',
    descriptionHint: '留空则根据机器人的名称和描述生成。',
    newChatWith: '与此机器人开新聊天',
    openBotChat: '打开机器人聊天',
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
    chatEmpty: '说点什么开始吧。',
    kickoff: '你好，介绍一下你自己吧！'
  },
  avatar: {
    auto: '自动',
    autoHint: '根据名称自动选择',
    lockFace: '固定面孔',
    lockFaceHint: '即使名称改变，也保持这张面孔',
    unlock: '解除固定',
    faceFollowsName: '面孔随名称变化。',
    faceLocked: '面孔已固定，重命名不会改变它。',
    imageModelUnavailable: restartAction =>
      `没有可用的图像模型。如果刚启用了图像模型或更新了 Hermes，请通过 Ctrl+K →“${restartAction}”重启网关。`,
    checkingImageBackend: '正在检查图像后端…',
    chooseImage: '选择图像…',
    blobKinds: {
      round: '圆形',
      organic: '不规则形',
      boxy: '方形',
      capsule: '胶囊',
      nub: '凸起',
      cloud: '云朵',
      droplet: '水滴',
      hexagon: '六边形',
      sun: '太阳',
      triangle: '三角形'
    },
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
    openChat: '打开群聊',
    availableMembers: (available, total) => `${total} 个中有 ${available} 个可用`,

    memberAdded: (name, group) => `已将 ${name} 添加到“${group}”`,
    memberRemoved: (name, group) => `已将 ${name} 从“${group}”移除`,
    newGroupPlaceholder: '新群组…',
    nameExample: '群组名称（例如：研究）',
    createAndJoin: '创建并加入',
    createdWithBots: (name, count) => `已创建“${name}”，包含 ${count} 个机器人`,
    createDesc: max => `请选择 2–${max} 个机器人。本地成员关系通过各机器人配置档案同步，跨计算机成员仅属于此聊天室。`,
    memberInGroups: (handle, groups) => `@${handle} · 所在群组：${groups}`,
    noBotMatches: query => `没有与“${query}”匹配的机器人`,
    createBotFirst: '还没有机器人，请先创建一个。',
    atLeastTwo: '请至少选择 2 个机器人',
    createAction: count => `创建群组${count}`,
    newTitle: '新建群聊',
    manageDesc: '一个机器人可以加入多个群聊。成员关系会同步到每台设备。',
    manageTitle: '管理群组',
    settingsTitle: '群组设置',
    settingsDesc: '重命名群组或设置房间图片。成员和历史都会保留。',
    nameLabel: '群组名称',
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
    pictureGenerationFailed: '群组图片生成失败',
    nameTaken: name => `已存在名为“${name}”的群聊。`,
    memberCount: count => `${count} 个机器人`,
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
    skillsHub: 'Hermes 技能中心',
    filterSkills: '筛选技能…',
    searchHub: '搜索技能中心（社区和常见来源）…',
    noMcpServers: '未配置 MCP 服务器，目录中也没有。'
  },
  cron: {
    untitledJob: '未命名任务',
    nameNulError: '任务名称不能包含 NUL（U+0000）。',
    instructionNulError: '任务指令不能包含 NUL（U+0000）。',
    resultSucceeded: '成功',
    resultFailed: '失败',
    resultDeliveryFailed: '已执行，但结果发送失败',
    resultBlockedConfig: '被配置阻止（未执行）',
    detailStatus: '状态',
    detailActive: '已启用',
    detailPaused: '已暂停',
    detailSchedule: '计划',
    detailRawSchedule: '计划（原始格式）',
    detailRepeat: '重复',
    detailNextRun: '下次执行',
    detailLastRun: '上次执行',
    detailLastResult: '上次结果',
    detailWorkdir: '工作目录',
    detailDesc: '查看此任务的执行内容及下次执行时间。',
    legacyPaused: '出于安全原因已暂停。再次运行前，请删除并重新创建此旧版任务。',

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
    unitFromNow: unit => `${unit}后`,
    stopAfterPrefix: '执行',
    stopAfterSuffix: '次后停止（留空则无限次）',
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
  model: {
    customProvider: '供應商（手動輸入）',
    customModel: '模型（手動輸入）',
    providerExample: '例如 omnirouter、inferx、9router',
    modelExample: '例如 antigravity/gemini-3.6-flash-high',
    backToDropdowns: '← 返回下拉選擇',
    inheritLaunchProfile: '使用設定檔的模型設定',
    enterManually: '✏️ 手動輸入…',
    gatewayDefault: '設定檔的模型設定',
    nameExample: '例如模型名稱'
  },
  roster: {
    activityToastsOn: '活動通知已開啟 — 點擊關閉',
    activityToastsOff: '活動通知已關閉 — 點擊開啟',
    filterRoster: '篩選清單',
    activeFilters: count => `篩選條件（已啟用 ${count} 個）`,
    filterRosterActive: count => `篩選清單，已啟用 ${count} 個條件`,
    allGateways: '所有閘道',
    hidden: '已隱藏',
    gatewayError: '閘道錯誤',
    thisDevice: '此裝置',
    attentionAuth: '請為此設定檔重新登入',
    attentionQuota: '配額或餘額已用盡',
    attentionConfig: '尚未設定供應商 — 請執行 hermes model',
    attentionBlocked: '機器人無法繼續 — 請查看最後一則訊息',

    title: '機器人',
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
    pinToTop: '置頂',
    pinnedToTop: name => `已將 ${name} 置頂`,
    unpinned: name => `已取消 ${name} 的置頂`,
    unhide: '取消隱藏',
    shownInRoster: name => `已在清單中重新顯示 ${name}`,
    hiddenFromRoster: name => `已隱藏 ${name} — 在清單中顯示隱藏的機器人即可再次找到`,
    metadataFailed: '無法載入機器人中繼資料',
    loadFailed: '無法載入機器人',
    groupsFailed: '無法載入機器人的群組',
    groupsMenu: groups => `群組：${groups}…`,
    manageGroups: '管理群組…',
    duplicating: name => `正在複製 ${name}…`,
    duplicated: (name, source) => `已建立 ${name} — ${source} 的完整副本`,

    draftDiscarded: name => `已捨棄代理草稿「${name}」`,
    draftCleanupFailed: name => `無法清理草稿設定檔「${name}」`,
    createError: '無法建立機器人。',
    created: name => `已建立機器人「${name}」`,
    createdOn: (name, target) => `已在 ${target} 上建立機器人「${name}」`,
    createDesc: '擁有獨立記憶、技能與對話的具名夥伴，可以與其他代理互傳訊息。',
    nameTaken: name => `名為「${name}」的代理已存在。`,
    nameTakenOn: (name, target) => `${target} 上已存在名為「${name}」的代理。`,
    createOn: '建立位置',
    currentConnection: name => `${name}（目前連線）`,
    remoteCreateHint: target => `代理將在 ${target} 上建立，並作為連線中的機器人顯示於清單。對話將由該電腦處理。`,
    titleLabel: '顯示標題',
    titlePlaceholder: '收件匣分類',
    descriptionLabel: '說明',
    generalTab: '一般',
    cloneFrom: '複製來源設定檔',
    cloneFromOn: target => `複製來源設定檔（位於 ${target}）`,
    freshProfile: '新設定檔（含內建技能）',
    inheritedModel: '設定檔的模型設定',
    soulLabel: 'SOUL.md（選填 — 取代自動產生的角色設定）',
    shareAuth: '與主要設定檔共用金鑰及帳號',
    shareAuthHint:
      '訂閱、OAuth 登入與 API 金鑰維持共用而非複製，因此權杖更新不會使彼此失效。取消勾選可建立目前狀態的獨立副本。',
    createEmpty: '建立空白設定（不含內建技能）',
    capabilitiesNameTaken: '此名稱已被使用，請先選擇其他名稱再設定功能。',
    capabilitiesNeedName: '請先為機器人命名。開啟此分頁時會建立草稿設定檔，取消時將刪除。',
    skillsUnsupported: '使用技能需要更新 Hermes Desktop。',
    catalogUnsupported: '功能目錄需要新版閘道。請更新 Hermes 後重新啟動閘道。',
    emptySkillsHint: '已勾選「建立空白設定」，不會安裝內建技能。',
    catalogFrom: source => `目錄來自 ${source}，未勾選的技能將在建立後停用。`,
    defaultToolsetsHint: '全選或全部取消將保留預設工具集行為。',
    catalogInstalled: '目錄 · 已安裝',
    catalog: '目錄',
    mcpCatalogHint:
      '已設定的伺服器從主要設定檔複製，目錄項目為內建 MCP 選單。需要 API 金鑰的項目會先進入設定，憑證遵循金鑰共用設定。',
    createAction: '建立機器人',
    newTitle: '新增機器人',
    editTitle: '編輯設定檔',
    editMenu: '編輯…',
    helpPromptPlaceholder: '這個機器人應該幫你做什麼？',
    descriptionHint: '留空則依機器人的名稱和描述產生。',
    newChatWith: '與此機器人開新聊天',
    openBotChat: '開啟機器人聊天',
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
    chatEmpty: '說點什麼開始吧。',
    kickoff: '你好，介紹一下你自己吧！'
  },
  avatar: {
    auto: '自動',
    autoHint: '根據名稱自動選擇',
    lockFace: '固定面孔',
    lockFaceHint: '即使名稱改變，也維持這張面孔',
    unlock: '解除固定',
    faceFollowsName: '面孔隨名稱變化。',
    faceLocked: '面孔已固定，重新命名不會改變它。',
    imageModelUnavailable: restartAction =>
      `沒有可用的圖像模型。如果剛啟用了圖像模型或更新了 Hermes，請透過 Ctrl+K →「${restartAction}」重新啟動閘道。`,
    checkingImageBackend: '正在檢查圖像後端…',
    chooseImage: '選擇圖像…',
    blobKinds: {
      round: '圓形',
      organic: '不規則形',
      boxy: '方形',
      capsule: '膠囊',
      nub: '凸起',
      cloud: '雲朵',
      droplet: '水滴',
      hexagon: '六邊形',
      sun: '太陽',
      triangle: '三角形'
    },
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
    openChat: '開啟群組聊天',
    availableMembers: (available, total) => `${total} 個中有 ${available} 個可用`,

    memberAdded: (name, group) => `已將 ${name} 加入「${group}」`,
    memberRemoved: (name, group) => `已將 ${name} 從「${group}」移除`,
    newGroupPlaceholder: '新群組…',
    nameExample: '群組名稱（例如：研究）',
    createAndJoin: '建立並加入',
    createdWithBots: (name, count) => `已建立「${name}」，包含 ${count} 個機器人`,
    createDesc: max => `請選擇 2–${max} 個機器人。本機成員關係透過各機器人設定檔同步，跨電腦成員僅屬於此聊天室。`,
    memberInGroups: (handle, groups) => `@${handle} · 所屬群組：${groups}`,
    noBotMatches: query => `沒有與「${query}」相符的機器人`,
    createBotFirst: '還沒有機器人，請先建立一個。',
    atLeastTwo: '請至少選擇 2 個機器人',
    createAction: count => `建立群組${count}`,
    newTitle: '新增群組聊天',
    manageDesc: '一個機器人可以加入多個群組聊天。成員關係會同步到每台裝置。',
    manageTitle: '管理群組',
    settingsTitle: '群組設定',
    settingsDesc: '重新命名群組或設定房間圖片。成員和歷史都會保留。',
    nameLabel: '群組名稱',
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
    pictureGenerationFailed: '群組圖片產生失敗',
    nameTaken: name => `已存在名為「${name}」的群組聊天。`,
    memberCount: count => `${count} 個機器人`,
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
    skillsHub: 'Hermes 技能中心',
    filterSkills: '篩選技能…',
    searchHub: '搜尋技能中心（社群和常見來源）…',
    noMcpServers: '未設定 MCP 伺服器，目錄中也沒有。'
  },
  cron: {
    untitledJob: '未命名工作',
    nameNulError: '工作名稱不能包含 NUL（U+0000）。',
    instructionNulError: '工作指示不能包含 NUL（U+0000）。',
    resultSucceeded: '成功',
    resultFailed: '失敗',
    resultDeliveryFailed: '已執行，但結果傳送失敗',
    resultBlockedConfig: '被設定阻擋（未執行）',
    detailStatus: '狀態',
    detailActive: '已啟用',
    detailPaused: '已暫停',
    detailSchedule: '排程',
    detailRawSchedule: '排程（原始格式）',
    detailRepeat: '重複',
    detailNextRun: '下次執行',
    detailLastRun: '上次執行',
    detailLastResult: '上次結果',
    detailWorkdir: '工作目錄',
    detailDesc: '查看此工作的執行內容及下次執行時間。',
    legacyPaused: '基於安全考量已暫停。再次執行前，請刪除並重新建立此舊版工作。',

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
    unitFromNow: unit => `${unit}後`,
    stopAfterPrefix: '執行',
    stopAfterSuffix: '次後停止（留空則無限次）',
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
export const BOTS_LOCALES: PluginLocaleBundles = { en, ko, ja, zh, 'zh-hant': zhHant }

// Bind the message SHAPE to a plugin translator: string leaves resolve on access,
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

    if (typeof value === 'function') {
      out[key] = (...args: unknown[]) => t(path, ...args)
    } else if (value && typeof value === 'object') {
      out[key] = bind(t, value as object, path)
    } else {
      Object.defineProperty(out, key, { enumerable: true, get: () => t(path) })
    }
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
 *  still repaints. Cache the binding, not its resolved strings: the getters
 *  read the current runtime locale even when ctx.i18n.t keeps its identity. */
export function botsText(): BotsText {
  const translate = getPluginCtx()?.i18n?.t ?? english

  if (bound?.translate !== translate) {
    bound = { text: bind(translate, en), translate }
  }

  return bound.text
}
