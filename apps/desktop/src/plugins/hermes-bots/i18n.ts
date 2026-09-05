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
    duplicate: string
    duplicateFailed: string
    deleteTitle: string
    removeFromAllGroups: string
    removeFromOtherGroups: string
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
    newDesc: string
    noBots: string
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
    attachHint: string
    downloadAttachment: string
    attachedFile: string
    downloadFile: (name: string) => string
    attachmentDownloadFailed: string
    sharedFiles: string
    sharedFilesDescription: (group: string) => string
    searchSharedFiles: string
    sharedFilesLoading: string
    sharedFilesError: string
    sharedFilesExpired: string
    sharedFilesOffline: string
    sharedFilesUnavailable: string
    sharedFilesEmpty: string
    sharedFilesPageEmpty: string
    sharedFilesNoResults: string
    sharedFilesRetry: string
    olderFiles: string
    newerFiles: string
    returnToLatest: string
    showLatest: string
    filesClassicDescription: string
    filesReconnected: string
    filesClearSearch: string
    filesRefresh: string
    fileGone: string
    fileVerificationFailed: string
    fileTimeout: string
    filesAccessUnavailable: string
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
    createAction: (count: number) => string
    created: (name: string, count: number) => string
    detailsSyncPending: string
    createFailed: string
    creating: string
    pickAtLeastTwo: string
    thisHost: string
    hostedFallbackToDesktop: (host: string) => string
    hostedAttachmentMemberUnavailable: (members: string) => string
    hostedSending: string
    hostedWorking: string
    hostedQueued: (host: string) => string
    hostedQueuedHint: (host: string) => string
    hostedNeedsAttention: string
    hostedSendFailed: (host: string) => string
    hostedStopping: string
    hostedStopped: string
    hostedStopQueued: (host: string) => string
    hostedStopQueuedHint: (host: string) => string
    hostedUnavailable: (host: string) => string
    hostedReconnectToStop: (host: string) => string
    hostedDeleted: string
    hostedDeleteLocally: string
    hostedMembersFixed: string
    hostedRenameQueued: (host: string) => string
    hostedRenameFailed: (host: string) => string
    hostRouteMissing: string
    hostUpdateNeeded: (host: string) => string
    hostReconnectToContinue: (host: string) => string
    hostedReconnectToDelete: (host: string) => string
    hostedSyncing: string
    continuityOnTitle: string
    continuityOnDesc: string
    continuityDesktopTitle: string
    continuityDesktopDesc: string
    continuityReadOnlyTitle: string
    continuityReadOnlyDesc: string
    retryTitle: string
    retryDesc: string
    retryAction: string
    reconnectAction: string
    reconnectingAction: string
    reconnectFailed: string
    botsNeedOneHost: string
    aBot: string
    memberUnavailable: (member: string) => string
    memberNeedsAttention: (member: string) => string
    memberReconnectToContinue: (member: string) => string
    memberCouldNotRespond: (member: string) => string
    memberRetryWhenOnline: (member: string) => string
    desktopStorageUnavailable: string
    hostedQueueRepaired: (count: number) => string
    hostedApprovalFailed: string
    hostedApprovalRetry: string
    hostRejectedCommand: string
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
    duplicate: 'Duplicate',
    duplicateFailed: 'Duplicate failed',
    deleteTitle: 'Delete bot and profile?',
    removeFromAllGroups: 'Remove from all groups',
    removeFromOtherGroups: 'Leave other groups',
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
    newDesc: 'Choose 2–6 Bots.',
    noBots: 'No bots yet. Create a bot first.',
    manageDesc: 'A Bot can join more than one Group Chat.',
    manageTitle: 'Manage groups',
    settingsTitle: 'Group settings',
    settingsDesc: 'Change the group name or picture. Members and history stay the same.',
    nameLabel: 'Group name',
    searchToAdd: 'Search bots to add',
    searchToAddPlaceholder: 'Search bots to add…',
    removeFromSelection: 'Remove from selection',
    disbandTitle: 'Disband group chat?',
    deleteTitle: 'Delete group chat?',
    deleteAction: 'Delete',
    composerPlaceholder: 'Message the group…',
    attachHint: 'Share files with this Group Chat',
    downloadAttachment: 'Download attachment',
    attachedFile: 'attached file',
    downloadFile: name => `Download ${name}`,
    attachmentDownloadFailed: 'This attachment could not be downloaded.',
    sharedFiles: 'Files',
    sharedFilesDescription: group => `Files shared in ${group}.`,
    searchSharedFiles: 'Search files',
    sharedFilesLoading: 'Loading files',
    sharedFilesError: 'Files could not be loaded.',
    sharedFilesExpired: 'Refresh the file list to continue.',
    sharedFilesOffline: 'Files are temporarily unavailable.',
    sharedFilesUnavailable: "File browsing isn't available for this Group Chat yet.",
    sharedFilesEmpty: 'No files shared yet.',
    sharedFilesPageEmpty: 'No files on this page',
    sharedFilesNoResults: 'No matching files.',
    sharedFilesRetry: 'Retry',
    olderFiles: 'Older',
    newerFiles: 'Newer',
    returnToLatest: 'Show latest',
    showLatest: 'Show latest',
    filesClassicDescription: 'Files available on this Desktop.',
    filesReconnected: 'Reconnected',
    filesClearSearch: 'Clear search',
    filesRefresh: 'Refresh list',
    fileGone: 'This file is no longer available.',
    fileVerificationFailed: "This file couldn't be verified. Nothing was downloaded.",
    fileTimeout: 'The download timed out.',
    filesAccessUnavailable: 'Files are unavailable for this Group Chat.',
    newThread: 'New Thread',
    reply: 'Reply',
    replyInThread: 'Reply in thread',
    replyInThreadPlaceholder: 'Reply in thread…',
    openThread: 'Open this thread',
    collapseThread: 'Collapse thread',
    collapseThreadLabel: 'Collapse this thread',
    activity: 'Activity',
    noActivityYet: 'No activity in this turn yet.',
    showActivity: 'Show group activity',
    hideActivity: 'Hide group activity',
    stop: 'Stop',
    stopHint: 'Stop the current Bot and pause the remaining turns',
    allHeldStatus: count => `All ${count} bots are paused`,
    heldMembersStatus: members => `Paused: ${members}`,
    holdReleaseHint: 'Mention a paused bot or send @all resume to release them.',
    needsYourInput: 'A bot in this group chat needs your input',
    pictureGenerationFailed: 'Group picture generation failed',
    createAction: count => `Create Group${count ? ` (${count})` : ''}`,
    created: (name, count) => `“${name}” created with ${count} bots`,
    detailsSyncPending: 'Some Bot details haven’t synced to your other devices.',
    createFailed: 'Could not create the Group Chat. Try again.',
    creating: 'Creating…',
    pickAtLeastTwo: 'Pick at least 2 bots',
    thisHost: 'this device',
    hostedFallbackToDesktop: host => `${host} can't keep this Group Chat running yet. Keep Desktop open.`,
    hostedAttachmentMemberUnavailable: members =>
      `Files cannot reach ${members || 'every Bot'} right now. Check that their devices are connected, then try again.`,
    hostedSending: 'Sending…',
    hostedWorking: 'Working',
    hostedQueued: host => `Waiting for ${host}`,
    hostedQueuedHint: host => `Saved. It will send when ${host} is online.`,
    hostedNeedsAttention: 'Needs attention',
    hostedSendFailed: host => `Not sent. Reconnect ${host} and retry.`,
    hostedStopping: 'Stopping…',
    hostedStopped: 'Stopped',
    hostedStopQueued: host => `Stop request saved. Waiting for ${host} to reconnect.`,
    hostedStopQueuedHint: host => `Waiting for ${host} to receive the stop request.`,
    hostedUnavailable: host => `${host} is offline`,
    hostedReconnectToStop: host => `Reconnect ${host} to stop this Group Chat.`,
    hostedDeleted: 'This Group Chat was deleted.',
    hostedDeleteLocally: 'Delete this Desktop’s copy of the group and its history.',
    hostedMembersFixed: 'Members cannot change while this Group Chat keeps running without Desktop.',
    hostedRenameQueued: host => `Rename saved. It will sync when ${host} is online.`,
    hostedRenameFailed: host => `Could not rename. Reconnect ${host} and retry.`,
    hostRouteMissing: 'This Group Chat connection is unavailable.',
    hostUpdateNeeded: host => `Update ${host} to keep this Group Chat running.`,
    hostReconnectToContinue: host => `Reconnect ${host} to continue.`,
    hostedReconnectToDelete: host => `Reconnect ${host} to delete this Group Chat.`,
    hostedSyncing: 'Syncing recent activity…',
    continuityOnTitle: 'Works without Desktop',
    continuityOnDesc: 'Bots can work after Desktop closes. The device running the group must stay online.',
    continuityDesktopTitle: 'Keep Desktop open',
    continuityDesktopDesc: 'Keep this Desktop open for the group to continue.',
    continuityReadOnlyTitle: 'Read-only history',
    continuityReadOnlyDesc: 'You can read this group’s history here, but cannot send messages from this connection.',
    retryTitle: 'Retry uncertain work?',
    retryDesc: 'The earlier attempt may have finished. Retrying could repeat actions.',
    retryAction: 'Retry',
    reconnectAction: 'Reconnect',
    reconnectingAction: 'Connecting…',
    reconnectFailed: 'Could not reconnect this Bot. Check that its device is online, then try again.',
    botsNeedOneHost: 'The selected Bots cannot continue when Desktop is closed.',
    aBot: 'A bot',
    memberUnavailable: member => `${member} is unavailable.`,
    memberNeedsAttention: member => `${member} needs your attention.`,
    memberReconnectToContinue: member => `Reconnect ${member} to continue this Group Chat.`,
    memberCouldNotRespond: member => `${member} could not respond.`,
    memberRetryWhenOnline: member => `${member} will retry when online.`,
    desktopStorageUnavailable: 'Desktop could not save this action. Try again.',
    hostedQueueRepaired: count =>
      `${count} damaged pending Group Chat ${count === 1 ? 'change was' : 'changes were'} removed; the rest were kept.`,
    hostedApprovalFailed: 'This approval is no longer available. Refresh the Group Chat and try again.',
    hostedApprovalRetry: 'Could not send this approval. Check the connection and try again.',
    hostRejectedCommand: 'The connected device rejected this action.',
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
    duplicate: '複製',
    duplicateFailed: '複製に失敗しました',
    deleteTitle: 'ボットとプロファイルを削除しますか？',
    removeFromAllGroups: 'すべてのグループから外す',
    removeFromOtherGroups: 'ほかのグループから外す',
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
    newDesc: '2〜6体のボットを選択してください。',
    noBots: 'ボットがまだありません。先にボットを作成してください。',
    manageDesc: 'ボットは複数のグループチャットに参加できます。',
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
    attachHint: 'ファイルを添付 — 応答するすべてのボットが見ます',
    downloadAttachment: '添付ファイルをダウンロード',
    attachedFile: '添付ファイル',
    downloadFile: name => `${name}をダウンロード`,
    attachmentDownloadFailed: 'この添付ファイルをダウンロードできませんでした。',
    sharedFiles: 'ファイル',
    sharedFilesDescription: group => `${group}で共有されたファイルです。`,
    searchSharedFiles: 'ファイルを検索',
    sharedFilesLoading: 'ファイルを読み込み中',
    sharedFilesError: 'ファイルを読み込めませんでした。',
    sharedFilesExpired: 'このファイル一覧の有効期限が切れました。',
    sharedFilesOffline: 'ファイルを一時的に利用できません。',
    sharedFilesUnavailable: 'このグループチャットではまだファイルを一覧表示できません。',
    sharedFilesEmpty: '共有されたファイルはまだありません。',
    sharedFilesPageEmpty: 'このページにファイルはありません',
    sharedFilesNoResults: '一致するファイルはありません',
    sharedFilesRetry: '再試行',
    olderFiles: '古いファイル',
    newerFiles: '新しいファイル',
    returnToLatest: '最新に戻る',
    showLatest: '最新を表示',
    filesClassicDescription: 'このDesktopで受け取ったファイルです。',
    filesReconnected: '再接続しました',
    filesClearSearch: '検索をクリア',
    filesRefresh: '一覧を更新',
    fileGone: 'このファイルは利用できなくなりました。',
    fileVerificationFailed: 'このファイルを検証できませんでした。何もダウンロードされていません。',
    fileTimeout: 'ダウンロードがタイムアウトしました。',
    filesAccessUnavailable: 'このグループチャットのファイルを利用できません。',
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
    createAction: count => `グループを作成${count ? ` (${count})` : ''}`,
    created: (name, count) => `「${name}」を${count}体のボットで作成しました`,
    detailsSyncPending: '一部のボット情報が他のデバイスにまだ同期されていません。',
    createFailed: 'グループチャットを作成できませんでした。もう一度お試しください。',
    creating: '作成中…',
    pickAtLeastTwo: '2体以上のボットを選択してください',
    thisHost: 'このデバイス',
    hostedFallbackToDesktop: host =>
      `${host} ではまだこのグループチャットを継続できません。Desktopを開いたままにしてください。`,
    hostedAttachmentMemberUnavailable: members =>
      `${members || '一部のボット'} にファイルを届けられません。該当するゲートウェイ接続を確認して、もう一度お試しください。`,
    hostedSending: '送信中…',
    hostedWorking: '作業中',
    hostedQueued: host => `${host} を待っています`,
    hostedQueuedHint: host => `保存しました。${host} がオンラインになると送信されます。`,
    hostedNeedsAttention: '確認が必要です',
    hostedSendFailed: host => `送信できませんでした。${host} を再接続して再試行してください。`,
    hostedStopping: '停止中…',
    hostedStopped: '停止しました',
    hostedStopQueued: host => `${host} への停止を保存しました`,
    hostedStopQueuedHint: host => `${host} がオンラインになると停止します。`,
    hostedUnavailable: host => `${host} はオフラインです`,
    hostedReconnectToStop: host => `このグループチャットを停止するには ${host} を再接続してください。`,
    hostedDeleted: 'このグループチャットは削除されました。',
    hostedDeleteLocally: 'ローカルのメンバーシップと履歴を削除するには、ここで削除してください。',
    hostedMembersFixed: 'Desktopなしで実行中のグループチャットではメンバーを変更できません。',
    hostedRenameQueued: host => `名前変更を保存しました。${host} がオンラインになると同期されます。`,
    hostedRenameFailed: host => `名前を変更できませんでした。${host} を再接続して再試行してください。`,
    hostRouteMissing: 'このグループチャットの接続を利用できません。',
    hostUpdateNeeded: host => `継続実行するには ${host} を更新してください。`,
    hostReconnectToContinue: host => `続行するには ${host} を再接続してください。`,
    hostedReconnectToDelete: host => `このグループチャットを削除するには ${host} を再接続してください。`,
    hostedSyncing: '最近のアクティビティを同期中…',
    continuityOnTitle: 'Desktopを閉じても大丈夫です',
    continuityOnDesc: 'このグループチャットのボットは作業を続けます。',
    continuityDesktopTitle: 'Desktopを開いたままにしてください',
    continuityDesktopDesc: 'Desktopを閉じると、このグループチャットは一時停止します。',
    continuityReadOnlyTitle: '閲覧専用の履歴',
    continuityReadOnlyDesc: 'このゲートウェイではグループチャットを表示できますが、実行を継続できません。',
    retryTitle: '不確かな作業を再試行しますか？',
    retryDesc: '前の試行が完了している可能性があります。再試行すると操作が重複する場合があります。',
    retryAction: '再試行',
    reconnectAction: '再接続',
    reconnectingAction: '接続中…',
    reconnectFailed: 'このボットを再接続できませんでした。ゲートウェイを確認して、もう一度お試しください。',
    botsNeedOneHost: '選択したボットはDesktopを閉じると継続できません。',
    aBot: 'ボット',
    memberUnavailable: member => `${member} は利用できません。`,
    memberNeedsAttention: member => `${member} に確認が必要です。`,
    memberReconnectToContinue: member => `このグループチャットを続けるには ${member} を再接続してください。`,
    memberCouldNotRespond: member => `${member} は応答できませんでした。`,
    memberRetryWhenOnline: member => `${member} はオンラインになると再試行します。`,
    desktopStorageUnavailable: 'Desktopでこの操作を保存できませんでした。もう一度お試しください。',
    hostedQueueRepaired: count => `破損した保留中のグループチャット変更 ${count} 件を削除し、残りは保持しました。`,
    hostedApprovalFailed: 'この承認は利用できなくなりました。グループチャットを更新して、もう一度お試しください。',
    hostedApprovalRetry: 'この承認を送信できませんでした。ゲートウェイ接続を確認して、もう一度お試しください。',
    hostRejectedCommand: '接続先がこの操作を拒否しました。',
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
    duplicate: '复制',
    duplicateFailed: '复制失败',
    deleteTitle: '删除机器人和配置档案？',
    removeFromAllGroups: '从所有群组中移除',
    removeFromOtherGroups: '退出其他群组',
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
    newDesc: '选择 2–6 个机器人。',
    noBots: '还没有机器人。请先创建一个机器人。',
    manageDesc: '一个机器人可以加入多个群聊。',
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
    attachHint: '附加文件 — 每个回应的机器人都能看到',
    downloadAttachment: '下载附件',
    attachedFile: '附件',
    downloadFile: name => `下载 ${name}`,
    attachmentDownloadFailed: '无法下载此附件。',
    sharedFiles: '文件',
    sharedFilesDescription: group => `${group} 中共享的文件。`,
    searchSharedFiles: '搜索文件',
    sharedFilesLoading: '正在加载文件',
    sharedFilesError: '无法加载文件。',
    sharedFilesExpired: '此文件列表已过期。',
    sharedFilesOffline: '文件暂时不可用。',
    sharedFilesUnavailable: '此群聊尚不支持文件浏览。',
    sharedFilesEmpty: '尚未共享任何文件。',
    sharedFilesPageEmpty: '此页没有文件',
    sharedFilesNoResults: '没有匹配的文件',
    sharedFilesRetry: '重试',
    olderFiles: '较早的文件',
    newerFiles: '较新的文件',
    returnToLatest: '返回最新内容',
    showLatest: '显示最新内容',
    filesClassicDescription: '此 Desktop 已接收的文件。',
    filesReconnected: '已重新连接',
    filesClearSearch: '清除搜索',
    filesRefresh: '刷新列表',
    fileGone: '此文件已不可用。',
    fileVerificationFailed: '无法验证此文件。未下载任何内容。',
    fileTimeout: '下载超时。',
    filesAccessUnavailable: '此群聊的文件不可用。',
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
    createAction: count => `创建群聊${count ? ` (${count})` : ''}`,
    created: (name, count) => `已创建“${name}”，包含 ${count} 个机器人`,
    detailsSyncPending: '部分机器人详情尚未同步到您的其他设备。',
    createFailed: '无法创建群聊。请重试。',
    creating: '正在创建…',
    pickAtLeastTwo: '请至少选择 2 个机器人',
    thisHost: '此设备',
    hostedFallbackToDesktop: host => `${host} 暂时无法保持此群聊运行。请保持 Desktop 打开。`,
    hostedAttachmentMemberUnavailable: members =>
      `文件目前无法送达${members || '所有机器人'}。请检查受影响的网关连接后重试。`,
    hostedSending: '正在发送…',
    hostedWorking: '正在工作',
    hostedQueued: host => `正在等待 ${host}`,
    hostedQueuedHint: host => `已保存。${host} 上线后将发送。`,
    hostedNeedsAttention: '需要处理',
    hostedSendFailed: host => `未发送。请重新连接 ${host} 后重试。`,
    hostedStopping: '正在停止…',
    hostedStopped: '已停止',
    hostedStopQueued: host => `已为 ${host} 保存停止请求`,
    hostedStopQueuedHint: host => `${host} 上线后将停止。`,
    hostedUnavailable: host => `${host} 已离线`,
    hostedReconnectToStop: host => `请重新连接 ${host} 以停止此群聊。`,
    hostedDeleted: '此群聊已被删除。',
    hostedDeleteLocally: '请在此处删除，以移除本地成员关系和历史记录。',
    hostedMembersFixed: '此群聊在没有 Desktop 的情况下运行时无法更改成员。',
    hostedRenameQueued: host => `重命名已保存。${host} 上线后将同步。`,
    hostedRenameFailed: host => `无法重命名。请重新连接 ${host} 后重试。`,
    hostRouteMissing: '此群聊连接不可用。',
    hostUpdateNeeded: host => `请更新 ${host} 以保持此群聊运行。`,
    hostReconnectToContinue: host => `请重新连接 ${host} 以继续。`,
    hostedReconnectToDelete: host => `请重新连接 ${host} 以删除此群聊。`,
    hostedSyncing: '正在同步近期活动…',
    continuityOnTitle: '可以关闭 Desktop',
    continuityOnDesc: '此群聊中的机器人会继续工作。',
    continuityDesktopTitle: '请保持 Desktop 打开',
    continuityDesktopDesc: '关闭 Desktop 会暂停此群聊。',
    continuityReadOnlyTitle: '只读历史记录',
    continuityReadOnlyDesc: '此网关可以显示此群聊，但无法让它持续运行。',
    retryTitle: '重试状态不确定的工作？',
    retryDesc: '之前的尝试可能已完成。重试可能会重复操作。',
    retryAction: '重试',
    reconnectAction: '重新连接',
    reconnectingAction: '正在连接…',
    reconnectFailed: '无法重新连接此机器人。请检查其网关后重试。',
    botsNeedOneHost: '关闭 Desktop 后，所选机器人无法继续工作。',
    aBot: '一个机器人',
    memberUnavailable: member => `${member} 不可用。`,
    memberNeedsAttention: member => `${member} 需要你的处理。`,
    memberReconnectToContinue: member => `请重新连接 ${member} 以继续此群聊。`,
    memberCouldNotRespond: member => `${member} 无法回复。`,
    memberRetryWhenOnline: member => `${member} 上线后将重试。`,
    desktopStorageUnavailable: 'Desktop 无法保存此操作。请重试。',
    hostedQueueRepaired: count => `已移除 ${count} 个损坏的待处理群聊更改，其余更改已保留。`,
    hostedApprovalFailed: '此审批已不可用。请刷新群聊后重试。',
    hostedApprovalRetry: '无法发送此审批。请检查网关连接后重试。',
    hostRejectedCommand: '连接的设备拒绝了此操作。',
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
    duplicate: '複製',
    duplicateFailed: '複製失敗',
    deleteTitle: '刪除機器人和設定檔？',
    removeFromAllGroups: '從所有群組中移除',
    removeFromOtherGroups: '退出其他群組',
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
    newDesc: '選擇 2–6 個機器人。',
    noBots: '還沒有機器人。請先建立一個機器人。',
    manageDesc: '一個機器人可以加入多個群組聊天。',
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
    attachHint: '附加檔案 — 每個回應的機器人都能看到',
    downloadAttachment: '下載附件',
    attachedFile: '附件',
    downloadFile: name => `下載 ${name}`,
    attachmentDownloadFailed: '無法下載此附件。',
    sharedFiles: '檔案',
    sharedFilesDescription: group => `${group} 中共享的檔案。`,
    searchSharedFiles: '搜尋檔案',
    sharedFilesLoading: '正在載入檔案',
    sharedFilesError: '無法載入檔案。',
    sharedFilesExpired: '此檔案清單已過期。',
    sharedFilesOffline: '檔案暫時無法使用。',
    sharedFilesUnavailable: '此群組聊天尚未支援檔案瀏覽。',
    sharedFilesEmpty: '尚未共享任何檔案。',
    sharedFilesPageEmpty: '此頁沒有檔案',
    sharedFilesNoResults: '找不到相符的檔案',
    sharedFilesRetry: '重試',
    olderFiles: '較舊的檔案',
    newerFiles: '較新的檔案',
    returnToLatest: '返回最新內容',
    showLatest: '顯示最新內容',
    filesClassicDescription: '此 Desktop 已接收的檔案。',
    filesReconnected: '已重新連線',
    filesClearSearch: '清除搜尋',
    filesRefresh: '重新整理清單',
    fileGone: '此檔案已無法使用。',
    fileVerificationFailed: '無法驗證此檔案。未下載任何內容。',
    fileTimeout: '下載逾時。',
    filesAccessUnavailable: '此群組聊天的檔案無法使用。',
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
    createAction: count => `建立群組聊天${count ? ` (${count})` : ''}`,
    created: (name, count) => `已建立「${name}」，包含 ${count} 個機器人`,
    detailsSyncPending: '部分機器人詳細資料尚未同步到您的其他裝置。',
    createFailed: '無法建立群組聊天。請再試一次。',
    creating: '正在建立…',
    pickAtLeastTwo: '請至少選擇 2 個機器人',
    thisHost: '此裝置',
    hostedFallbackToDesktop: host => `${host} 暫時無法保持此群組聊天運作。請保持 Desktop 開啟。`,
    hostedAttachmentMemberUnavailable: members =>
      `檔案目前無法送達${members || '所有機器人'}。請檢查受影響的閘道連線後再試一次。`,
    hostedSending: '正在傳送…',
    hostedWorking: '正在工作',
    hostedQueued: host => `正在等待 ${host}`,
    hostedQueuedHint: host => `已儲存。${host} 上線後將傳送。`,
    hostedNeedsAttention: '需要處理',
    hostedSendFailed: host => `未傳送。請重新連接 ${host} 後再試一次。`,
    hostedStopping: '正在停止…',
    hostedStopped: '已停止',
    hostedStopQueued: host => `已為 ${host} 儲存停止要求`,
    hostedStopQueuedHint: host => `${host} 上線後將停止。`,
    hostedUnavailable: host => `${host} 已離線`,
    hostedReconnectToStop: host => `請重新連接 ${host} 以停止此群組聊天。`,
    hostedDeleted: '此群組聊天已被刪除。',
    hostedDeleteLocally: '請在此處刪除，以移除本機成員關係和歷史記錄。',
    hostedMembersFixed: '此群組聊天在沒有 Desktop 的情況下運作時無法變更成員。',
    hostedRenameQueued: host => `重新命名已儲存。${host} 上線後將同步。`,
    hostedRenameFailed: host => `無法重新命名。請重新連接 ${host} 後再試一次。`,
    hostRouteMissing: '此群組聊天連線無法使用。',
    hostUpdateNeeded: host => `請更新 ${host} 以保持此群組聊天運作。`,
    hostReconnectToContinue: host => `請重新連接 ${host} 以繼續。`,
    hostedReconnectToDelete: host => `請重新連接 ${host} 以刪除此群組聊天。`,
    hostedSyncing: '正在同步近期活動…',
    continuityOnTitle: '可以關閉 Desktop',
    continuityOnDesc: '此群組聊天中的機器人會繼續工作。',
    continuityDesktopTitle: '請保持 Desktop 開啟',
    continuityDesktopDesc: '關閉 Desktop 會暫停此群組聊天。',
    continuityReadOnlyTitle: '唯讀歷史記錄',
    continuityReadOnlyDesc: '此閘道可以顯示此群組聊天，但無法讓它持續執行。',
    retryTitle: '重試狀態不確定的工作？',
    retryDesc: '先前的嘗試可能已完成。重試可能會重複操作。',
    retryAction: '重試',
    reconnectAction: '重新連接',
    reconnectingAction: '正在連接…',
    reconnectFailed: '無法重新連接此機器人。請檢查其閘道後再試一次。',
    botsNeedOneHost: '關閉 Desktop 後，所選機器人無法繼續工作。',
    aBot: '一個機器人',
    memberUnavailable: member => `${member} 無法使用。`,
    memberNeedsAttention: member => `${member} 需要您的處理。`,
    memberReconnectToContinue: member => `請重新連接 ${member} 以繼續此群組聊天。`,
    memberCouldNotRespond: member => `${member} 無法回覆。`,
    memberRetryWhenOnline: member => `${member} 上線後將重試。`,
    desktopStorageUnavailable: 'Desktop 無法儲存此操作。請再試一次。',
    hostedQueueRepaired: count => `已移除 ${count} 個損壞的待處理群組聊天變更，其餘變更已保留。`,
    hostedApprovalFailed: '此核准已無法使用。請重新整理群組聊天後再試一次。',
    hostedApprovalRetry: '無法傳送此核准。請檢查閘道連線後再試一次。',
    hostRejectedCommand: '已連接的裝置拒絕了此操作。',
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
export const BOTS_LOCALES: PluginLocaleBundles = {
  en,
  ja,
  zh,
  'zh-hant': zhHant,
  ar: {
    group: {
      attachedFile: 'ملف مرفق',
      downloadFile: (name: string) => `تنزيل ${name}`,
      attachmentDownloadFailed: 'تعذر تنزيل هذا المرفق.',
      sharedFiles: 'الملفات',
      sharedFilesDescription: (group: string) => `الملفات المشتركة في ${group}.`,
      searchSharedFiles: 'البحث في الملفات',
      sharedFilesLoading: 'جارٍ تحميل الملفات',
      sharedFilesError: 'تعذر تحميل الملفات.',
      sharedFilesExpired: 'انتهت صلاحية قائمة الملفات هذه.',
      sharedFilesOffline: 'الملفات غير متاحة مؤقتاً.',
      sharedFilesUnavailable: 'تصفح الملفات غير متاح لهذه المحادثة الجماعية بعد.',
      sharedFilesEmpty: 'لم تتم مشاركة أي ملفات بعد.',
      sharedFilesPageEmpty: 'لا توجد ملفات في هذه الصفحة',
      sharedFilesNoResults: 'لا توجد ملفات مطابقة',
      sharedFilesRetry: 'إعادة المحاولة',
      olderFiles: 'ملفات أقدم',
      newerFiles: 'ملفات أحدث',
      returnToLatest: 'العودة إلى الأحدث',
      showLatest: 'عرض الأحدث',
      filesClassicDescription: 'الملفات التي تلقاها هذا Desktop.',
      filesReconnected: 'تمت إعادة الاتصال',
      filesClearSearch: 'مسح البحث',
      filesRefresh: 'تحديث القائمة',
      fileGone: 'لم يعد هذا الملف متاحاً.',
      fileVerificationFailed: 'تعذر التحقق من هذا الملف. لم يتم تنزيل أي شيء.',
      fileTimeout: 'انتهت مهلة التنزيل.',
      filesAccessUnavailable: 'الملفات غير متاحة لهذه المحادثة الجماعية.'
    }
  },
  ru: {
    group: {
      attachedFile: 'вложенный файл',
      downloadFile: (name: string) => `Скачать ${name}`,
      attachmentDownloadFailed: 'Не удалось скачать это вложение.',
      sharedFiles: 'Файлы',
      sharedFilesDescription: (group: string) => `Файлы, опубликованные в ${group}.`,
      searchSharedFiles: 'Поиск файлов',
      sharedFilesLoading: 'Загрузка файлов',
      sharedFilesError: 'Не удалось загрузить файлы.',
      sharedFilesExpired: 'Срок действия этого списка файлов истёк.',
      sharedFilesOffline: 'Файлы временно недоступны.',
      sharedFilesUnavailable: 'Просмотр файлов пока недоступен для этого группового чата.',
      sharedFilesEmpty: 'Файлами ещё не делились.',
      sharedFilesPageEmpty: 'На этой странице нет файлов',
      sharedFilesNoResults: 'Подходящие файлы не найдены',
      sharedFilesRetry: 'Повторить',
      olderFiles: 'Более старые файлы',
      newerFiles: 'Более новые файлы',
      returnToLatest: 'Вернуться к последним',
      showLatest: 'Показать последние',
      filesClassicDescription: 'Файлы, полученные этим Desktop.',
      filesReconnected: 'Соединение восстановлено',
      filesClearSearch: 'Очистить поиск',
      filesRefresh: 'Обновить список',
      fileGone: 'Этот файл больше недоступен.',
      fileVerificationFailed: 'Не удалось проверить этот файл. Ничего не скачано.',
      fileTimeout: 'Время ожидания скачивания истекло.',
      filesAccessUnavailable: 'Файлы недоступны для этого группового чата.'
    }
  }
}

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
