/**
 * Bot Mode's wording in the redesigned shell: ONLY the strings whose words differ from the base
 * bundles (`i18n.ts`), which keep `main`'s classic wording verbatim. Registered as the `redesign`
 * layer of `ctx.i18n.register`, so the redesign reads "agent", "machine" and "schedule" where
 * classic reads "bot", "profile", "gateway" and "job" (DESIGN.md, Terminology: classic vs redesign).
 * A key missing here reads the same in both shells; a locale missing a key falls back to English
 * in the same shell.
 */

import type { PluginLocaleBundles } from '@hermes/plugin-sdk'

import { russianNoun } from './group-chat-locales'
import type { BotsMessages } from './i18n-messages'

type BotsWording = { [Namespace in keyof BotsMessages]?: Partial<BotsMessages[Namespace]> }

function arabicAgents(count: number, dual = 'وكيلان'): string {
  return count === 0
    ? 'لا وكلاء'
    : count === 1
      ? 'وكيل واحد'
      : count === 2
        ? dual
        : `${count} ${count % 100 >= 3 && count % 100 <= 10 ? 'وكلاء' : count % 100 >= 11 ? 'وكيلًا' : 'وكيل'}`
}

const en: BotsWording = {
  avatar: {
    blobFromName: 'Blob face — drawn from the agent’s name',
    pickPet: 'Pick a pet as this agent’s picture.',
    tabBot: 'Agent',
    unlockFollowsName: 'Unlock — the face follows the agent’s name again'
  },
  bot: {
    attentionBlocked: 'Agent is blocked — see its last message',
    attentionProviderAuth: 'Sign in again for this agent',
    createFailed: 'Could not create the agent yet',
    createFirstHint: 'Open the Agents pane and hit “New agent”.',
    deleteTitle: 'Delete agent?',
    descriptionHint: 'Leave blank to generate from the agent’s name and description.',
    editTitle: 'Edit agent',
    groupsLoadFailed: 'Could not load agent groups',
    helpPromptPlaceholder: 'What should this agent help with?',
    hiddenToast: name => `${name} hidden — use the eye button in the Agents header to see hidden agents`,
    loadFailed: 'Could not load agent',
    metadataLoadFailed: 'Could not load agent metadata',
    newChatWith: 'New chat with this agent',
    newTitle: 'New agent',
    openAnotherChatUnsupported: 'Update Hermes Desktop to open another agent chat.',
    openBotChat: 'Open agent chat',
    openBotChatInNewTab: 'Open agent chat in new tab',
    openGateways: 'Open Machines',
    openNeedsUpdateTitle: 'This agent lives on an older Hermes',
    openUnreachableTitle: 'Hermes couldn’t reach the computer this agent runs on',
    remoteConnectionsUnsupported: 'Update Hermes Desktop to chat with agents on other connections.'
  },
  canonical: {
    createDuplicateMember:
      'A group chat hosted on this machine needs distinct agents with unique handles; @all and @everyone are reserved.',
    createEnded:
      'Hermes refused this group chat, so it may have been changed or removed somewhere else. Try again to start a new one.',
    createGroupSize: 'A group chat hosted on this machine needs two to six agents.',
    createOtherConnection: 'A group chat hosted on this machine can only seat agents on this connection.',
    createRefused: 'Hermes couldn’t create this group chat.',
    createUncertain:
      'Hermes didn’t answer, so this group chat may already exist. Try again to open it. It won’t be created twice.',
    disbandMemberHomeUnknown:
      'The group chat may still be working, but one of its agents is no longer configured on the computer running the group chat. Add that agent back to hosted_rooms.profiles so its Stop can be confirmed, then try Disband again.',
    disbandServiceRequired:
      'Hermes must be running on the computer that runs this group chat to stop its agents and remove their access. Start or reconnect Hermes there, then try Disband again.',
    disbandStopPending:
      'The agents in this group chat haven’t all confirmed they’ve stopped. Try Disband again once they have. For a waiting turn, use Retry or Discard if offered in the group chat.',
    disbandUnconfirmed:
      'We couldn’t confirm that the group chat ended. An agent’s computer may be asleep or offline. Try again once it’s back.',
    discardWarning:
      'Side effects may already have occurred. Discarding does not undo them. For an agent on another computer, Discard waits for that computer to confirm the turn has stopped or never started.',
    driverUnavailable:
      'This group chat isn’t available. Update Hermes on the machine that hosts it, or reconnect to that machine.',
    groupDriverStopped:
      'Group chats can’t open here right now: group chats aren’t running on this connection’s machine. Restart or update Hermes on that machine, then try again. Nothing is lost: earlier messages stay where they are.',
    groupsUnreachable:
      'Couldn’t reach this connection’s machine to open this group chat. Check that Hermes is running there and that you’re connected, then try again. Nothing is lost: earlier messages stay where they are.',
    legacyRoom:
      'Desktop ran this group chat itself, so it can’t continue on this machine. Start a new group with the same agents: it keeps working when Desktop is closed. Earlier messages aren’t copied into it.',
    legacyRoomBlocked:
      'Desktop ran this group chat itself, so it can’t continue on this machine, and these agents can’t form a group chat hosted on this machine as they are (the reason is below). Start a new group chat and choose its agents. Earlier messages aren’t copied into it.',
    legacyRoomElsewhere:
      'This group chat can’t continue here: it has agents on other computers, and a group chat hosted on this machine seats agents from one connection only. Switch to the connection where those agents run and start a group there, or start a new group chat with agents from this connection. Earlier messages aren’t copied into a new group.',
    otherConnections: 'Agents on other connections: {connections}',
    peerJoinChanged:
      '{name} can’t join this group chat: its computer’s agent settings changed since the group chat was created. Start a new group chat to include it.',
    peerStartWithoutConfirm:
      'A new group chat starts with the other agents. The original ends only after they all join and every original member confirms Stop and access revocation. If either step fails, both group chats stay. Earlier messages aren’t copied. Using this action again after creation makes another group chat.',
    peerStartWithoutRetry: 'Use Retry for these agents in the new group chat.',
    peerTargetSetup: '{connection} can’t host an agent for a group chat on another computer yet. {reason}',
    readTimedOut:
      'Hermes didn’t answer in time. It may be waiting for an agent’s computer. Choose Refresh to try again.',
    reasonAgentBlocked: 'The agent is blocked',
    reasonGatewayRestarted: 'Hermes on the agent’s machine restarted',
    reasonMemberUnavailable: 'The agent is unavailable',
    reasonPrepareFailed: 'The agents could not be started',
    reasonRuntimeOffline: 'The agent is offline',
    reasonSlotWait: 'Too many agents are running',
    reasonTargetBusy: 'The agent is busy',
    reasonTargetScope: 'The agent could not access its credentials',
    sendRefused: 'Hermes refused this message. Edit it and send again.',
    turnNoAnswer: 'the agent didn’t answer in time'
  },
  cron: {
    botChatTarget: bot => `${bot}’s chat (agent responds)`,
    detailDescription: 'What this schedule runs, and when it runs next.',
    filterHint:
      'Schedules exist for this agent but none have its tag. Name a schedule "[bot:<name>] …" to show it here, or see them in Schedules below.',
    instructionNul: 'Schedule instruction cannot contain NUL (U+0000).',
    legacyUnsafe: 'Paused for security: delete and recreate this legacy schedule before running it again.',
    nameNul: 'Schedule name cannot contain NUL (U+0000).',
    needsRosterFirst: 'This agent has to appear in the roster first.',
    staleNotice: 'Could not refresh schedules. Showing the last list we had.',
    untitled: 'Untitled schedule'
  },
  crossGateway: {
    gatewayPeer: (owner, local = false) =>
      `Runs on ${local ? 'this device' : owner}, including when this Desktop is closed. Only text is sent to agents on other computers.`,
    ownerCannotSeatPeers: place =>
      `Hermes on ${place} can’t seat agents from other computers yet. Update Hermes there and run its gateway (hermes gateway run), then retry.`,
    peerGatewayChanged: place =>
      `${place} now runs a different Hermes installation than this agent came from. Reconnect it, then retry.`,
    peerRouteMissing: 'A selected agent’s computer isn’t connected. Reconnect it, then retry.',
    pickBots: max => `Pick 2–${max} agents.`,
    selectionElsewhere: connection =>
      `These agents all live on ${connection}. Add an agent from this connection, or switch to ${connection} to start a group of its agents.`
  },
  editor: {
    cloneFrom: 'Clone from agent',
    cloneFromOn: target => `Clone from agent (on ${target})`,
    createBot: 'Create agent',
    created: name => `Agent "${name}" created`,
    createdOn: (name, target) => `Agent "${name}" created on ${target}`,
    freshProfile: 'Fresh agent (bundled skills)',
    fullConfigHint: 'Full configuration needs a newer Hermes version (update Hermes, then restart it).',
    gatewayDefault: 'machine default',
    inheritLaunch: 'Inherit (launch agent)',
    inheritedModel: 'inherited from launch agent',
    mcpHint:
      'Configured servers copy from the main agent; catalog entries are the bundled MCP menu. Entries needing API keys route through setup first (credentials follow the shared keys setting).',
    nameFirstHint: 'Name the agent first — a draft agent is created when you open this tab (discarded if you cancel).',
    newerGateway: 'Capability catalog needs a newer Hermes version (update Hermes, then restart it).',
    noImageModel:
      'No image model available. If you just enabled one (or updated Hermes), restart Hermes: Ctrl+K → "Restart Hermes".',
    remoteHint: target =>
      `The agent is created on ${target} and appears in the roster as a Connections agent. Chat routes to that machine.`,
    shareKeys: 'Share keys & accounts with the main agent',
    shareKeysOn: target => `Share keys & accounts with the default agent on ${target}`
  },
  group: {
    activityBot: 'An agent',
    activitySlotWait: "couldn't start — too many agents running",
    allHeldStatus: count => (count === 1 ? 'The agent is paused' : `All ${count} agents are paused`),
    attachHint: 'Attach files — every responding agent sees them',
    composerPlaceholder: 'Say something — every agent in this group chat sees your message.',
    createBotFirst: 'No agents yet — create one first.',
    noSeatableBots: 'None of your agents can join a group chat here yet:',
    cannotJoinHere: 'These agents can’t join a group chat here:',
    nonePicked: 'No agents picked yet',
    unseatableHere:
      'Hermes on this machine doesn’t seat it in group chats — it isn’t served here, or it’s being retired',
    createDesc: max =>
      `Pick 2–${max} agents. Local memberships sync through each agent; for agents on other computers, membership is saved in this group chat only.`,
    created: (group, count) => `“${group}” created with ${count} agent${count === 1 ? '' : 's'}`,
    deleteDesc: group =>
      `This removes “${group}” from its agents and clears the shared group chat history. The agents and their individual chats are kept.`,
    disbandDescSuffix: count =>
      ` grouping from its ${count} agent${count === 1 ? '' : 's'} and clears the shared group chat log. ${count === 1 ? 'The agent itself and its per-group sessions are' : 'The agents themselves and their per-group sessions are'} kept.`,
    dropToRoom: 'Drop to attach — every responding agent sees it',
    everyoneMeta: 'Every agent in the group chat',
    holdReleaseHint: 'Mention a paused agent or send @all resume to release them.',
    manageDesc: 'An agent can join multiple group chats. Memberships sync to every machine.',
    manageMembersDesc: (max, group) =>
      `Pick 2–${max} agents for “${group}”. The group chat, its history and its member sessions stay as they are.`,
    memberCount: count => `${count} agent${count === 1 ? '' : 's'}`,
    memberSizeError: max => `A group chat needs between 2 and ${max} agents`,
    mentionBot: 'Agent in this group chat',
    needsYourInput: 'An agent in this group chat needs your input',
    noBots: 'No agents in this group chat',
    noBotsMatch: query => `No agents match “${query}”`,
    noMembersToSend: group =>
      `${group} has no members to send to — add an agent, or reopen the group chat if members are still loading.`,
    searchToAdd: 'Search agents to add',
    searchToAddPlaceholder: 'Search agents to add…',
    sendRouteRequired: 'Sending to this group chat requires its connection, agent and group chat ID',
    slashCommandsUnsupported:
      'Slash commands are not supported in group chats. Open an individual agent chat to use them.'
  },
  roster: {
    allGateways: 'All machines',
    allHidden: 'All agents are hidden',
    botsAndGroups: 'Agents and group chats',
    botsOnly: 'Agents only',
    currentGateway: 'Current machine',
    emptyDesc: 'Create your first agent.',
    emptyTitle: 'No agents yet',
    gatewayRemoved: 'Machine removed',
    newBotOrGroup: 'New agent or group chat',
    noHiddenMatch: 'No hidden agents match these filters.',
    noMatchFilters: 'No agents or group chats match these filters.',
    noMatchFiltersOn: gateway => `No agents or group chats match these filters on ${gateway}`,
    noMatchQuery: query => `No agents or group chats match “${query}”`,
    noMatchQueryOn: (query, gateway) => `No agents or group chats match “${query}” on ${gateway}`,
    rosterUnavailable: reason =>
      `Roster unavailable: ${reason}. If your Hermes version predates profiles.list, update Hermes and restart it.`,
    search: 'Search agents and group chats',
    searchPlaceholder: 'Search agents and group chats…',
    showHidden: 'Show hidden agents',
    title: 'Agents',
    waitingForGateway:
      'Waiting for the machine connection… (remote machines can take a few seconds; retries automatically)'
  },
  screen: {
    agentControls: 'Agent is in control',
    autoOpenMenu: 'Open Screen when the agent uses it',
    imageSwitchBody: (current, target) =>
      `Your sandbox still runs ${current}, which has no desktop. Switching to ${target} recreates the container the next time the agent uses its terminal: files in /root and /workspace stay on this machine, packages installed inside the container are reinstalled on demand.`,
    installHint: 'Runs on the machine as the user Hermes runs as; sudo is asked for once, through Hermes.',
    managedNotIncludedBody:
      'Hermes Cloud includes Screen on agents with 8 GB of memory or more. It can’t be installed on a smaller agent.',
    managedNotIncludedTitle: 'Screen isn’t included on this agent',
    needsHttps:
      'Agent screens need a secure connection. Open the Webapp over HTTPS, or through an SSH tunnel to 127.0.0.1, to view this one.',
    noPackageManager: 'No supported package manager (apt, dnf, pacman) was found on the machine.',
    notInstalledBody: 'The machine needs TigerVNC and the Xfce core to give this agent a screen. Run on the machine:',
    openNeedsUpdate: 'Update Hermes Desktop to open agent screens.',
    portalUnavailable: 'Update the agent’s Hermes to use Screen',
    portalWatching: 'Live · agent in control',
    stoppedBody: 'Start this agent’s desktop to watch what it does and take over when it needs you.',
    unsupportedBody: 'Agent screens run on Linux machines. This agent uses the machine’s own display.',
    unsupportedTitle: 'No agent screen on this host'
  },
  sections: {
    deleted: (name, count) =>
      count === 0
        ? `Deleted “${name}”`
        : `Deleted “${name}” — ${count} ${count === 1 ? 'agent' : 'agents'} moved to Unassigned`,
    emptyHint: 'Drag agents here',
    headingTip: 'Drop agents here · double-click to rename'
  },
  tools: {
    needsSetup: keys => `needs setup (${keys}) — restart Hermes to enable in-app setup`,
    noTarget: 'No target agent'
  }
}

const ja: BotsWording = {
  avatar: {
    blobFromName: 'ブロブ顔 — エージェントの名前から描画',
    pickPet: 'このエージェントのプロフィール画像としてペットを選びます。',
    tabBot: 'エージェント',
    unlockFollowsName: 'ロック解除 — 顔がエージェントの名前に再び追従します'
  },
  bot: {
    attentionBlocked: 'エージェントがブロックされています — 最後のメッセージを確認してください',
    attentionProviderAuth: 'このエージェントで再度サインインしてください',
    createFailed: 'エージェントをまだ作成できませんでした',
    createFirstHint: 'エージェントパネルを開いて「新しいエージェント」を押してください。',
    deleteTitle: 'エージェントを削除しますか？',
    descriptionHint: '空欄のままにすると、エージェントの名前と説明から生成します。',
    editTitle: 'エージェントを編集',
    groupsLoadFailed: 'エージェントのグループを読み込めませんでした',
    helpPromptPlaceholder: 'このエージェントは何を手伝いますか？',
    hiddenToast: name =>
      `${name}を非表示にしました — エージェントヘッダーの目のボタンで非表示のエージェントを表示できます`,
    loadFailed: 'エージェントを読み込めませんでした',
    metadataLoadFailed: 'エージェントのメタデータを読み込めませんでした',
    newChatWith: 'このエージェントと新しいチャット',
    newTitle: '新しいエージェント',
    openAnotherChatUnsupported: '別のエージェントチャットを開くには Hermes Desktop を更新してください。',
    openBotChat: 'エージェントチャットを開く',
    openBotChatInNewTab: 'エージェントチャットを新しいタブで開く',
    openGateways: 'マシンを開く',
    openNeedsUpdateTitle: 'このエージェントは古い Hermes 上で動いています',
    openUnreachableTitle: 'このエージェントが動いているコンピューターに Hermes が接続できませんでした',
    remoteConnectionsUnsupported: '他の接続上のエージェントとチャットするには Hermes Desktop を更新してください。'
  },
  canonical: {
    createDuplicateMember:
      'このマシンで運用されるグループチャットには、それぞれ固有のハンドルを持つ別々のエージェントが必要です。@all と @everyone は予約済みです。',
    createEnded:
      'Hermes がこのグループチャットを拒否しました。別の場所で変更または削除された可能性があります。もう一度試すと新しく作成します。',
    createGroupSize: 'このマシンで運用されるグループチャットには 2〜6 体のエージェントが必要です。',
    createOtherConnection:
      'このマシンで運用されるグループチャットには、この接続上のエージェントだけを参加させられます。',
    createRefused: 'Hermes はこのグループチャットを作成できませんでした。',
    createUncertain:
      'Hermes から応答がなかったため、このグループチャットはすでに作成されている可能性があります。もう一度試すと開きます。二重に作成されることはありません。',
    disbandMemberHomeUnknown:
      'このグループチャットの処理はまだ続いている可能性がありますが、エージェントの一つがグループチャットを実行しているコンピューターの設定から外れています。そのエージェントを hosted_rooms.profiles に戻して停止を確認できるようにしてから、もう一度解散してください。',
    disbandServiceRequired:
      'エージェントを停止してアクセスを解除するには、このグループチャットを実行するコンピューターで Hermes が動作している必要があります。そこで Hermes を起動するか再接続してから、もう一度解散してください。',
    disbandStopPending:
      'このグループチャットのすべてのエージェントから停止の確認が取れたわけではありません。全員の確認が取れたら、もう一度解散してください。保留中のターンには、グループチャットに再試行や破棄が表示されていれば、その操作を使えます。',
    disbandUnconfirmed:
      'グループチャットの終了を確認できませんでした。エージェントのコンピューターがスリープ中かオフラインの可能性があります。復帰してからもう一度お試しください。',
    discardWarning:
      'すでに変更が生じている場合があります。破棄しても元には戻りません。別のコンピューターのエージェントでは、そのコンピューターがターンの停止または未開始を確認するまで待ちます。',
    driverUnavailable:
      'このグループチャットは利用できません。実行元のマシンの Hermes を更新するか、そのマシンに再接続してください。',
    groupDriverStopped:
      'いまはここでグループチャットを開けません。この接続先のマシンでグループチャットが動いていないためです。そのマシンの Hermes を再起動または更新してから、もう一度お試しください。何も失われません。以前のメッセージはそのまま残ります。',
    groupsUnreachable:
      'このグループチャットを開くための接続先のマシンに接続できませんでした。そのマシンで Hermes が動いていて、接続されていることを確認してから、もう一度お試しください。何も失われません。以前のメッセージはそのまま残ります。',
    legacyRoom:
      'このグループチャットは Desktop 自身が動かしていたため、このマシンでは続けられません。同じエージェントで新しいグループを始めると、Desktop を閉じても動き続けます。以前のメッセージは新しいグループにコピーされません。',
    legacyRoomBlocked:
      'このグループチャットは Desktop 自身が動かしていたため、このマシンでは続けられません。また、これらのエージェントはこのままではこのマシンで運用されるグループチャットを作れません（理由は下記）。新しいグループチャットを始めてエージェントを選んでください。以前のメッセージはコピーされません。',
    legacyRoomElsewhere:
      'このグループチャットはここでは続けられません。ほかのコンピューター上のエージェントが含まれていて、このマシンで運用されるグループチャットには 1 つの接続のエージェントしか参加させられないためです。それらのエージェントが動いている接続に切り替えてそこでグループを始めるか、この接続のエージェントで新しいグループチャットを始めてください。以前のメッセージは新しいグループにコピーされません。',
    otherConnections: 'ほかの接続上のエージェント: {connections}',
    peerJoinChanged:
      '{name} はこのグループチャットに参加できません。グループチャットの作成後に、そのコンピューターのエージェントの設定が変わりました。含めるには新しいグループチャットを始めてください。',
    peerStartWithoutConfirm:
      'ほかのエージェントで新しいグループチャットを作成します。全員が参加し、元の全メンバーの停止とアクセス取り消しが確認されてから、元のチャットを終了します。どちらかが失敗した場合は両方を保持します。以前のメッセージはコピーされません。 作成後にこの操作をもう一度行うと、別のグループチャットが作成されます。',
    peerStartWithoutRetry: '新しいグループチャットで、これらのエージェントの「再試行」を使ってください。',
    peerTargetSetup:
      '{connection} は、まだほかのコンピューターのグループチャットにエージェントを参加させられません。{reason}',
    readTimedOut:
      'Hermes が時間内に応答しませんでした。エージェントのコンピューターの応答を待っている可能性があります。「更新」を選んで再試行してください。',
    reasonAgentBlocked: 'エージェントはブロックされています',
    reasonGatewayRestarted: 'エージェントのマシンの Hermes が再起動しました',
    reasonMemberUnavailable: 'エージェントを利用できません',
    reasonPrepareFailed: 'エージェントを起動できませんでした',
    reasonRuntimeOffline: 'エージェントはオフラインです',
    reasonSlotWait: '実行中のエージェントが多すぎます',
    reasonTargetBusy: 'エージェントは処理中です',
    reasonTargetScope: 'エージェントが認証情報にアクセスできませんでした',
    sendRefused: 'Hermes がこのメッセージを拒否しました。編集してからもう一度送信してください。',
    turnNoAnswer: 'エージェントが時間内に応答しませんでした'
  },
  cron: {
    botChatTarget: bot => `${bot}のチャット（エージェントが応答）`,
    detailDescription: 'このスケジュールの実行内容と次回の実行日時。',
    filterHint:
      'このエージェントにはスケジュールがありますが、タグが付いたものはありません。スケジュール名を「[bot:<名前>] …」にするとここに表示されます。下のスケジュールでも確認できます。',
    instructionNul: 'スケジュールの指示に NUL (U+0000) は使用できません。',
    legacyUnsafe: '安全のため一時停止中です。この旧形式のスケジュールを削除して作り直してから実行してください。',
    nameNul: 'スケジュール名に NUL (U+0000) は使用できません。',
    needsRosterFirst: 'このエージェントは先に名簿に表示される必要があります。',
    staleNotice: 'スケジュールを更新できませんでした。最後に取得したリストを表示しています。',
    untitled: '無題のスケジュール'
  },
  crossGateway: {
    gatewayPeer: (owner, local = false) =>
      `このデスクトップを閉じていても ${local ? 'このデバイス' : owner} で動作します。ほかのコンピューターのエージェントはテキストのみを受け取ります。`,
    ownerCannotSeatPeers: place =>
      `${place} の Hermes は、まだほかのコンピューターのエージェントを参加させられません。そこで Hermes を更新し、ゲートウェイ（hermes gateway run）を実行してから、再試行してください。`,
    peerGatewayChanged: place =>
      `${place} は、このエージェントとは別の Hermes インストールで動作しています。再接続してから、再試行してください。`,
    peerRouteMissing: '選択したエージェントのコンピューターが接続されていません。再接続してから、再試行してください。',
    pickBots: max => `2〜${max} 個のエージェントを選択してください。`,
    selectionElsewhere: connection =>
      `これらのエージェントはすべて ${connection} にあります。この接続のエージェントを追加するか、${connection} に切り替えてそのエージェントだけのグループを作成してください。`
  },
  editor: {
    cloneFrom: '複製元のエージェント',
    cloneFromOn: target => `複製元のエージェント（${target} 上）`,
    createBot: 'エージェントを作成',
    createEmpty: '空のエージェントを作成（同梱スキルを除外）',
    created: name => `エージェント「${name}」を作成しました`,
    createdOn: (name, target) => `${target} にエージェント「${name}」を作成しました`,
    emptySkillsHint: '「空のエージェントを作成」が選択されているため、同梱スキルはインストールされません。',
    freshProfile: '新規エージェント（同梱スキル）',
    fullConfigHint:
      'すべての設定を使うには新しいバージョンの Hermes が必要です（Hermes を更新して再起動してください）。',
    gatewayDefault: 'マシンの既定値',
    inheritLaunch: '継承（起動エージェント）',
    inheritedModel: '起動時のエージェントから継承',
    mcpHint:
      '設定済みサーバーはメインエージェントからコピーされます。カタログは同梱 MCP メニューです。API キーが必要な項目は先に設定を行います（認証情報はキー共有設定に従います）。',
    nameFirstHint:
      '先にエージェントに名前を付けてください。このタブを開くと下書きエージェントが作成されます（キャンセルすると破棄されます）。',
    newerGateway: '機能カタログには新しいバージョンの Hermes が必要です（Hermes を更新して再起動してください）。',
    noImageModel:
      '画像モデルがありません。有効にした直後や Hermes 更新後の場合は、Ctrl+K →「Hermes を再起動」で Hermes を再起動してください。',
    remoteHint: target =>
      `エージェントは ${target} に作成され、接続先のエージェントとして一覧に表示されます。チャットはそのマシンに送られます。`,
    shareKeys: 'メインエージェントとキー・アカウントを共有',
    shareKeysOn: target => `${target} の default エージェントとキー・アカウントを共有`
  },
  group: {
    activityBot: 'エージェント',
    activitySlotWait: '開始できませんでした — 実行中のエージェントが多すぎます',
    allHeldStatus: count => `すべてのエージェント（${count}体）が一時停止中`,
    attachHint: 'ファイルを添付 — 応答するすべてのエージェントが見ます',
    composerPlaceholder: '何か書いてください — このグループチャットのすべてのエージェントがメッセージを受け取ります。',
    createBotFirst: 'まだエージェントがいません。まず作成してください。',
    noSeatableBots: 'ここではまだどのエージェントもグループチャットに参加できません：',
    cannotJoinHere: 'これらのエージェントはここではグループチャットに参加できません：',
    nonePicked: 'まだエージェントが選ばれていません',
    unseatableHere:
      'このマシンの Hermes はこのエージェントをグループチャットに参加させません — ここで提供されていないか、廃止中です',
    createDesc: max =>
      `エージェントを2〜${max}体選んでください。同じコンピューター上のメンバー情報は各エージェントを通じて同期されます。ほかのコンピューターのエージェントの参加情報は、このグループチャット内だけで保持されます。`,
    created: (group, count) => `エージェント${count}体で「${group}」を作成しました`,
    deleteDesc: group =>
      `「${group}」からエージェントを外し、グループチャットの共有履歴を消去します。エージェントと個別のチャットは保持されます。`,
    disbandDescSuffix: count =>
      `からエージェント${count}体を外し、グループチャットの共有履歴を消去します。エージェント自体と各グループのセッションは保持されます。`,
    dropToRoom: 'ドロップして添付 — 応答するすべてのエージェントが見られます',
    everyoneMeta: 'グループチャット内のすべてのエージェント',
    holdReleaseHint: '一時停止中のエージェントにメンションするか、@all resume を送信して再開します。',
    manageDesc: 'エージェントは複数のグループチャットに参加できます。メンバーシップはすべてのマシンに同期されます。',
    manageMembersDesc: (max, group) =>
      `「${group}」のエージェントを2〜${max}体選んでください。グループチャット、履歴、メンバーのセッションはそのまま保持されます。`,
    memberCount: count => `エージェント${count}体`,
    memberSizeError: max => `グループチャットには2〜${max}体のエージェントが必要です`,
    mentionBot: 'このグループチャットのエージェント',
    needsYourInput: 'このグループチャットのエージェントが入力を待っています',
    noBots: 'このグループチャットにはエージェントがいません',
    noBotsMatch: query => `「${query}」に一致するエージェントはありません`,
    noMembersToSend: group =>
      `${group} に送信先のメンバーがいません。エージェントを追加するか、メンバーの読み込み中であればグループチャットを開き直してください。`,
    searchToAdd: '追加するエージェントを検索',
    searchToAddPlaceholder: '追加するエージェントを検索…',
    sendRouteRequired: 'このグループチャットへの送信には接続、エージェント、グループチャットIDが必要です',
    slashCommandsUnsupported:
      'グループチャットではスラッシュコマンドを使用できません。個別のエージェントチャットを開いて使用してください。'
  },
  roster: {
    allGateways: 'すべてのマシン',
    allHidden: 'すべてのエージェントが非表示です',
    botsAndGroups: 'エージェントとグループチャット',
    botsOnly: 'エージェントのみ',
    currentGateway: '現在のマシン',
    emptyDesc: '最初のエージェントを作成しましょう。',
    emptyTitle: 'エージェントはまだありません',
    gatewayRemoved: 'マシンが削除されました',
    newBotOrGroup: '新しいエージェントまたはグループチャット',
    noHiddenMatch: 'これらのフィルタに一致する非表示エージェントはありません。',
    noMatchFilters: 'これらのフィルタに一致するエージェントやグループチャットはありません。',
    noMatchFiltersOn: gateway => `${gateway} にこれらのフィルタに一致するエージェントやグループチャットはありません`,
    noMatchQuery: query => `「${query}」に一致するエージェントやグループチャットはありません`,
    noMatchQueryOn: (query, gateway) => `${gateway} に「${query}」に一致するエージェントやグループチャットはありません`,
    rosterUnavailable: reason =>
      `名簿を取得できません: ${reason}。Hermes のバージョンが profiles.list に対応していない場合は、Hermes を更新して再起動してください。`,
    search: 'エージェントとグループチャットを検索',
    searchPlaceholder: 'エージェントとグループチャットを検索…',
    showHidden: '非表示のエージェントを表示',
    title: 'エージェント',
    waitingForGateway:
      'マシンへの接続を待っています…（リモートマシンへの接続には数秒かかることがあります。自動で再試行します）'
  },
  screen: {
    agentControls: 'エージェントが操作中',
    autoOpenMenu: 'エージェントが画面を使い始めたら Screen を開く',
    imageSwitchBody: (current, target) =>
      `サンドボックスはまだ ${current} で動作しており、デスクトップがありません。${target} に切り替えると、エージェントが次にターミナルを使うときにコンテナが再作成されます。/root と /workspace のファイルはこのマシンに残り、コンテナ内にインストールしたパッケージは必要に応じて再インストールされます。`,
    installHint:
      'Hermes を実行しているユーザーとしてマシン上で実行されます。sudo は Hermes 経由で一度だけ求められます。',
    managedNotIncludedBody:
      'Hermes Cloud では、メモリが 8 GB 以上のエージェントに Screen が含まれます。これより小さいエージェントにはインストールできません。',
    managedNotIncludedTitle: 'このエージェントには Screen が含まれていません',
    needsHttps:
      'エージェントの画面には安全な接続が必要です。表示するには、Webapp を HTTPS か 127.0.0.1 への SSH トンネル経由で開いてください。',
    noPackageManager: 'マシンに対応するパッケージマネージャー (apt, dnf, pacman) が見つかりません。',
    notInstalledBody: 'このエージェントに画面を与えるには、マシンに TigerVNC と Xfce コアが必要です。マシンで実行:',
    openNeedsUpdate: 'エージェントの画面を開くには Hermes Desktop を更新してください。',
    portalUnavailable: 'Screen を使うにはエージェントの Hermes を更新してください',
    portalWatching: 'ライブ · エージェントが操作中',
    stoppedBody: 'このエージェントのデスクトップを起動すると、動作を見守り、必要なときに操作を引き継げます。',
    unsupportedBody:
      'エージェント画面は Linux マシンで動作します。このエージェントはマシン自身のディスプレイを使います。',
    unsupportedTitle: 'このホストにはエージェント画面がありません'
  },
  sections: {
    deleted: (name, count) =>
      count === 0
        ? `「${name}」を削除しました`
        : `「${name}」を削除しました — ${count} 件のエージェントを未分類に移動しました`,
    emptyHint: 'ここにエージェントをドラッグ',
    headingTip: 'ここにエージェントをドロップ · ダブルクリックで名前を変更'
  },
  tools: {
    needsSetup: keys => `設定が必要（${keys}）— アプリ内で設定するには Hermes を再起動してください`,
    noTarget: '対象エージェントがありません'
  }
}

const zh: BotsWording = {
  avatar: {
    blobFromName: '斑点脸 — 根据智能体名称绘制',
    pickPet: '选择一只宠物作为此智能体的头像。',
    tabBot: '智能体',
    unlockFollowsName: '解锁 — 面孔再次跟随智能体名称'
  },
  bot: {
    attentionBlocked: '智能体已被阻止 — 请查看其最后一条消息',
    attentionProviderAuth: '请为此智能体重新登录',
    createFailed: '暂时无法创建智能体',
    createFirstHint: '打开智能体面板，点击“新建智能体”。',
    deleteTitle: '删除智能体？',
    descriptionHint: '留空则根据智能体的名称和描述生成。',
    editTitle: '编辑智能体',
    groupsLoadFailed: '无法加载智能体的群聊',
    helpPromptPlaceholder: '这个智能体应该帮你做什么？',
    hiddenToast: name => `已隐藏 ${name} — 点击智能体标题栏的眼睛按钮可查看隐藏的智能体`,
    loadFailed: '无法加载智能体',
    metadataLoadFailed: '无法加载智能体元数据',
    newChatWith: '与此智能体开新聊天',
    newTitle: '新建智能体',
    openAnotherChatUnsupported: '请更新 Hermes Desktop 以打开另一个智能体聊天。',
    openBotChat: '打开智能体聊天',
    openBotChatInNewTab: '在新标签页中打开智能体聊天',
    openGateways: '打开机器',
    openNeedsUpdateTitle: '这个智能体运行在较旧的 Hermes 上',
    openUnreachableTitle: 'Hermes 无法连接到运行这个智能体的电脑',
    remoteConnectionsUnsupported: '请更新 Hermes Desktop 以与其他连接上的智能体聊天。'
  },
  canonical: {
    createDuplicateMember: '在此机器上托管的群聊中的智能体需要使用不同的身份和名称标识；@all 和 @everyone 为保留名称。',
    createEnded: 'Hermes 拒绝了此群聊，它可能已在其他地方被更改或移除。请重试以新建一个。',
    createGroupSize: '在此机器上托管的群聊需要 2 到 6 个智能体。',
    createOtherConnection: '在此机器上托管的群聊只能加入此连接上的智能体。',
    createRefused: 'Hermes 无法创建此群聊。',
    createUncertain: 'Hermes 没有响应，此群聊可能已经创建。请重试以打开它，不会重复创建。',
    disbandMemberHomeUnknown:
      '群聊可能仍在运行，但其中一个智能体已不在运行群聊的电脑的配置中。请将该智能体重新加入 hosted_rooms.profiles，以便确认其已停止，然后再次尝试解散。',
    disbandServiceRequired:
      '必须在运行此群聊的电脑上启动 Hermes，才能停止智能体并撤销其访问权限。请在那台电脑上启动或重新连接 Hermes，然后重试解散。',
    disbandStopPending:
      '此群聊中的智能体尚未全部确认已停止。请在全部确认后再次尝试解散。如果群聊中为等待处理的回合提供了重试或丢弃操作，可以使用这些操作。',
    disbandUnconfirmed: '无法确认群聊是否已经结束。某个智能体的电脑可能处于休眠或离线状态。请在它恢复连接后重试。',
    discardWarning:
      '可能已经产生了更改，丢弃不会撤销这些更改。对于其他电脑上的智能体，丢弃会等待该电脑确认回合已停止或从未开始。',
    driverUnavailable: '此群聊暂不可用。请更新运行它的机器上的 Hermes，或重新连接该机器。',
    groupDriverStopped:
      '暂时无法在此打开群聊：此连接的机器上没有运行群聊。请在那台机器上重启或更新 Hermes，然后再试一次。不会丢失任何内容：之前的消息仍保留在原处。',
    groupsUnreachable:
      '无法连接到此连接的机器来打开这个群聊。请确认那台机器上的 Hermes 正在运行且你已连接，然后再试一次。不会丢失任何内容：之前的消息仍保留在原处。',
    legacyRoom:
      '这个群聊由 Desktop 自己运行，因此无法在此机器上继续。用相同的智能体开始一个新群组：关闭 Desktop 后它仍会继续工作。之前的消息不会复制到新群组。',
    legacyRoomBlocked:
      '这个群聊由 Desktop 自己运行，因此无法在此机器上继续，而且这些智能体目前无法组成在此机器上托管的群聊（原因见下方）。请开始一个新群聊并选择它的智能体。之前的消息不会复制到新群组。',
    legacyRoomElsewhere:
      '这个群聊无法在此继续：它包含其他电脑上的智能体，而在此机器上托管的群聊只能加入同一个连接上的智能体。请切换到这些智能体所在的连接并在那里开始群组，或者用此连接上的智能体开始一个新群聊。之前的消息不会复制到新群组。',
    otherConnections: '其他连接上的智能体：{connections}',
    peerJoinChanged:
      '{name} 无法加入此群聊：自群聊创建以来，它所在电脑上的智能体设置已更改。要加入它，请开始一个新群聊。',
    peerStartWithoutConfirm:
      '将用其他智能体创建新群聊。只有它们全部加入，且原群聊所有成员确认停止并撤销访问权限后，原群聊才会结束。任一步骤失败都会保留两个群聊。之前的消息不会复制。 创建后再次使用此操作会创建另一个群聊。',
    peerStartWithoutRetry: '请在新群聊中对这些智能体使用“重试”。',
    peerTargetSetup: '{connection} 目前还无法让智能体加入其他电脑上的群聊。{reason}',
    readTimedOut: 'Hermes 未及时响应。它可能正在等待某个智能体的电脑响应。请选择“刷新”重试。',
    reasonAgentBlocked: '智能体已被阻止运行',
    reasonGatewayRestarted: '智能体所在机器上的 Hermes 已重启',
    reasonMemberUnavailable: '智能体不可用',
    reasonPrepareFailed: '无法启动智能体',
    reasonRuntimeOffline: '智能体已离线',
    reasonSlotWait: '运行中的智能体过多',
    reasonTargetBusy: '智能体正忙',
    reasonTargetScope: '智能体无法访问其凭据',
    sendRefused: 'Hermes 拒绝了这条消息。请编辑后重新发送。',
    turnNoAnswer: '智能体未及时响应'
  },
  cron: {
    botChatTarget: bot => `${bot} 的聊天（智能体会回应）`,
    detailDescription: '此计划的内容和下次运行时间。',
    filterHint:
      '此智能体有计划，但没有一个带有它的标记。将计划命名为“[bot:<名称>] …”即可显示在这里，也可以在下方的计划中查看。',
    instructionNul: '计划指令不能包含 NUL (U+0000)。',
    legacyUnsafe: '为安全起见已暂停：请删除并重新创建此旧计划，然后再运行。',
    nameNul: '计划名称不能包含 NUL (U+0000)。',
    needsRosterFirst: '这个智能体需要先出现在名册中。',
    staleNotice: '无法刷新计划。显示的是上一次获取的列表。',
    untitled: '未命名计划'
  },
  crossGateway: {
    gatewayPeer: (owner, local = false) =>
      `在 ${local ? '此设备' : owner} 上运行，即使此桌面端已关闭。其他电脑上的智能体仅接收文本。`,
    ownerCannotSeatPeers: place =>
      `${place} 上的 Hermes 暂时无法让其他电脑上的智能体加入。请在那里更新 Hermes 并运行其网关（hermes gateway run），然后重试。`,
    peerGatewayChanged: place => `${place} 现在运行的 Hermes 安装与此智能体来源的不同。请重新连接，然后重试。`,
    peerRouteMissing: '所选智能体的电脑未连接。请重新连接，然后重试。',
    pickBots: max => `选择 2–${max} 个智能体。`,
    selectionElsewhere: connection =>
      `这些智能体都在 ${connection} 上。请添加一个此连接上的智能体，或切换到 ${connection} 来创建只包含其智能体的群组。`
  },
  editor: {
    cloneFrom: '从智能体克隆',
    cloneFromOn: target => `从智能体克隆（位于 ${target}）`,
    createBot: '创建智能体',
    created: name => `已创建智能体“${name}”`,
    createdOn: (name, target) => `已在 ${target} 上创建智能体“${name}”`,
    freshProfile: '新智能体（内置技能）',
    fullConfigHint: '完整配置需要更新版本的 Hermes（请更新 Hermes，然后重启 Hermes）。',
    gatewayDefault: '机器默认值',
    inheritLaunch: '继承（启动智能体）',
    inheritedModel: '继承启动时的智能体',
    mcpHint:
      '已配置的服务器从主智能体复制；目录条目来自内置 MCP 菜单。需要 API 密钥的条目先进行设置（凭据遵循共享密钥设置）。',
    nameFirstHint: '请先为智能体命名，打开此标签页时将创建草稿智能体（取消时会丢弃）。',
    newerGateway: '能力目录需要更新版本的 Hermes（请更新 Hermes，然后重启 Hermes）。',
    noImageModel: '没有可用的图像模型。如果刚启用模型或更新了 Hermes，请重启 Hermes：Ctrl+K →“重启 Hermes”。',
    remoteHint: target => `智能体将在 ${target} 上创建，并作为连接智能体显示在名册中。聊天将路由到该机器。`,
    shareKeys: '与主智能体共享密钥和账户',
    shareKeysOn: target => `与 ${target} 上的 default 智能体共享密钥和账户`
  },
  group: {
    activityBot: '某个智能体',
    activitySlotWait: '无法开始 — 运行中的智能体过多',
    allHeldStatus: count => `全部 ${count} 个智能体已暂停`,
    attachHint: '附加文件 — 每个回应的智能体都能看到',
    composerPlaceholder: '说点什么 — 这个群聊里的每个智能体都会看到你的消息。',
    createBotFirst: '还没有智能体，请先创建一个。',
    noSeatableBots: '这里还没有可加入群聊的智能体：',
    cannotJoinHere: '这些智能体无法在此加入群聊：',
    nonePicked: '尚未选择智能体',
    unseatableHere: '这台机器上的 Hermes 不会让它加入群聊——它不在此处提供，或正在停用',
    createDesc: max =>
      `选择 2–${max} 个智能体。本机成员关系通过各智能体同步；其他电脑上智能体的成员关系仅保存在此群聊中。`,
    created: (group, count) => `已创建“${group}”，包含 ${count} 个智能体`,
    deleteDesc: group => `这将把智能体移出“${group}”并清除群聊的共享历史记录。智能体及其单独聊天会保留。`,
    disbandDescSuffix: count =>
      ` 的分组将从 ${count} 个智能体中移除，并清空共享群聊日志。智能体本身及其各群组会话都会保留。`,
    dropToRoom: '拖放以附加 — 每个回应的智能体都能看到',
    everyoneMeta: '群聊里的所有智能体',
    holdReleaseHint: '提及已暂停的智能体，或发送 @all resume 以恢复它们。',
    manageDesc: '一个智能体可以加入多个群聊。成员关系会同步到每台设备。',
    manageMembersDesc: (max, group) => `为“${group}”选择 2–${max} 个智能体。群聊、历史记录和成员会话均保持不变。`,
    memberCount: count => `${count} 个智能体`,
    memberSizeError: max => `群聊需要 2 到 ${max} 个智能体`,
    mentionBot: '此群聊中的智能体',
    needsYourInput: '此群聊中有智能体需要你输入',
    noBots: '此群聊中没有智能体',
    noBotsMatch: query => `没有匹配“${query}”的智能体`,
    noMembersToSend: group => `${group} 没有可发送的成员——请添加智能体，如果成员仍在加载，请重新打开该群聊。`,
    searchToAdd: '搜索要添加的智能体',
    searchToAddPlaceholder: '搜索要添加的智能体…',
    slashCommandsUnsupported: '群聊不支持斜杠命令。请打开单个智能体的聊天来使用。'
  },
  roster: {
    allGateways: '所有机器',
    allHidden: '所有智能体都已隐藏',
    botsAndGroups: '智能体和群聊',
    botsOnly: '仅智能体',
    currentGateway: '当前机器',
    emptyDesc: '创建你的第一个智能体。',
    emptyTitle: '还没有智能体',
    gatewayRemoved: '机器已移除',
    newBotOrGroup: '新建智能体或群聊',
    noHiddenMatch: '没有已隐藏的智能体匹配这些筛选条件。',
    noMatchFilters: '没有智能体或群聊匹配这些筛选条件。',
    noMatchFiltersOn: gateway => `${gateway} 上没有智能体或群聊匹配这些筛选条件`,
    noMatchQuery: query => `没有智能体或群聊匹配“${query}”`,
    noMatchQueryOn: (query, gateway) => `${gateway} 上没有智能体或群聊匹配“${query}”`,
    rosterUnavailable: reason => `无法获取名单：${reason}。如果 Hermes 版本不支持 profiles.list，请更新并重启 Hermes。`,
    search: '搜索智能体和群聊',
    searchPlaceholder: '搜索智能体和群聊…',
    showHidden: '显示已隐藏的智能体',
    title: '智能体',
    waitingForGateway: '正在等待连接到机器…（远程机器可能需要几秒；会自动重试）'
  },
  screen: {
    agentControls: '智能体正在控制',
    autoOpenMenu: '智能体使用屏幕时自动打开',
    imageSwitchBody: (current, target) =>
      `沙箱仍在运行 ${current}，其中没有桌面。切换到 ${target} 后，智能体下次使用终端时会重建容器：/root 和 /workspace 中的文件保留在本机，容器内安装的软件包会按需重新安装。`,
    installHint: '在机器上以运行 Hermes 的用户身份执行；sudo 只会通过 Hermes 询问一次。',
    managedNotIncludedBody: 'Hermes Cloud 为内存 8 GB 及以上的智能体提供屏幕，较小的智能体无法安装。',
    managedNotIncludedTitle: '此智能体不包含屏幕',
    needsHttps: '智能体屏幕需要安全连接。请通过 HTTPS 或到 127.0.0.1 的 SSH 隧道打开 Webapp 后查看。',
    noPackageManager: '机器上未找到受支持的包管理器（apt、dnf、pacman）。',
    notInstalledBody: '机器需要 TigerVNC 和 Xfce 核心组件才能为此智能体提供屏幕。在机器上运行:',
    openNeedsUpdate: '更新 Hermes Desktop 以打开智能体屏幕。',
    portalUnavailable: '更新智能体的 Hermes 以使用屏幕',
    portalWatching: '直播 · 智能体控制中',
    stoppedBody: '启动此智能体的桌面，观看它的操作，并在需要时接管。',
    unsupportedBody: '智能体屏幕在 Linux 机器上运行。此智能体使用机器自身的显示器。',
    unsupportedTitle: '此主机没有智能体屏幕'
  },
  sections: {
    deleted: (name, count) => (count === 0 ? `已删除“${name}”` : `已删除“${name}” — ${count} 个智能体已移至未分类`),
    emptyHint: '将智能体拖到此处',
    headingTip: '将智能体拖放到此处 · 双击重命名'
  },
  tools: {
    needsSetup: keys => `需要设置（${keys}）— 重启 Hermes 以启用应用内设置`,
    noTarget: '没有目标智能体'
  }
}

const zhHant: BotsWording = {
  avatar: {
    blobFromName: '斑點臉 — 依智慧體名稱繪製',
    pickPet: '選擇一隻寵物作為此智慧體的頭像。',
    tabBot: '智慧體',
    unlockFollowsName: '解鎖 — 面孔再次跟隨智慧體名稱'
  },
  bot: {
    attentionBlocked: '智慧體已被封鎖 — 請查看其最後一則訊息',
    attentionProviderAuth: '請為此智慧體重新登入',
    createFailed: '暫時無法建立智慧體',
    createFirstHint: '開啟智慧體面板，點「新增智慧體」。',
    deleteTitle: '刪除智慧體？',
    descriptionHint: '留空則依智慧體的名稱和描述產生。',
    editTitle: '編輯智慧體',
    groupsLoadFailed: '無法載入智慧體的群組',
    helpPromptPlaceholder: '這個智慧體應該幫你做什麼？',
    hiddenToast: name => `已隱藏 ${name} — 點擊智慧體標題列的眼睛按鈕可查看隱藏的智慧體`,
    loadFailed: '無法載入智慧體',
    metadataLoadFailed: '無法載入智慧體中繼資料',
    newChatWith: '與此智慧體開新聊天',
    newTitle: '新增智慧體',
    openAnotherChatUnsupported: '請更新 Hermes Desktop 以開啟另一個智慧體聊天。',
    openBotChat: '開啟智慧體聊天',
    openBotChatInNewTab: '在新分頁中開啟智慧體聊天',
    openGateways: '開啟機器',
    openNeedsUpdateTitle: '這個智慧體運行在較舊的 Hermes 上',
    openUnreachableTitle: 'Hermes 無法連線到運行這個智慧體的電腦',
    remoteConnectionsUnsupported: '請更新 Hermes Desktop 以與其他連線上的智慧體聊天。'
  },
  canonical: {
    createDuplicateMember: '在此機器上託管的群組聊天中的智慧體需要使用不同的身分和代稱；@all 與 @everyone 為保留名稱。',
    createEnded: 'Hermes 拒絕了此群組聊天，它可能已在其他地方被變更或移除。請再試一次以建立新的群組聊天。',
    createGroupSize: '在此機器上託管的群組聊天需要 2 到 6 個智慧體。',
    createOtherConnection: '在此機器上託管的群組聊天只能加入此連線上的智慧體。',
    createRefused: 'Hermes 無法建立此群組聊天。',
    createUncertain: 'Hermes 沒有回應，此群組聊天可能已經建立。請再試一次以開啟它，不會重複建立。',
    disbandMemberHomeUnknown:
      '群組聊天可能仍在執行，但其中一個智慧體已不在執行群組聊天的電腦的設定中。請將該智慧體重新加入 hosted_rooms.profiles，以便確認其已停止，然後再次嘗試解散。',
    disbandServiceRequired:
      '必須在執行此群組聊天的電腦上啟動 Hermes，才能停止智慧體並撤銷其存取權。請在那台電腦上啟動或重新連線至 Hermes，再重試解散。',
    disbandStopPending:
      '此群組聊天中的智慧體尚未全部確認已停止。請在全部確認後再次嘗試解散。如果群組聊天中為等待處理的回合提供了重試或捨棄操作，可以使用這些操作。',
    disbandUnconfirmed: '無法確認群組聊天是否已經結束。某個智慧體的電腦可能處於休眠或離線狀態。請在它恢復連線後重試。',
    discardWarning:
      '可能已經產生了變更，捨棄不會復原這些變更。對於其他電腦上的智慧體，捨棄會等待該電腦確認回合已停止或從未開始。',
    driverUnavailable: '此群組聊天目前無法使用。請更新執行它的機器上的 Hermes，或重新連線至該機器。',
    groupDriverStopped:
      '暫時無法在此開啟群組聊天：此連線的機器上沒有執行群組聊天。請在那台機器上重新啟動或更新 Hermes，然後再試一次。不會遺失任何內容：先前的訊息仍保留在原處。',
    groupsUnreachable:
      '無法連上此連線的機器來開啟這個群組聊天。請確認那台機器上的 Hermes 正在執行且你已連線，然後再試一次。不會遺失任何內容：先前的訊息仍保留在原處。',
    legacyRoom:
      '這個群組聊天由 Desktop 自行執行，因此無法在此機器上繼續。用相同的智慧體開始新群組：關閉 Desktop 後它仍會繼續運作。先前的訊息不會複製到新群組。',
    legacyRoomBlocked:
      '這個群組聊天由 Desktop 自行執行，因此無法在此機器上繼續，而且這些智慧體目前無法組成在此機器上託管的群組聊天（原因見下方）。請開始新的群組聊天並選擇它的智慧體。先前的訊息不會複製到新群組。',
    legacyRoomElsewhere:
      '這個群組聊天無法在此繼續：它包含其他電腦上的智慧體，而在此機器上託管的群組聊天只能加入同一個連線上的智慧體。請切換到這些智慧體所在的連線並在那裡開始群組，或用此連線上的智慧體開始新的群組聊天。先前的訊息不會複製到新群組。',
    otherConnections: '其他連線上的智慧體：{connections}',
    peerJoinChanged:
      '{name} 無法加入此群組聊天：自群組聊天建立以來，它所在電腦上的智慧體設定已變更。若要加入它，請開始新的群組聊天。',
    peerStartWithoutConfirm:
      '將用其他智慧體建立新的群組聊天。只有它們全部加入，且原有群組聊天的所有成員確認停止並撤銷存取權後，原有聊天才會結束。任一步驟失敗都會保留兩個群組聊天。先前的訊息不會複製。 建立後再次使用此操作會建立另一個群組聊天。',
    peerStartWithoutRetry: '請在新的群組聊天中對這些智慧體使用「重試」。',
    peerTargetSetup: '{connection} 目前還無法讓智慧體加入其他電腦上的群組聊天。{reason}',
    readTimedOut: 'Hermes 未及時回應。它可能正在等待某個智慧體的電腦回應。請選擇「重新整理」重試。',
    reasonAgentBlocked: '智慧體已被阻止執行',
    reasonGatewayRestarted: '智慧體所在機器上的 Hermes 已重新啟動',
    reasonMemberUnavailable: '智慧體無法使用',
    reasonPrepareFailed: '無法啟動智慧體',
    reasonRuntimeOffline: '智慧體已離線',
    reasonSlotWait: '執行中的智慧體過多',
    reasonTargetBusy: '智慧體正忙',
    reasonTargetScope: '智慧體無法存取其憑證',
    sendRefused: 'Hermes 拒絕了這則訊息。請編輯後再傳送。',
    turnNoAnswer: '智慧體未及時回應'
  },
  cron: {
    botChatTarget: bot => `${bot} 的聊天（智慧體會回應）`,
    detailDescription: '此排程的內容和下次執行時間。',
    filterHint:
      '此智慧體有排程，但沒有任何一個帶有它的標記。將排程命名為「[bot:<名稱>] …」即可顯示在這裡，也可以在下方的排程中查看。',
    instructionNul: '排程指令不能包含 NUL (U+0000)。',
    legacyUnsafe: '基於安全考量已暫停：請刪除並重新建立此舊排程，然後再執行。',
    nameNul: '排程名稱不能包含 NUL (U+0000)。',
    needsRosterFirst: '這個智慧體需要先出現在名冊中。',
    staleNotice: '無法重新整理排程。顯示的是上一次取得的清單。',
    untitled: '未命名排程'
  },
  crossGateway: {
    gatewayPeer: (owner, local = false) =>
      `在 ${local ? '此裝置' : owner} 上執行，即使此桌面端已關閉。其他電腦上的智慧體僅接收文字。`,
    ownerCannotSeatPeers: place =>
      `${place} 上的 Hermes 暫時無法讓其他電腦上的智慧體加入。請在那裡更新 Hermes 並執行其閘道（hermes gateway run），然後再試一次。`,
    peerGatewayChanged: place => `${place} 現在執行的 Hermes 安裝與此智慧體來源的不同。請重新連線，然後再試一次。`,
    peerRouteMissing: '所選智慧體的電腦未連線。請重新連線，然後再試一次。',
    pickBots: max => `選擇 2–${max} 個智慧體。`,
    selectionElsewhere: connection =>
      `這些智慧體都在 ${connection} 上。請加入一個此連線上的智慧體，或切換到 ${connection} 來建立只包含其智慧體的群組。`
  },
  editor: {
    cloneFrom: '從智慧體複製',
    cloneFromOn: target => `從智慧體複製（位於 ${target}）`,
    createBot: '建立智慧體',
    created: name => `已建立智慧體「${name}」`,
    createdOn: (name, target) => `已在 ${target} 上建立智慧體「${name}」`,
    editSoul: 'SOUL.md（人格 + 智慧體訊息協定）',
    freshProfile: '新智慧體（內建技能）',
    fullConfigHint: '完整設定需要更新版本的 Hermes（請更新 Hermes，然後重新啟動 Hermes）。',
    gatewayDefault: '機器預設值',
    inheritLaunch: '繼承（啟動智慧體）',
    inheritedModel: '繼承啟動時的智慧體',
    mcpHint:
      '已設定的伺服器從主要智慧體複製；目錄項目來自內建 MCP 選單。需要 API 金鑰的項目先進行設定（憑證遵循共用金鑰設定）。',
    nameFirstHint: '請先為智慧體命名，開啟此分頁時將建立草稿智慧體（取消時會捨棄）。',
    nameTaken: name => `名為「${name}」的智慧體已存在。`,
    nameTakenOn: (name, target) => `${target} 上已存在名為「${name}」的智慧體。`,
    newDescription: '擁有獨立記憶、技能和聊天的具名隊友，可以與你的其他智慧體互傳訊息。',
    newerGateway: '功能目錄需要更新版本的 Hermes（請更新 Hermes，然後重新啟動 Hermes）。',
    noImageModel: '沒有可用的影像模型。如果剛啟用模型或更新了 Hermes，請重新啟動 Hermes：Ctrl+K →「重新啟動 Hermes」。',
    remoteHint: target => `智慧體將在 ${target} 上建立，並作為連線智慧體顯示在名冊中。聊天將路由到該機器。`,
    shareKeys: '與主要智慧體共用金鑰和帳戶',
    shareKeysOn: target => `與 ${target} 上的 default 智慧體共用金鑰和帳戶`
  },
  group: {
    activityBot: '某個智慧體',
    activitySlotWait: '無法開始 — 執行中的智慧體過多',
    allHeldStatus: count => `全部 ${count} 個智慧體已暫停`,
    attachHint: '附加檔案 — 每個回應的智慧體都能看到',
    composerPlaceholder: '說點什麼 — 這個群組聊天裡的每個智慧體都會看到你的訊息。',
    createBotFirst: '還沒有智慧體，請先建立一個。',
    noSeatableBots: '這裡還沒有可加入群組聊天的智慧體：',
    cannotJoinHere: '這些智慧體無法在此加入群組聊天：',
    nonePicked: '尚未選擇智慧體',
    unseatableHere: '這台機器上的 Hermes 不會讓它加入群組聊天——它不在此處提供，或正在停用',
    createDesc: max =>
      `選擇 2–${max} 個智慧體。本機成員關係透過各智慧體同步；其他電腦上智慧體的成員關係僅保存在此群組聊天中。`,
    created: (group, count) => `已建立「${group}」，包含 ${count} 個智慧體`,
    deleteDesc: group => `這會將智慧體移出「${group}」並清除群組聊天的共用歷史記錄。智慧體及其個別聊天會保留。`,
    disbandDescSuffix: count =>
      ` 的分組將從 ${count} 個智慧體中移除，並清空共享群組聊天日誌。智慧體本身及其各群組工作階段都會保留。`,
    dropToRoom: '拖放以附加 — 每個回應的智慧體都能看到',
    everyoneMeta: '群組聊天裡的所有智慧體',
    holdReleaseHint: '提及已暫停的智慧體，或傳送 @all resume 以恢復它們。',
    manageDesc: '一個智慧體可以加入多個群組聊天。成員關係會同步到每台裝置。',
    manageMembersDesc: (max, group) =>
      `為「${group}」選擇 2–${max} 個智慧體。群組聊天、歷史記錄和成員工作階段均保持不變。`,
    memberCount: count => `${count} 個智慧體`,
    memberSizeError: max => `群組聊天需要 2 到 ${max} 個智慧體`,
    mentionBot: '此群組聊天中的智慧體',
    needsYourInput: '此群組聊天中有智慧體需要您的輸入',
    noBots: '此群組聊天中沒有智慧體',
    noBotsMatch: query => `沒有符合「${query}」的智慧體`,
    noMembersToSend: group => `${group} 沒有可傳送的成員——請新增智慧體，若成員仍在載入中，請重新開啟該群組聊天。`,
    searchToAdd: '搜尋要加入的智慧體',
    searchToAddPlaceholder: '搜尋要加入的智慧體…',
    sendRouteRequired: '向此群組聊天傳送訊息需要其連線、智慧體和群組聊天識別碼',
    slashCommandsUnsupported: '群組聊天不支援斜線命令。請開啟個別智慧體的聊天來使用。'
  },
  roster: {
    allGateways: '所有機器',
    allHidden: '所有智慧體都已隱藏',
    botsAndGroups: '智慧體和群組聊天',
    botsOnly: '僅智慧體',
    currentGateway: '目前的機器',
    emptyDesc: '建立你的第一個智慧體。',
    emptyTitle: '還沒有智慧體',
    gatewayRemoved: '機器已移除',
    newBotOrGroup: '新增智慧體或群組聊天',
    noHiddenMatch: '沒有已隱藏的智慧體符合這些篩選條件。',
    noMatchFilters: '沒有智慧體或群組聊天符合這些篩選條件。',
    noMatchFiltersOn: gateway => `${gateway} 上沒有智慧體或群組聊天符合這些篩選條件`,
    noMatchQuery: query => `沒有智慧體或群組聊天符合「${query}」`,
    noMatchQueryOn: (query, gateway) => `${gateway} 上沒有智慧體或群組聊天符合「${query}」`,
    rosterUnavailable: reason =>
      `無法取得名單：${reason}。如果 Hermes 版本不支援 profiles.list，請更新並重新啟動 Hermes。`,
    search: '搜尋智慧體和群組聊天',
    searchPlaceholder: '搜尋智慧體和群組聊天…',
    showHidden: '顯示已隱藏的智慧體',
    title: '智慧體',
    waitingForGateway: '正在等待連線至機器…（遠端機器可能需要幾秒；會自動重試）'
  },
  screen: {
    agentControls: '智慧體正在控制',
    autoOpenMenu: '智慧體使用螢幕時自動開啟',
    imageSwitchBody: (current, target) =>
      `沙箱仍在執行 ${current}，其中沒有桌面。切換到 ${target} 後，智慧體下次使用終端時會重建容器：/root 和 /workspace 中的檔案保留在本機，容器內安裝的套件會按需重新安裝。`,
    installHint: '在機器上以執行 Hermes 的使用者身分執行；sudo 只會透過 Hermes 詢問一次。',
    managedNotIncludedBody: 'Hermes Cloud 為記憶體 8 GB 以上的智慧體提供螢幕，較小的智慧體無法安裝。',
    managedNotIncludedTitle: '此智慧體不包含螢幕',
    needsHttps: '智慧體螢幕需要安全連線。請透過 HTTPS 或到 127.0.0.1 的 SSH 通道開啟 Webapp 後檢視。',
    noPackageManager: '機器上找不到受支援的套件管理器（apt、dnf、pacman）。',
    notInstalledBody: '機器需要 TigerVNC 與 Xfce 核心元件才能為此智慧體提供螢幕。在機器上執行:',
    openNeedsUpdate: '更新 Hermes Desktop 以開啟智慧體螢幕。',
    portalUnavailable: '更新智慧體的 Hermes 以使用螢幕',
    portalWatching: '直播 · 智慧體控制中',
    stoppedBody: '啟動此智慧體的桌面，觀看它的操作，並在需要時接手。',
    unsupportedBody: '智慧體螢幕在 Linux 機器上執行。此智慧體使用機器自身的顯示器。',
    unsupportedTitle: '此主機沒有智慧體螢幕'
  },
  sections: {
    deleted: (name, count) => (count === 0 ? `已刪除「${name}」` : `已刪除「${name}」— ${count} 個智慧體已移至未分類`),
    emptyHint: '將智慧體拖到此處',
    headingTip: '將智慧體拖放到此處 · 雙擊重新命名'
  },
  tools: {
    needsSetup: keys => `需要設定（${keys}）— 重新啟動 Hermes 以啟用應用程式內設定`,
    noTarget: '沒有目標智慧體'
  }
}

const ar: BotsWording = {
  bot: {
    attentionBlocked: 'الوكيل محظور — راجع آخر رسالة له',
    groupsLoadFailed: 'تعذّر تحميل مجموعات الوكيل',
    hiddenToast: name => `تم إخفاء ${name} — استخدم زر العين أعلى قائمة الوكلاء لإظهاره`,
    loadFailed: 'تعذّر تحميل الوكيل',
    metadataLoadFailed: 'تعذّر تحميل بيانات الوكيل',
    newChatWith: 'محادثة جديدة مع هذا الوكيل',
    newTitle: 'وكيل جديد',
    openBotChat: 'فتح محادثة الوكيل',
    openBotChatInNewTab: 'فتح محادثة الوكيل في علامة تبويب جديدة'
  },
  canonical: {
    composerPlaceholder: 'راسل المجموعة — اذكر @وكيلًا لتكليفه',
    createDuplicateMember:
      'تحتاج محادثة المجموعة المستضافة على هذا الجهاز إلى وكلاء مختلفين بمعرّفات فريدة؛ ‎@all و‎@everyone محجوزان.',
    createEnded:
      'رفض Hermes محادثة المجموعة هذه، فربما تم تغييرها أو إزالتها في مكان آخر. أعد المحاولة لبدء محادثة جديدة.',
    createGroupSize: 'تحتاج محادثة المجموعة المستضافة على هذا الجهاز إلى ما بين اثنين وستة وكلاء.',
    createOtherConnection:
      'لا يمكن لمحادثة المجموعة المستضافة على هذا الجهاز أن تضم إلا الوكلاء الموجودة على هذا الاتصال.',
    createRefused: 'تعذّر على Hermes إنشاء محادثة المجموعة هذه.',
    createUncertain:
      'لم يستجب Hermes، لذا ربما تكون محادثة المجموعة هذه قد أُنشئت بالفعل. أعد المحاولة لفتحها، ولن تُنشأ مرتين.',
    disbandMemberHomeUnknown:
      'قد يستمر العمل في محادثة المجموعة، لكن أحد الوكلاء لم يعد مضافًا إلى إعدادات الجهاز الذي يشغّل محادثة المجموعة. أعد إضافته إلى hosted_rooms.profiles حتى يمكن تأكيد توقفه، ثم حاول حل محادثة المجموعة مجددًا.',
    disbandServiceRequired:
      'يجب تشغيل Hermes على الجهاز الذي يشغّل محادثة المجموعة هذه لإيقاف وكلائها وإلغاء وصولها. شغّل Hermes هناك أو أعد الاتصال به، ثم حاول حل المحادثة مجددًا.',
    disbandStopPending:
      'لم تؤكد جميع الوكلاء في محادثة المجموعة هذه أنها توقفت. حاول حل محادثة المجموعة مجددًا بعد تأكيدها جميعًا. للدور المنتظر، استخدم إعادة المحاولة أو التجاهل إذا كان الخيار متاحًا في محادثة المجموعة.',
    disbandUnconfirmed:
      'لم نتمكن من تأكيد انتهاء محادثة المجموعة. قد يكون جهاز أحد الوكلاء في وضع السكون أو غير متصل. أعد المحاولة بعد عودته للاتصال.',
    discardWarning:
      'ربما حدثت تغييرات بالفعل. لن يؤدي تجاهل الدور إلى التراجع عنها. بالنسبة لوكيل على جهاز آخر، ينتظر التجاهل تأكيد ذلك الجهاز أن الدور توقف أو لم يبدأ أصلًا.',
    driverUnavailable:
      'محادثة المجموعة هذه غير متاحة. حدّث Hermes على الجهاز الذي يستضيفها أو أعد الاتصال بذلك الجهاز.',
    groupDriverStopped:
      'لا يمكن فتح محادثات المجموعات هنا الآن: محادثات المجموعات لا تعمل على الجهاز المرتبط بهذا الاتصال. أعد تشغيل Hermes على ذلك الجهاز أو حدّثه، ثم حاول مرة أخرى. لن يضيع شيء: تبقى الرسائل السابقة في مكانها.',
    groupsUnreachable:
      'تعذّر الوصول إلى الجهاز المرتبط بهذا الاتصال لفتح محادثة المجموعة هذه. تأكد من أن Hermes يعمل هناك وأنك متصل، ثم حاول مرة أخرى. لن يضيع شيء: تبقى الرسائل السابقة في مكانها.',
    legacyRoom:
      'شغّل Desktop محادثة المجموعة هذه بنفسه، لذا لا يمكنها المتابعة على هذا الجهاز. ابدأ مجموعة جديدة بالوكلاء نفسها: ستستمر في العمل عند إغلاق Desktop. لن تُنسخ الرسائل السابقة إليها.',
    legacyRoomBlocked:
      'شغّل Desktop محادثة المجموعة هذه بنفسه، لذا لا يمكنها المتابعة على هذا الجهاز، ولا تستطيع هذه الوكلاء بدء محادثة مجموعة مستضافة على هذا الجهاز كما هي (السبب أدناه). ابدأ محادثة مجموعة جديدة واختر وكلائها. لن تُنسخ الرسائل السابقة إليها.',
    legacyRoomElsewhere:
      'لا يمكن متابعة محادثة المجموعة هذه هنا: فيها وكلاء تعمل على أجهزة كمبيوتر أخرى، ولا تضم محادثة المجموعة المستضافة على هذا الجهاز إلا وكلاء من اتصال واحد. انتقل إلى الاتصال الذي تعمل عليه تلك الوكلاء وابدأ مجموعة هناك، أو ابدأ محادثة مجموعة جديدة بوكلاء من هذا الاتصال. لن تُنسخ الرسائل السابقة إلى مجموعة جديدة.',
    otherConnections: 'وكلاء على اتصالات أخرى: {connections}',
    peerJoinChanged:
      'لا يمكن لـ {name} الانضمام إلى محادثة المجموعة هذه: تغيّرت إعدادات الوكيل على جهازه منذ إنشاء محادثة المجموعة. ابدأ محادثة مجموعة جديدة لتضمينه.',
    peerStartWithoutConfirm:
      'تبدأ دردشة جماعية جديدة بالوكلاء الأخرى. لا تنتهي الأصلية إلا بعد انضمامها جميعًا وتأكيد إيقاف كل عضو أصلي وإلغاء وصوله. إذا فشلت أي خطوة، تبقى الدردشتان. لن تُنسخ الرسائل السابقة. استخدام هذا الإجراء مجددًا بعد الإنشاء ينشئ دردشة جماعية أخرى.',
    peerStartWithoutRetry: 'استخدم إعادة المحاولة لهذه الوكلاء في محادثة المجموعة الجديدة.',
    peerTargetSetup: 'لا يستطيع {connection} بعد استضافة وكيل لمحادثة مجموعة على جهاز آخر. {reason}',
    readTimedOut: 'لم يستجب Hermes في الوقت المحدد. ربما ينتظر استجابة جهاز أحد الوكلاء. اختر «تحديث» لإعادة المحاولة.',
    reasonAgentBlocked: 'الوكيل محظور',
    reasonGatewayRestarted: 'أُعيد تشغيل Hermes على جهاز الوكيل',
    reasonMemberUnavailable: 'الوكيل غير متاح',
    reasonPrepareFailed: 'تعذّر بدء الوكلاء',
    reasonRuntimeOffline: 'الوكيل غير متصل',
    reasonSlotWait: 'هناك وكلاء كثيرة قيد التشغيل',
    reasonTargetBusy: 'الوكيل مشغول',
    reasonTargetScope: 'تعذّر على الوكيل الوصول إلى بيانات اعتماده',
    sendRefused: 'رفض Hermes هذه الرسالة. عدّلها ثم أرسلها مرة أخرى.',
    turnNoAnswer: 'لم يستجب الوكيل في الوقت المناسب'
  },
  crossGateway: {
    gatewayPeer: (owner, local = false) =>
      `تعمل على ${local ? 'هذا الجهاز' : owner} حتى عند إغلاق سطح المكتب هذا. تقبل الوكلاء على الأجهزة الأخرى النص فقط.`,
    ownerCannotSeatPeers: place =>
      `لا يستطيع Hermes على ${place} بعدُ ضمّ وكلاء من أجهزة أخرى. حدّث Hermes هناك وشغّل بوابته (hermes gateway run)، ثم أعد المحاولة.`,
    peerGatewayChanged: place =>
      `يعمل ${place} الآن بتثبيت Hermes مختلف عن الذي جاء منه هذا الوكيل. أعد الاتصال به، ثم أعد المحاولة.`,
    peerRouteMissing: 'جهاز أحد الوكلاء المحددة غير متصل. أعد الاتصال به، ثم أعد المحاولة.',
    selectionElsewhere: connection =>
      `كل هذه الوكلاء موجودة على ${connection}. أضف وكيلًا من هذا الاتصال، أو انتقل إلى ${connection} لبدء مجموعة من وكلاءه.`
  },
  group: {
    activityBot: 'وكيل',
    activitySlotWait: 'تعذّر عليه البدء — هناك وكلاء كثيرة قيد التشغيل',
    allHeldStatus: count =>
      count === 0
        ? 'لا توجد وكلاء لإيقافها مؤقتًا'
        : count === 1
          ? 'الوكيل متوقف مؤقتًا'
          : count === 2
            ? 'الوكيلان متوقفان مؤقتًا'
            : `جميع الوكلاء متوقفة مؤقتًا (${count})`,
    attachHint: 'إرفاق ملفات — يراها كل وكيل يرد',
    composerPlaceholder: 'اكتب رسالة — سيراها كل وكيل في محادثة المجموعة هذه.',
    createBotFirst: 'لا توجد وكلاء بعد — أنشئ واحدًا أولًا.',
    noSeatableBots: 'لا يمكن لأي من وكلائك الانضمام إلى محادثة مجموعة هنا بعد:',
    cannotJoinHere: 'لا يمكن لهؤلاء الوكلاء الانضمام إلى محادثة مجموعة هنا:',
    nonePicked: 'لم يتم اختيار أي وكيل بعد',
    unseatableHere: 'لا يضمّه Hermes على هذا الجهاز إلى محادثات المجموعات — فهو غير مُقدَّم هنا أو قيد الإيقاف',
    createDesc: max =>
      `اختر ما بين وكيلين و${arabicAgents(max, 'وكيلين')}. تُزامَن العضويات المحلية عبر كل وكيل؛ أما عضوية الوكلاء على أجهزة أخرى فتُحفظ في محادثة المجموعة هذه فقط.`,
    created: (group, count) => `أُنشئت «${group}». عدد الوكلاء: ${count}`,
    deleteDesc: group =>
      `يؤدي هذا إلى إزالة «${group}» من وكلائها ومسح السجل المشترك لمحادثة المجموعة. يبقى الوكلاء ومحادثاتهم الفردية محفوظين.`,
    disbandDescSuffix: count =>
      ` من أعضائها (${arabicAgents(count, 'وكيلين')}) ومسح السجل المشترك لمحادثة المجموعة. ${count === 1 ? 'يبقى الوكيل نفسه وجلساته الخاصة بكل مجموعة محفوظة.' : count === 2 ? 'يبقى الوكيلان وجلساتهما الخاصة بكل مجموعة محفوظة.' : 'تبقى الوكلاء نفسها وجلساتها الخاصة بكل مجموعة محفوظة.'}`,
    dropToRoom: 'أفلت للإرفاق — يراه كل وكيل يرد',
    everyoneMeta: 'كل وكيل في محادثة المجموعة',
    holdReleaseHint: 'أشر إلى وكيل متوقف مؤقتًا أو أرسل @all resume للاستئناف.',
    manageDesc: 'يمكن للوكيل الانضمام إلى عدة محادثات مجموعات. تُزامن العضويات على جميع أجهزة الكمبيوتر.',
    manageMembersDesc: (max, group) =>
      `اختر ما بين وكيلين و${arabicAgents(max, 'وكيلين')} لـ «${group}». تبقى محادثة المجموعة وسجلها وجلسات أعضائها كما هي.`,
    memberCount: count => arabicAgents(count),
    memberSizeError: max => `تحتاج محادثة المجموعة إلى ما بين وكيلين و${arabicAgents(max, 'وكيلين')}`,
    mentionBot: 'وكيل في محادثة المجموعة هذه',
    needsYourInput: 'يحتاج وكيل في محادثة المجموعة هذه إلى ردك',
    noBots: 'لا توجد وكلاء في محادثة المجموعة هذه',
    noBotsMatch: query => `لا توجد وكلاء تطابق «${query}»`,
    noMembersToSend: group =>
      `لا يوجد أعضاء لإرسال الرسالة إليهم في «${group}» — أضف وكيلًا أو أعد فتح محادثة المجموعة إذا كان تحميل الأعضاء مستمرًا.`,
    searchToAdd: 'البحث عن وكلاء لإضافتها',
    searchToAddPlaceholder: 'ابحث عن وكلاء لإضافتها…',
    slashCommandsUnsupported:
      'الأوامر التي تبدأ بشرطة مائلة غير متاحة في محادثات المجموعات. افتح محادثة فردية مع وكيل لاستخدامها.'
  },
  roster: {
    allGateways: 'كل الأجهزة',
    allHidden: 'جميع الوكلاء مخفية',
    botsAndGroups: 'الوكلاء ومحادثات المجموعات',
    botsOnly: 'الوكلاء فقط',
    currentGateway: 'الجهاز الحالي',
    emptyDesc: 'أنشئ وكيلك الأول.',
    emptyTitle: 'لا توجد وكلاء بعد',
    gatewayRemoved: 'أُزيل الجهاز',
    newBotOrGroup: 'وكيل أو محادثة مجموعة جديدة',
    noHiddenMatch: 'لا توجد وكلاء مخفية تطابق هذه المرشحات.',
    noMatchFilters: 'لا توجد وكلاء أو محادثات مجموعات تطابق هذه المرشحات.',
    noMatchFiltersOn: gateway => `لا توجد وكلاء أو محادثات مجموعات تطابق هذه المرشحات على ${gateway}`,
    noMatchQuery: query => `لا توجد وكلاء أو محادثات مجموعات تطابق «${query}»`,
    noMatchQueryOn: (query, gateway) => `لا توجد وكلاء أو محادثات مجموعات تطابق «${query}» على ${gateway}`,
    rosterUnavailable: reason =>
      `القائمة غير متاحة: ${reason}. إذا كان إصدار Hermes لا يدعم profiles.list، فحدّث Hermes وأعد تشغيله.`,
    search: 'البحث عن وكلاء ومحادثات مجموعات',
    searchPlaceholder: 'ابحث عن وكلاء ومحادثات مجموعات…',
    showHidden: 'إظهار الوكلاء المخفية',
    title: 'الوكلاء',
    waitingForGateway: 'في انتظار الاتصال بالجهاز… (قد تستغرق الأجهزة البعيدة بضع ثوانٍ؛ تُعاد المحاولة تلقائيًا)'
  },
  screen: {
    autoOpenMenu: 'فتح الشاشة عندما يستخدمها الوكيل',
    autoOpenOnToast: name => `تُفتح شاشة ${name} عندما يبدأ الوكيل باستخدام سطح مكتبه`
  },
  sections: {
    deleted: (name, count) => (count === 0 ? `تم حذف «${name}»` : `تم حذف «${name}» — عدد الوكلاء دون قسم: ${count}`),
    emptyHint: 'اسحب الوكلاء هنا',
    headingTip: 'أفلت الوكلاء هنا · انقر مرتين لتغيير الاسم'
  }
}

const ru: BotsWording = {
  bot: {
    attentionBlocked: 'Агент заблокирован — см. последнее сообщение',
    attentionProviderAuth: 'Войдите заново для этого агента',
    groupsLoadFailed: 'Не удалось загрузить группы агента',
    hiddenToast: name => `${name} скрыт — нажмите значок глаза над списком агентов, чтобы его увидеть`,
    loadFailed: 'Не удалось загрузить агента',
    metadataLoadFailed: 'Не удалось загрузить метаданные агента',
    newChatWith: 'Новый чат с этим агентом',
    newTitle: 'Новый агент',
    openBotChat: 'Открыть чат с агентом',
    openBotChatInNewTab: 'Открыть чат с агентом в новой вкладке'
  },
  canonical: {
    createDuplicateMember:
      'Для группового чата, размещённого на этом компьютере, нужны разные агенты с уникальными именами; @all и @everyone зарезервированы.',
    createEnded:
      'Hermes отклонил этот групповой чат: возможно, он был изменён или удалён в другом месте. Повторите попытку, чтобы создать новый.',
    createGroupSize: 'В групповом чате, размещённом на этом компьютере, должно быть от двух до шести агентов.',
    createOtherConnection:
      'В групповой чат, размещённый на этом компьютере, можно добавить только агентов этого подключения.',
    createRefused: 'Hermes не смог создать этот групповой чат.',
    createUncertain:
      'Hermes не ответил, поэтому этот групповой чат, возможно, уже создан. Повторите попытку, чтобы открыть его. Второй чат создан не будет.',
    disbandMemberHomeUnknown:
      'Работа в групповом чате может продолжаться, но один из агентов больше не настроен на компьютере, на котором работает групповой чат. Верните этого агента в hosted_rooms.profiles, чтобы подтвердить его остановку, затем повторите роспуск группового чата.',
    disbandServiceRequired:
      'Чтобы остановить агентов и отозвать их доступ, Hermes должен работать на компьютере, который ведёт этот групповой чат. Запустите Hermes там или восстановите подключение к нему, затем повторите роспуск.',
    disbandStopPending:
      'Не все агенты в этом групповом чате подтвердили остановку. Повторите роспуск группового чата, когда все подтвердят её. Для ожидающего хода используйте повтор или отмену, если эти действия доступны в групповом чате.',
    disbandUnconfirmed:
      'Не удалось подтвердить завершение группового чата. Компьютер одного из агентов может быть в спящем режиме или не в сети. Повторите попытку, когда он снова будет доступен.',
    discardWarning:
      'Изменения уже могли произойти. Отмена хода не отменяет их. Для агента на другом компьютере нужно дождаться подтверждения, что ход остановлен или не начинался.',
    driverUnavailable:
      'Этот групповой чат недоступен. Обновите Hermes на компьютере, на котором он работает, или подключитесь к этому компьютеру заново.',
    groupDriverStopped:
      'Сейчас групповые чаты здесь не открываются: на компьютере этого подключения не запущены групповые чаты. Перезапустите или обновите Hermes на том компьютере и повторите попытку. Ничего не потеряно: прежние сообщения остаются на месте.',
    groupsUnreachable:
      'Не удалось связаться с компьютером этого подключения, чтобы открыть этот групповой чат. Убедитесь, что Hermes там запущен и подключение есть, и повторите попытку. Ничего не потеряно: прежние сообщения остаются на месте.',
    legacyRoom:
      'Этим групповым чатом управлял сам Desktop, поэтому он не может продолжиться на этом компьютере. Начните новую группу с теми же агентами: она продолжит работать, когда Desktop закрыт. Прежние сообщения в неё не копируются.',
    legacyRoomBlocked:
      'Этим групповым чатом управлял сам Desktop, поэтому он не может продолжиться на этом компьютере, а из этих агентов в нынешнем составе нельзя создать групповой чат, размещённый на этом компьютере (причина ниже). Начните новый групповой чат и выберите для него агентов. Прежние сообщения в него не копируются.',
    legacyRoomElsewhere:
      'Этот групповой чат не может продолжиться здесь: в нём есть агенты на других компьютерах, а в групповой чат, размещённый на этом компьютере, можно добавить агентов только одного подключения. Переключитесь на подключение, где работают эти агенты, и начните группу там или начните новый групповой чат с агентами этого подключения. Прежние сообщения в новую группу не копируются.',
    otherConnections: 'Агенты на других подключениях: {connections}',
    peerJoinChanged:
      '{name} не может присоединиться к этому групповому чату: настройки агента на его компьютере изменились после создания группового чата. Чтобы добавить его, начните новый групповой чат.',
    peerStartWithoutConfirm:
      'Создаётся новый групповой чат с остальными агентами. Исходный завершится только после их присоединения и подтверждения остановки и отзыва доступа каждого исходного участника. При сбое любого шага оба чата сохраняются. Прежние сообщения не копируются. Повторное действие после создания создаст ещё один групповой чат.',
    peerStartWithoutRetry: 'Нажмите «Повторить» для этих агентов в новом групповом чате.',
    peerTargetSetup: '{connection} пока не может предоставить агента групповому чату на другом компьютере. {reason}',
    readTimedOut:
      'Hermes не ответил вовремя. Возможно, он ждёт, пока ответит компьютер агента. Нажмите «Обновить», чтобы повторить попытку.',
    reasonAgentBlocked: 'Агент заблокирован',
    reasonGatewayRestarted: 'Hermes на компьютере агента перезапущен',
    reasonMemberUnavailable: 'Агент недоступен',
    reasonPrepareFailed: 'Не удалось запустить агентов',
    reasonRuntimeOffline: 'Агент не в сети',
    reasonSlotWait: 'Работает слишком много агентов',
    reasonTargetBusy: 'Агент занят',
    reasonTargetScope: 'Агент не смог получить доступ к своим учётным данным',
    sendRefused: 'Hermes отклонил это сообщение. Измените его и отправьте снова.',
    turnNoAnswer: 'агент не ответил вовремя'
  },
  crossGateway: {
    gatewayPeer: (owner, local = false) =>
      `Работает на ${local ? 'этом устройстве' : owner}, даже когда этот Desktop закрыт. На других компьютерах агенты принимают только текст.`,
    ownerCannotSeatPeers: place =>
      `Hermes на ${place} пока не может принимать агентов с других компьютеров. Обновите там Hermes, запустите его шлюз (hermes gateway run) и повторите.`,
    peerGatewayChanged: place =>
      `На ${place} теперь работает другая установка Hermes, не та, из которой этот агент. Переподключите его и повторите.`,
    peerRouteMissing: 'Компьютер одного из выбранных агентов не подключён. Переподключите его и повторите.',
    pickBots: max => `Выберите от 2 до ${max} агентов.`,
    selectionElsewhere: connection =>
      `Все эти агенты находятся на ${connection}. Добавьте агента из этого подключения или переключитесь на ${connection}, чтобы создать группу из его агентов.`
  },
  group: {
    activityBot: 'Агент',
    activitySlotWait: 'не смог начать — работает слишком много агентов',
    allHeldStatus: count => (count === 1 ? 'Агент приостановлен' : `Все агенты приостановлены (${count})`),
    attachHint: 'Прикрепить файлы — все отвечающие агенты их увидят',
    composerPlaceholder: 'Напишите что-нибудь — все агенты в этом групповом чате увидят сообщение.',
    createBotFirst: 'Агентов пока нет — сначала создайте агента.',
    noSeatableBots: 'Пока ни один ваш агент не может присоединиться к групповому чату здесь:',
    cannotJoinHere: 'Эти агенты не могут присоединиться к групповому чату здесь:',
    nonePicked: 'Агенты ещё не выбраны',
    unseatableHere:
      'Hermes на этой машине не добавляет его в групповые чаты — он здесь не обслуживается или выводится из работы',
    createDesc: max =>
      `Выберите от 2 до ${max} агентов. Локальное членство синхронизируется через каждого агента; для агентов с других компьютеров сведения об участии хранятся только в этом групповом чате.`,
    created: (group, count) =>
      `Группа «${group}» создана. В ней ${count} ${russianNoun(count, 'агент', 'агента', 'агентов')}.`,
    deleteDesc: group =>
      `Группа «${group}» будет удалена из списка групп её агентов, а общая история группового чата будет очищена. Агенты и их личные чаты сохранятся.`,
    disbandDescSuffix: count =>
      ` будет удалена у ${count} ${russianNoun(count, 'агента', 'агентов', 'агентов')}, а общая история группового чата будет очищена. ${count === 1 ? 'Сам агент и его сеансы' : 'Сами агенты и их сеансы'} в группах сохранятся.`,
    dropToRoom: 'Перетащите для прикрепления — все отвечающие агенты увидят файл',
    everyoneMeta: 'Все агенты в групповом чате',
    holdReleaseHint: 'Упомяните приостановленного агента или отправьте @all resume для продолжения.',
    manageDesc:
      'Агент может участвовать в нескольких групповых чатах. Состав групп синхронизируется на всех компьютерах.',
    manageMembersDesc: (max, group) =>
      `Выберите от 2 до ${max} агентов для «${group}». Групповой чат, его история и сеансы участников сохранятся.`,
    memberCount: count => `${count} ${russianNoun(count, 'агент', 'агента', 'агентов')}`,
    memberSizeError: max => `Для группового чата нужно от 2 до ${max} агентов`,
    mentionBot: 'Агент в этом групповом чате',
    needsYourInput: 'Агенту в этом групповом чате нужен ваш ответ',
    noBots: 'В этом групповом чате нет агентов',
    noBotsMatch: query => `Нет агентов по запросу «${query}»`,
    noMembersToSend: group =>
      `В группе «${group}» нет получателей — добавьте агента или откройте групповой чат заново, если участники ещё загружаются.`,
    searchToAdd: 'Поиск агентов для добавления',
    searchToAddPlaceholder: 'Найти агентов для добавления…',
    sendRouteRequired: 'Для отправки нужны подключение, агент и идентификатор группового чата',
    slashCommandsUnsupported:
      'Команды с косой чертой недоступны в групповых чатах. Используйте их в личном чате с агентом.'
  },
  roster: {
    allGateways: 'Все компьютеры',
    allHidden: 'Все агенты скрыты',
    botsAndGroups: 'Агенты и групповые чаты',
    botsOnly: 'Только агенты',
    currentGateway: 'Текущий компьютер',
    emptyDesc: 'Создайте первого агента.',
    emptyTitle: 'Агентов пока нет',
    gatewayRemoved: 'Компьютер удалён',
    newBotOrGroup: 'Новый агент или групповой чат',
    noHiddenMatch: 'Нет скрытых агентов с такими фильтрами.',
    noMatchFilters: 'Нет агентов или групповых чатов с такими фильтрами.',
    noMatchFiltersOn: gateway => `Нет агентов или групповых чатов с такими фильтрами на ${gateway}`,
    noMatchQuery: query => `Нет агентов или групповых чатов по запросу «${query}»`,
    noMatchQueryOn: (query, gateway) => `Нет агентов или групповых чатов по запросу «${query}» на ${gateway}`,
    rosterUnavailable: reason =>
      `Список недоступен: ${reason}. Если ваша версия Hermes не поддерживает profiles.list, обновите и перезапустите Hermes.`,
    search: 'Поиск агентов и групповых чатов',
    searchPlaceholder: 'Поиск агентов и групповых чатов…',
    showHidden: 'Показать скрытых агентов',
    title: 'Агенты',
    waitingForGateway:
      'Ожидание подключения к компьютеру… (удалённые компьютеры могут отвечать несколько секунд; повторная попытка автоматически)'
  },
  screen: {
    autoOpenMenu: 'Открывать экран, когда агент его использует',
    autoOpenOnToast: name => `Экран ${name} открывается, когда агент начинает использовать рабочий стол`
  },
  sections: {
    deleted: (name, count) =>
      count === 0 ? `Раздел «${name}» удалён` : `Раздел «${name}» удалён — агентов без раздела: ${count}`,
    emptyHint: 'Перетащите сюда агентов',
    headingTip: 'Перетащите сюда агентов · двойной щелчок для переименования'
  }
}

const fr: BotsWording = {
  bot: {
    attentionBlocked: 'Agent bloqué — consultez son dernier message',
    attentionProviderAuth: 'Reconnectez-vous pour cet agent',
    groupsLoadFailed: 'Impossible de charger les groupes de l’agent',
    hiddenToast: name => `${name} masqué — utilisez le bouton en forme d’œil de la liste des agents pour le retrouver`,
    loadFailed: 'Impossible de charger l’agent',
    metadataLoadFailed: 'Impossible de charger les données de l’agent',
    newChatWith: 'Nouvelle discussion avec cet agent',
    newTitle: 'Nouvel agent',
    openBotChat: 'Ouvrir la discussion avec l’agent',
    openBotChatInNewTab: 'Ouvrir la discussion avec l’agent dans un nouvel onglet'
  },
  canonical: {
    createDuplicateMember:
      'Une discussion de groupe hébergée sur cette machine nécessite des agents distincts avec des identifiants uniques ; @all et @everyone sont réservés.',
    createEnded:
      'Hermes a refusé cette discussion de groupe : elle a peut-être été modifiée ou supprimée ailleurs. Réessayez pour en créer une nouvelle.',
    createGroupSize: 'Une discussion de groupe hébergée sur cette machine nécessite de deux à six agents.',
    createOtherConnection:
      'Une discussion de groupe hébergée sur cette machine ne peut accueillir que des agents de cette connexion.',
    createRefused: 'Hermes n’a pas pu créer cette discussion de groupe.',
    createUncertain:
      'Hermes n’a pas répondu : cette discussion de groupe existe peut-être déjà. Réessayez pour l’ouvrir. Elle ne sera pas créée deux fois.',
    disbandMemberHomeUnknown:
      'La discussion de groupe peut encore être en cours, mais un de ses agents n’est plus configuré sur l’ordinateur qui exécute la discussion de groupe. Ajoutez à nouveau cet agent à hosted_rooms.profiles pour pouvoir confirmer son arrêt, puis réessayez de dissoudre la discussion de groupe.',
    disbandServiceRequired:
      'Hermes doit fonctionner sur l’ordinateur qui exécute cette discussion de groupe pour arrêter ses agents et révoquer leurs accès. Démarrez ou reconnectez Hermes sur cet ordinateur, puis réessayez de dissoudre la discussion de groupe.',
    disbandStopPending:
      'Les agents de cette discussion de groupe n’ont pas tous confirmé leur arrêt. Réessayez de dissoudre la discussion une fois que tous l’auront confirmé. Pour un tour en attente, utilisez Réessayer ou Ignorer si ces actions sont proposées dans la discussion de groupe.',
    disbandUnconfirmed:
      'La fin de la discussion de groupe n’a pas pu être confirmée. L’ordinateur d’un agent est peut-être en veille ou hors ligne. Réessayez lorsqu’il sera de nouveau disponible.',
    discardWarning:
      'Des changements ont peut-être déjà eu lieu. Abandonner le tour ne les annule pas. Pour un agent sur un autre ordinateur, il faut attendre que cet ordinateur confirme que le tour est arrêté ou n’a jamais démarré.',
    driverUnavailable:
      'Cette discussion de groupe est indisponible. Mettez à jour Hermes sur la machine qui l’héberge ou reconnectez-vous à cette machine.',
    groupDriverStopped:
      'Les discussions de groupe ne peuvent pas s’ouvrir ici pour le moment : les discussions de groupe ne s’exécutent pas sur la machine de cette connexion. Redémarrez ou mettez à jour Hermes sur cette machine, puis réessayez. Rien n’est perdu : les messages précédents restent où ils sont.',
    groupsUnreachable:
      'Impossible de joindre la machine de cette connexion pour ouvrir cette discussion de groupe. Vérifiez que Hermes y fonctionne et que vous êtes connecté, puis réessayez. Rien n’est perdu : les messages précédents restent où ils sont.',
    legacyRoom:
      'Desktop gérait lui-même cette discussion de groupe, elle ne peut donc pas continuer sur cette machine. Démarrez un nouveau groupe avec les mêmes agents : il continue de fonctionner quand Desktop est fermé. Les messages précédents n’y sont pas copiés.',
    legacyRoomBlocked:
      'Desktop gérait lui-même cette discussion de groupe, elle ne peut donc pas continuer sur cette machine, et ces agents ne peuvent pas former une discussion de groupe hébergée sur cette machine tels quels (la raison figure ci-dessous). Démarrez une nouvelle discussion de groupe et choisissez ses agents. Les messages précédents n’y sont pas copiés.',
    legacyRoomElsewhere:
      'Cette discussion de groupe ne peut pas continuer ici : elle comprend des agents sur d’autres ordinateurs, et une discussion de groupe hébergée sur cette machine n’accueille que les agents d’une seule connexion. Passez à la connexion où ces agents fonctionnent et démarrez-y un groupe, ou démarrez une nouvelle discussion de groupe avec des agents de cette connexion. Les messages précédents ne sont pas copiés dans un nouveau groupe.',
    otherConnections: 'Agents sur d’autres connexions : {connections}',
    peerJoinChanged:
      '{name} ne peut pas rejoindre cette discussion de groupe : les réglages de l’agent sur son ordinateur ont changé depuis la création de la discussion de groupe. Démarrez une nouvelle discussion de groupe pour l’inclure.',
    peerStartWithoutConfirm:
      'Une nouvelle discussion de groupe est créée avec les autres agents. L’originale ne se termine qu’après leur arrivée et la confirmation de l’arrêt et de la révocation des accès de chaque membre d’origine. Si une étape échoue, les deux discussions sont conservées. Les messages précédents ne sont pas copiés. Réutiliser cette action après la création crée une autre discussion de groupe.',
    peerStartWithoutRetry: 'Utilisez Réessayer pour ces agents dans la nouvelle discussion de groupe.',
    peerTargetSetup:
      '{connection} ne peut pas encore accueillir de agent pour une discussion de groupe sur un autre ordinateur. {reason}',
    readTimedOut:
      'Hermes n’a pas répondu à temps. Il attend peut-être l’ordinateur d’un agent. Choisissez Actualiser pour réessayer.',
    reasonAgentBlocked: 'L’agent est bloqué',
    reasonGatewayRestarted: 'Hermes a redémarré sur la machine de l’agent',
    reasonMemberUnavailable: 'L’agent est indisponible',
    reasonPrepareFailed: 'Les agents n’ont pas pu démarrer',
    reasonRuntimeOffline: 'L’agent est hors ligne',
    reasonSlotWait: 'Trop de agents sont en cours d’exécution',
    reasonTargetBusy: 'L’agent est occupé',
    reasonTargetScope: 'L’agent n’a pas pu accéder à ses identifiants',
    sendRefused: 'Hermes a refusé ce message. Modifiez-le puis renvoyez-le.',
    turnNoAnswer: 'l’agent n’a pas répondu à temps'
  },
  crossGateway: {
    gatewayPeer: (owner, local = false) =>
      `Fonctionne sur ${local ? 'cet appareil' : owner}, même lorsque ce Desktop est fermé. Les agents des autres ordinateurs acceptent uniquement du texte.`,
    ownerCannotSeatPeers: place =>
      `Hermes sur ${place} ne peut pas encore accueillir de agents d’autres ordinateurs. Mettez-y Hermes à jour et lancez sa passerelle (hermes gateway run), puis réessayez.`,
    peerGatewayChanged: place =>
      `${place} exécute maintenant une autre installation de Hermes que celle de cet agent. Reconnectez-le, puis réessayez.`,
    peerRouteMissing: 'L’ordinateur d’un agent sélectionné n’est pas connecté. Reconnectez-le, puis réessayez.',
    pickBots: max => `Choisissez de 2 à ${max} agents.`,
    selectionElsewhere: connection =>
      `Ces agents se trouvent tous sur ${connection}. Ajoutez un agent de cette connexion, ou passez sur ${connection} pour créer un groupe avec ses agents.`
  },
  group: {
    activityBot: 'Un agent',
    activitySlotWait: 'n’a pas pu démarrer — trop de agents en cours d’exécution',
    allHeldStatus: count => (count === 1 ? 'L’agent est en pause' : `Les ${count} agents sont tous en pause`),
    attachHint: 'Joindre des fichiers — tous les agents qui répondent les verront',
    composerPlaceholder: 'Écrivez un message — tous les agents de cette discussion de groupe le verront.',
    createBotFirst: 'Aucun agent pour le moment — créez-en un d’abord.',
    noSeatableBots: 'Aucun de vos agents ne peut encore rejoindre une discussion de groupe ici :',
    cannotJoinHere: 'Ces agents ne peuvent pas rejoindre une discussion de groupe ici :',
    nonePicked: 'Aucun agent choisi pour l’instant',
    unseatableHere:
      'Hermes sur cette machine ne l’accepte pas dans les discussions de groupe — il n’est pas servi ici ou il est en cours de retrait',
    createDesc: max =>
      `Choisissez de 2 à ${max} agents. Les appartenances locales se synchronisent via chaque agent ; pour les agents sur d’autres ordinateurs, l’appartenance n’est enregistrée que dans cette discussion de groupe.`,
    created: (group, count) => `« ${group} » créé avec ${count} agent${count > 1 ? 's' : ''}`,
    deleteDesc: group =>
      `Ceci retire « ${group} » des groupes de ses agents et efface l’historique partagé de la discussion de groupe. Les agents et leurs discussions individuelles sont conservés.`,
    disbandDescSuffix: count =>
      ` de ${count === 1 ? 'son agent' : `ses ${count} agents`} et efface l’historique partagé de la discussion de groupe. ${count === 1 ? 'L’agent et ses sessions' : 'Les agents et leurs sessions'} par groupe sont conservés.`,
    dropToRoom: 'Déposer pour joindre — tous les agents qui répondent le verront',
    everyoneMeta: 'Tous les agents de la discussion de groupe',
    holdReleaseHint: 'Mentionnez un agent en pause ou envoyez @all resume pour reprendre.',
    manageDesc:
      'Un agent peut rejoindre plusieurs discussions de groupe. Les adhésions sont synchronisées sur chaque ordinateur.',
    manageMembersDesc: (max, group) =>
      `Choisissez de 2 à ${max} agents pour « ${group} ». La discussion de groupe, son historique et les sessions de ses membres sont conservés.`,
    memberCount: count => `${count} agent${count > 1 ? 's' : ''}`,
    memberSizeError: max => `Une discussion de groupe nécessite de 2 à ${max} agents`,
    mentionBot: 'Agent de cette discussion de groupe',
    needsYourInput: 'Un agent de cette discussion de groupe attend votre réponse',
    noBots: 'Aucun agent dans cette discussion de groupe',
    noBotsMatch: query => `Aucun agent ne correspond à « ${query} »`,
    noMembersToSend: group =>
      `${group} n’a aucun membre à qui envoyer le message — ajoutez un agent ou rouvrez la discussion de groupe si les membres sont encore en cours de chargement.`,
    searchToAdd: 'Rechercher des agents à ajouter',
    searchToAddPlaceholder: 'Rechercher des agents à ajouter…',
    sendRouteRequired: 'L’envoi dans cette discussion de groupe nécessite sa connexion, son agent et son identifiant',
    slashCommandsUnsupported:
      'Les commandes slash ne sont pas disponibles dans les discussions de groupe. Ouvrez une discussion individuelle avec un agent pour les utiliser.'
  },
  roster: {
    allGateways: 'Toutes les machines',
    allHidden: 'Tous les agents sont masqués',
    botsAndGroups: 'Agents et discussions de groupe',
    botsOnly: 'Agents uniquement',
    currentGateway: 'Machine actuelle',
    emptyDesc: 'Créez votre premier agent.',
    emptyTitle: 'Aucun agent pour le moment',
    gatewayRemoved: 'Machine supprimée',
    newBotOrGroup: 'Nouvel agent ou discussion de groupe',
    noHiddenMatch: 'Aucun agent masqué ne correspond à ces filtres.',
    noMatchFilters: 'Aucun agent ni discussion de groupe ne correspond à ces filtres.',
    noMatchFiltersOn: gateway => `Aucun agent ni discussion de groupe ne correspond à ces filtres sur ${gateway}`,
    noMatchQuery: query => `Aucun agent ni discussion de groupe ne correspond à « ${query} »`,
    noMatchQueryOn: (query, gateway) =>
      `Aucun agent ni discussion de groupe ne correspond à « ${query} » sur ${gateway}`,
    rosterUnavailable: reason =>
      `Liste indisponible : ${reason}. Si votre version de Hermes ne prend pas en charge profiles.list, mettez Hermes à jour et redémarrez-le.`,
    search: 'Rechercher des agents et des discussions de groupe',
    searchPlaceholder: 'Rechercher des agents et des discussions de groupe…',
    showHidden: 'Afficher les agents masqués',
    title: 'Agents',
    waitingForGateway:
      'En attente de la connexion à la machine… (les machines distantes peuvent prendre quelques secondes ; nouvelle tentative automatique)'
  },
  screen: {
    autoOpenMenu: 'Ouvrir l’écran lorsque l’agent l’utilise',
    autoOpenOnToast: name => `L’écran de ${name} s’ouvre lorsque l’agent commence à utiliser son bureau`
  },
  sections: {
    deleted: (name, count) =>
      count === 0
        ? `Section « ${name} » supprimée`
        : `Section « ${name} » supprimée — ${count} agent${count === 1 ? '' : 's'} sans section`,
    emptyHint: 'Glissez des agents ici',
    headingTip: 'Déposez des agents ici · double-cliquez pour renommer'
  }
}

const de: BotsWording = {
  bot: {
    attentionBlocked: 'Agent blockiert — letzte Nachricht ansehen',
    attentionProviderAuth: 'Für diesen Agenten erneut anmelden',
    groupsLoadFailed: 'Agent-Gruppen konnten nicht geladen werden',
    hiddenToast: name => `${name} ausgeblendet — das Augensymbol über der Agenten-Liste zeigt ausgeblendete Agenten`,
    loadFailed: 'Agent konnte nicht geladen werden',
    metadataLoadFailed: 'Agent-Metadaten konnten nicht geladen werden',
    newChatWith: 'Neuer Chat mit diesem Agenten',
    newTitle: 'Neuer Agent',
    openBotChat: 'Agenten-Chat öffnen',
    openBotChatInNewTab: 'Agenten-Chat in neuem Tab öffnen'
  },
  canonical: {
    createDuplicateMember:
      'Ein auf diesem Rechner gehosteter Gruppenchat braucht unterschiedliche Agenten mit eindeutigen Handles; @all und @everyone sind reserviert.',
    createEnded:
      'Hermes hat diesen Gruppenchat abgelehnt. Er wurde möglicherweise an anderer Stelle geändert oder entfernt. Versuchen Sie es erneut, um einen neuen zu erstellen.',
    createGroupSize: 'Ein auf diesem Rechner gehosteter Gruppenchat braucht zwei bis sechs Agenten.',
    createOtherConnection:
      'Ein auf diesem Rechner gehosteter Gruppenchat kann nur Agenten dieser Verbindung aufnehmen.',
    createRefused: 'Hermes konnte diesen Gruppenchat nicht erstellen.',
    createUncertain:
      'Hermes hat nicht geantwortet, daher gibt es diesen Gruppenchat möglicherweise schon. Versuchen Sie es erneut, um ihn zu öffnen. Er wird nicht doppelt erstellt.',
    disbandMemberHomeUnknown:
      'Der Gruppenchat arbeitet möglicherweise noch, aber einer seiner Agenten ist auf dem Computer, auf dem der Gruppenchat läuft, nicht mehr konfiguriert. Fügen Sie den Agenten wieder zu hosted_rooms.profiles hinzu, damit sein Stopp bestätigt werden kann. Versuchen Sie dann erneut, den Gruppenchat aufzulösen.',
    disbandServiceRequired:
      'Hermes muss auf dem Computer laufen, der diesen Gruppenchat ausführt, um die Agenten zu stoppen und ihren Zugriff aufzuheben. Starten Sie Hermes dort oder verbinden Sie es erneut und versuchen Sie dann erneut, den Gruppenchat aufzulösen.',
    disbandStopPending:
      'Die Agenten in diesem Gruppenchat haben noch nicht alle bestätigt, dass sie gestoppt wurden. Versuchen Sie erneut, den Gruppenchat aufzulösen, sobald alle dies bestätigt haben. Nutzen Sie bei einem wartenden Turn „Erneut versuchen“ oder „Verwerfen“, wenn diese Aktionen im Gruppenchat angeboten werden.',
    disbandUnconfirmed:
      'Das Ende des Gruppenchats konnte nicht bestätigt werden. Der Computer eines Agenten ist möglicherweise im Ruhezustand oder offline. Versuchen Sie es erneut, sobald er wieder erreichbar ist.',
    discardWarning:
      'Es können bereits Änderungen erfolgt sein. Verwerfen macht sie nicht rückgängig. Bei einem Agent auf einem anderen Computer wird gewartet, bis dieser Computer bestätigt, dass der Turn gestoppt wurde oder nie begonnen hat.',
    driverUnavailable:
      'Dieser Gruppenchat ist nicht verfügbar. Aktualisieren Sie Hermes auf dem Rechner, auf dem er läuft, oder verbinden Sie sich erneut mit diesem Rechner.',
    groupDriverStopped:
      'Gruppenchats lassen sich hier gerade nicht öffnen: Auf dem Rechner dieser Verbindung laufen keine Gruppenchats. Starten Sie Hermes auf diesem Rechner neu oder aktualisieren Sie es, und versuchen Sie es dann erneut. Nichts geht verloren: Frühere Nachrichten bleiben, wo sie sind.',
    groupsUnreachable:
      'Der Rechner dieser Verbindung war nicht erreichbar, um diesen Gruppenchat zu öffnen. Prüfen Sie, ob Hermes dort läuft und Sie verbunden sind, und versuchen Sie es dann erneut. Nichts geht verloren: Frühere Nachrichten bleiben, wo sie sind.',
    legacyRoom:
      'Desktop hat diesen Gruppenchat selbst ausgeführt, daher kann er auf diesem Rechner nicht weiterlaufen. Starten Sie eine neue Gruppe mit denselben Agenten: Sie arbeitet weiter, wenn Desktop geschlossen ist. Frühere Nachrichten werden nicht in sie kopiert.',
    legacyRoomBlocked:
      'Desktop hat diesen Gruppenchat selbst ausgeführt, daher kann er auf diesem Rechner nicht weiterlaufen, und diese Agenten können so keinen auf diesem Rechner gehosteten Gruppenchat bilden (den Grund finden Sie unten). Starten Sie einen neuen Gruppenchat und wählen Sie seine Agenten. Frühere Nachrichten werden nicht in ihn kopiert.',
    legacyRoomElsewhere:
      'Dieser Gruppenchat kann hier nicht weiterlaufen: Er enthält Agenten auf anderen Computern, und ein auf diesem Rechner gehosteter Gruppenchat nimmt nur Agenten einer einzigen Verbindung auf. Wechseln Sie zu der Verbindung, auf der diese Agenten laufen, und starten Sie dort eine Gruppe, oder starten Sie einen neuen Gruppenchat mit Agenten dieser Verbindung. Frühere Nachrichten werden nicht in eine neue Gruppe kopiert.',
    otherConnections: 'Agenten auf anderen Verbindungen: {connections}',
    peerJoinChanged:
      '{name} kann diesem Gruppenchat nicht beitreten: Die Agent-Einstellungen auf seinem Computer haben sich seit dem Erstellen des Gruppenchats geändert. Starten Sie einen neuen Gruppenchat, um ihn aufzunehmen.',
    peerStartWithoutConfirm:
      'Ein neuer Gruppenchat wird mit den anderen Agenten erstellt. Der ursprüngliche endet erst, wenn alle beigetreten sind und Stopp und Zugriffsentzug aller ursprünglichen Mitglieder bestätigt wurden. Schlägt ein Schritt fehl, bleiben beide Gruppenchats erhalten. Frühere Nachrichten werden nicht kopiert. Nach der Erstellung legt diese Aktion beim nächsten Mal einen weiteren Gruppenchat an.',
    peerStartWithoutRetry: 'Verwenden Sie für diese Agenten im neuen Gruppenchat „Erneut versuchen“.',
    peerTargetSetup:
      '{connection} kann noch keinen Agenten für einen Gruppenchat auf einem anderen Computer bereitstellen. {reason}',
    readTimedOut:
      'Hermes hat nicht rechtzeitig geantwortet. Möglicherweise wartet es auf den Computer eines Agenten. Wählen Sie „Aktualisieren“, um es erneut zu versuchen.',
    reasonAgentBlocked: 'Der Agent ist blockiert',
    reasonGatewayRestarted: 'Hermes auf dem Rechner des Agenten wurde neu gestartet',
    reasonMemberUnavailable: 'Der Agent ist nicht verfügbar',
    reasonPrepareFailed: 'Die Agenten konnten nicht gestartet werden',
    reasonRuntimeOffline: 'Der Agent ist offline',
    reasonSlotWait: 'Es laufen zu viele Agenten',
    reasonTargetBusy: 'Der Agent ist beschäftigt',
    reasonTargetScope: 'Der Agent konnte nicht auf seine Zugangsdaten zugreifen',
    sendRefused: 'Hermes hat diese Nachricht abgelehnt. Bearbeiten Sie sie und senden Sie sie erneut.',
    turnNoAnswer: 'der Agent hat nicht rechtzeitig geantwortet'
  },
  crossGateway: {
    gatewayPeer: (owner, local = false) =>
      `Läuft auf ${local ? 'diesem Gerät' : owner}, auch wenn dieser Desktop geschlossen ist. Auf anderen Computern empfangen Agenten nur Text.`,
    ownerCannotSeatPeers: place =>
      `Hermes auf ${place} kann noch keine Agenten anderer Computer aufnehmen. Aktualisieren Sie Hermes dort, starten Sie sein Gateway (hermes gateway run) und versuchen Sie es erneut.`,
    peerGatewayChanged: place =>
      `Auf ${place} läuft jetzt eine andere Hermes-Installation als die, von der dieser Agent stammt. Verbinden Sie ihn neu und versuchen Sie es erneut.`,
    peerRouteMissing:
      'Der Computer eines ausgewählten Agenten ist nicht verbunden. Verbinden Sie ihn neu und versuchen Sie es erneut.',
    pickBots: max => `Wählen Sie 2–${max} Agenten.`,
    selectionElsewhere: connection =>
      `Diese Agenten liegen alle auf ${connection}. Füge einen Agenten dieser Verbindung hinzu oder wechsle zu ${connection}, um eine Gruppe mit dessen Agenten zu starten.`
  },
  group: {
    activityBot: 'Ein Agent',
    activitySlotWait: 'konnte nicht starten — zu viele Agenten laufen',
    allHeldStatus: count => (count === 1 ? 'Der Agent ist pausiert' : `Alle ${count} Agenten sind pausiert`),
    attachHint: 'Dateien anhängen — jeder antwortende Agent sieht sie',
    composerPlaceholder: 'Schreibe etwas — jeder Agent in diesem Gruppenchat sieht deine Nachricht.',
    createBotFirst: 'Noch keine Agenten — erstellen Sie zuerst einen.',
    noSeatableBots: 'Noch keiner Ihrer Agenten kann hier einem Gruppenchat beitreten:',
    cannotJoinHere: 'Diese Agenten können hier keinem Gruppenchat beitreten:',
    nonePicked: 'Noch keine Agenten ausgewählt',
    unseatableHere:
      'Hermes auf dieser Maschine nimmt ihn nicht in Gruppenchats auf – er wird hier nicht bereitgestellt oder wird gerade stillgelegt',
    createDesc: max =>
      `Wählen Sie 2–${max} Agenten. Lokale Mitgliedschaften werden über den jeweiligen Agenten synchronisiert; für Agenten auf anderen Computern wird die Mitgliedschaft nur in diesem Gruppenchat gespeichert.`,
    created: (group, count) => `„${group}“ mit ${count} Agenten erstellt`,
    deleteDesc: group =>
      `Dies entfernt „${group}“ aus den Gruppen seiner Agenten und löscht den gemeinsamen Verlauf des Gruppenchats. Die Agenten und ihre einzelnen Chats bleiben erhalten.`,
    disbandDescSuffix: count =>
      ` von ${count === 1 ? 'ihrem Agenten' : `ihren ${count} Agenten`} und löscht den gemeinsamen Verlauf des Gruppenchats. ${count === 1 ? 'Der Agent selbst und seine Sitzungen' : 'Die Agenten selbst und ihre Sitzungen'} pro Gruppe bleiben erhalten.`,
    dropToRoom: 'Zum Anhängen ablegen — jeder antwortende Agent sieht es',
    everyoneMeta: 'Jeder Agent im Gruppenchat',
    holdReleaseHint: 'Erwähne einen pausierten Agenten oder sende @all resume, um fortzufahren.',
    manageDesc:
      'Ein Agent kann mehreren Gruppenchats beitreten. Mitgliedschaften werden auf allen Computern synchronisiert.',
    manageMembersDesc: (max, group) =>
      `Wählen Sie 2–${max} Agenten für „${group}“. Der Gruppenchat, sein Verlauf und die Sitzungen seiner Mitglieder bleiben erhalten.`,
    memberCount: count => `${count} Agent${count === 1 ? '' : 'en'}`,
    memberSizeError: max => `Ein Gruppenchat braucht 2 bis ${max} Agenten`,
    mentionBot: 'Agent in diesem Gruppenchat',
    needsYourInput: 'Ein Agent in diesem Gruppenchat benötigt deine Eingabe',
    noBots: 'Keine Agenten in diesem Gruppenchat',
    noBotsMatch: query => `Keine Agenten für „${query}“ gefunden`,
    noMembersToSend: group =>
      `${group} hat keine Empfänger — füge einen Agenten hinzu oder öffne den Gruppenchat erneut, falls die Mitglieder noch laden.`,
    searchToAdd: 'Agenten zum Hinzufügen suchen',
    searchToAddPlaceholder: 'Agenten zum Hinzufügen suchen…',
    sendRouteRequired: 'Zum Senden an diesen Gruppenchat werden Verbindung, Agent und Gruppenchat-ID benötigt',
    slashCommandsUnsupported:
      'Slash-Befehle sind in Gruppenchats nicht verfügbar. Öffne dafür einen einzelnen Agenten-Chat.'
  },
  roster: {
    allGateways: 'Alle Rechner',
    allHidden: 'Alle Agenten sind ausgeblendet',
    botsAndGroups: 'Agenten und Gruppenchats',
    botsOnly: 'Nur Agenten',
    currentGateway: 'Aktueller Rechner',
    emptyDesc: 'Erstelle deinen ersten Agenten.',
    emptyTitle: 'Noch keine Agenten',
    gatewayRemoved: 'Rechner entfernt',
    newBotOrGroup: 'Neuer Agent oder Gruppenchat',
    noHiddenMatch: 'Keine ausgeblendeten Agenten entsprechen diesen Filtern.',
    noMatchFilters: 'Keine Agenten oder Gruppenchats entsprechen diesen Filtern.',
    noMatchFiltersOn: gateway => `Keine Agenten oder Gruppenchats entsprechen diesen Filtern auf ${gateway}`,
    noMatchQuery: query => `Keine Agenten oder Gruppenchats für „${query}“ gefunden`,
    noMatchQueryOn: (query, gateway) => `Keine Agenten oder Gruppenchats für „${query}“ auf ${gateway} gefunden`,
    rosterUnavailable: reason =>
      `Liste nicht verfügbar: ${reason}. Falls deine Hermes-Version profiles.list noch nicht unterstützt, aktualisiere Hermes und starte es neu.`,
    search: 'Agenten und Gruppenchats suchen',
    searchPlaceholder: 'Agenten und Gruppenchats suchen…',
    showHidden: 'Ausgeblendete Agenten anzeigen',
    title: 'Agenten',
    waitingForGateway:
      'Warte auf die Verbindung zum Rechner… (entfernte Rechner können einige Sekunden benötigen; automatischer neuer Versuch)'
  },
  screen: {
    autoOpenMenu: 'Bildschirm öffnen, wenn der Agent ihn verwendet',
    autoOpenOnToast: name => `Der Bildschirm von ${name} öffnet sich, sobald der Agent seinen Desktop verwendet`
  },
  sections: {
    deleted: (name, count) =>
      count === 0
        ? `„${name}“ gelöscht`
        : `„${name}“ gelöscht — ${count} Agent${count === 1 ? '' : 'en'} jetzt nicht zugeordnet`,
    emptyHint: 'Agenten hierher ziehen',
    headingTip: 'Agenten hier ablegen · zum Umbenennen doppelklicken'
  }
}

const es: BotsWording = {
  bot: {
    attentionBlocked: 'Agente bloqueado — consulta su último mensaje',
    attentionProviderAuth: 'Vuelve a iniciar sesión con este agente',
    groupsLoadFailed: 'No se pudieron cargar los grupos del agente',
    hiddenToast: name => `${name} oculto — usa el botón del ojo en la lista de agentes para verlo`,
    loadFailed: 'No se pudo cargar el agente',
    metadataLoadFailed: 'No se pudieron cargar los datos del agente',
    newChatWith: 'Nuevo chat con este agente',
    newTitle: 'Nuevo agente',
    openBotChat: 'Abrir chat con el agente',
    openBotChatInNewTab: 'Abrir chat con el agente en una pestaña nueva'
  },
  canonical: {
    createDuplicateMember:
      'Un chat de grupo alojado en esta máquina necesita agentes distintos con identificadores únicos; @all y @everyone están reservados.',
    createEnded:
      'Hermes rechazó este chat de grupo; puede que se haya cambiado o eliminado en otro lugar. Inténtalo de nuevo para crear uno nuevo.',
    createGroupSize: 'Un chat de grupo alojado en esta máquina necesita de dos a seis agentes.',
    createOtherConnection: 'Un chat de grupo alojado en esta máquina solo puede incluir agentes de esta conexión.',
    createRefused: 'Hermes no pudo crear este chat de grupo.',
    createUncertain:
      'Hermes no respondió, así que este chat de grupo puede que ya exista. Inténtalo de nuevo para abrirlo. No se creará dos veces.',
    disbandMemberHomeUnknown:
      'El chat de grupo puede seguir trabajando, pero uno de sus agentes ya no está configurado en el equipo que ejecuta el chat de grupo. Vuelve a añadir ese agente a hosted_rooms.profiles para poder confirmar su parada y vuelve a intentar disolver el chat de grupo.',
    disbandServiceRequired:
      'Hermes debe estar en ejecución en el equipo que gestiona este chat de grupo para detener sus agentes y revocar sus accesos. Inicia o reconecta Hermes allí y vuelve a intentar disolver el chat de grupo.',
    disbandStopPending:
      'Los agentes de este chat de grupo aún no han confirmado todos que se hayan detenido. Vuelve a intentar disolver el chat cuando todos lo confirmen. Para un turno pendiente, usa Reintentar o Descartar si esas acciones se ofrecen en el chat de grupo.',
    disbandUnconfirmed:
      'No pudimos confirmar que el chat de grupo haya terminado. El equipo de un agente puede estar suspendido o sin conexión. Reinténtalo cuando vuelva a estar disponible.',
    discardWarning:
      'Es posible que ya se hayan producido cambios. Descartar no los deshace. Si el agente está en otro equipo, se espera a que ese equipo confirme que el turno se ha detenido o nunca llegó a empezar.',
    driverUnavailable:
      'Este chat de grupo no está disponible. Actualiza Hermes en la máquina que lo aloja o vuelve a conectarte a esa máquina.',
    groupDriverStopped:
      'Los chats de grupo no se pueden abrir aquí ahora mismo: los chats de grupo no se están ejecutando en la máquina de esta conexión. Reinicia o actualiza Hermes en esa máquina y vuelve a intentarlo. No se pierde nada: los mensajes anteriores siguen donde están.',
    groupsUnreachable:
      'No se pudo contactar con la máquina de esta conexión para abrir este chat de grupo. Comprueba que Hermes se esté ejecutando allí y que estés conectado, y vuelve a intentarlo. No se pierde nada: los mensajes anteriores siguen donde están.',
    legacyRoom:
      'Desktop ejecutaba este chat de grupo por sí mismo, así que no puede continuar en esta máquina. Inicia un grupo nuevo con los mismos agentes: seguirá funcionando cuando Desktop esté cerrado. Los mensajes anteriores no se copian en él.',
    legacyRoomBlocked:
      'Desktop ejecutaba este chat de grupo por sí mismo, así que no puede continuar en esta máquina, y estos agentes no pueden formar un chat de grupo alojado en esta máquina tal como están (el motivo está abajo). Inicia un chat de grupo nuevo y elige sus agentes. Los mensajes anteriores no se copian en él.',
    legacyRoomElsewhere:
      'Este chat de grupo no puede continuar aquí: tiene agentes en otros equipos, y un chat de grupo alojado en esta máquina solo incluye agentes de una conexión. Cambia a la conexión donde funcionan esos agentes e inicia un grupo allí, o inicia un chat de grupo nuevo con agentes de esta conexión. Los mensajes anteriores no se copian en un grupo nuevo.',
    otherConnections: 'Agentes en otras conexiones: {connections}',
    peerJoinChanged:
      '{name} no puede unirse a este chat de grupo: la configuración del agente en su equipo cambió desde que se creó el chat de grupo. Inicia un chat de grupo nuevo para incluirlo.',
    peerStartWithoutConfirm:
      'Se crea un nuevo chat de grupo con los demás agentes. El original solo termina cuando todos se unen y cada miembro original confirma la detención y la revocación de acceso. Si falla algún paso, se conservan ambos chats. Los mensajes anteriores no se copian. Usar esta acción de nuevo tras la creación genera otro chat de grupo.',
    peerStartWithoutRetry: 'Usa Reintentar para estos agentes en el nuevo chat de grupo.',
    peerTargetSetup: '{connection} todavía no puede aportar un agente a un chat de grupo de otro equipo. {reason}',
    readTimedOut:
      'Hermes no respondió a tiempo. Puede que esté esperando al equipo de un agente. Elige Actualizar para reintentarlo.',
    reasonAgentBlocked: 'El agente está bloqueado',
    reasonGatewayRestarted: 'Hermes se reinició en la máquina del agente',
    reasonMemberUnavailable: 'El agente no está disponible',
    reasonPrepareFailed: 'No se pudieron iniciar los agentes',
    reasonRuntimeOffline: 'El agente está desconectado',
    reasonSlotWait: 'Hay demasiados agentes en ejecución',
    reasonTargetBusy: 'El agente está ocupado',
    reasonTargetScope: 'El agente no pudo acceder a sus credenciales',
    sendRefused: 'Hermes rechazó este mensaje. Edítalo y envíalo de nuevo.',
    turnNoAnswer: 'el agente no respondió a tiempo'
  },
  crossGateway: {
    gatewayPeer: (owner, local = false) =>
      `Funciona en ${local ? 'este dispositivo' : owner}, incluso cuando este Desktop está cerrado. Los agentes de otros equipos solo reciben texto.`,
    ownerCannotSeatPeers: place =>
      `Hermes en ${place} todavía no puede incorporar agentes de otros equipos. Actualiza Hermes allí, ejecuta su pasarela (hermes gateway run) y vuelve a intentarlo.`,
    peerGatewayChanged: place =>
      `${place} ejecuta ahora otra instalación de Hermes distinta de la de este agente. Vuelve a conectarlo y vuelve a intentarlo.`,
    peerRouteMissing:
      'El equipo de un agente seleccionado no está conectado. Vuelve a conectarlo y vuelve a intentarlo.',
    pickBots: max => `Elige de 2 a ${max} agentes.`,
    selectionElsewhere: connection =>
      `Todos estos agentes están en ${connection}. Añade un agente de esta conexión o cambia a ${connection} para crear un grupo con sus agentes.`
  },
  group: {
    activityBot: 'Un agente',
    activitySlotWait: 'no pudo iniciar — hay demasiados agentes en ejecución',
    allHeldStatus: count => (count === 1 ? 'El agente está en pausa' : `Los ${count} agentes están en pausa`),
    attachHint: 'Adjuntar archivos — todos los agentes que respondan los verán',
    composerPlaceholder: 'Escribe algo — todos los agentes de este chat de grupo verán tu mensaje.',
    createBotFirst: 'Aún no hay agentes; crea uno primero.',
    noSeatableBots: 'Ninguno de tus agentes puede unirse todavía a un chat de grupo aquí:',
    cannotJoinHere: 'Estos agentes no pueden unirse a un chat de grupo aquí:',
    nonePicked: 'Aún no has elegido agentes',
    unseatableHere: 'Hermes en esta máquina no lo admite en chats de grupo: no se sirve aquí o se está retirando',
    createDesc: max =>
      `Elige de 2 a ${max} agentes. Las pertenencias locales se sincronizan a través de cada agente; para los agentes de otros equipos, la pertenencia solo se guarda en este chat de grupo.`,
    created: (group, count) => `«${group}» creado con ${count} agente${count === 1 ? '' : 's'}`,
    deleteDesc: group =>
      `Esto quita «${group}» de los grupos de sus agentes y borra el historial compartido del chat de grupo. Los agentes y sus chats individuales se conservan.`,
    disbandDescSuffix: count =>
      ` de ${count === 1 ? 'su agente' : `sus ${count} agentes`} y borra el historial compartido del chat de grupo. ${count === 1 ? 'El agente y sus sesiones' : 'Los agentes y sus sesiones'} por grupo se conservan.`,
    dropToRoom: 'Suelta para adjuntar — todos los agentes que respondan lo verán',
    everyoneMeta: 'Todos los agentes del chat de grupo',
    holdReleaseHint: 'Menciona a un agente en pausa o envía @all resume para continuar.',
    manageDesc: 'Un agente puede unirse a varios chats de grupo. Las membresías se sincronizan en todos los equipos.',
    manageMembersDesc: (max, group) =>
      `Elige de 2 a ${max} agentes para «${group}». El chat de grupo, su historial y las sesiones de sus miembros se conservan.`,
    memberCount: count => `${count} agente${count === 1 ? '' : 's'}`,
    memberSizeError: max => `Un chat de grupo necesita entre 2 y ${max} agentes`,
    mentionBot: 'Agente de este chat de grupo',
    needsYourInput: 'Un agente de este chat de grupo necesita tu respuesta',
    noBots: 'No hay agentes en este chat de grupo',
    noBotsMatch: query => `Ningún agente coincide con «${query}»`,
    noMembersToSend: group =>
      `${group} no tiene destinatarios — añade un agente o vuelve a abrir el chat de grupo si aún se están cargando los miembros.`,
    searchToAdd: 'Buscar agentes para añadir',
    searchToAddPlaceholder: 'Buscar agentes para añadir…',
    sendRouteRequired: 'Para enviar a este chat de grupo se necesitan su conexión, agente e identificador',
    slashCommandsUnsupported:
      'Los comandos con barra no están disponibles en los chats de grupo. Abre un chat individual con un agente para usarlos.'
  },
  roster: {
    allGateways: 'Todas las máquinas',
    allHidden: 'Todos los agentes están ocultos',
    botsAndGroups: 'Agentes y chats de grupo',
    botsOnly: 'Solo agentes',
    currentGateway: 'Máquina actual',
    emptyDesc: 'Crea tu primer agente.',
    emptyTitle: 'Aún no hay agentes',
    gatewayRemoved: 'Máquina eliminada',
    newBotOrGroup: 'Nuevo agente o chat de grupo',
    noHiddenMatch: 'Ningún agente oculto coincide con estos filtros.',
    noMatchFilters: 'Ningún agente ni chat de grupo coincide con estos filtros.',
    noMatchFiltersOn: gateway => `Ningún agente ni chat de grupo coincide con estos filtros en ${gateway}`,
    noMatchQuery: query => `Ningún agente ni chat de grupo coincide con «${query}»`,
    noMatchQueryOn: (query, gateway) => `Ningún agente ni chat de grupo coincide con «${query}» en ${gateway}`,
    rosterUnavailable: reason =>
      `Lista no disponible: ${reason}. Si tu versión de Hermes no admite profiles.list, actualiza y reinicia Hermes.`,
    search: 'Buscar agentes y chats de grupo',
    searchPlaceholder: 'Buscar agentes y chats de grupo…',
    showHidden: 'Mostrar agentes ocultos',
    title: 'Agentes',
    waitingForGateway:
      'Esperando la conexión con la máquina… (las máquinas remotas pueden tardar unos segundos; se reintenta automáticamente)'
  },
  screen: {
    autoOpenMenu: 'Abrir la pantalla cuando el agente la use',
    autoOpenOnToast: name => `La pantalla de ${name} se abre cuando el agente empieza a usar su escritorio`
  },
  sections: {
    deleted: (name, count) =>
      count === 0
        ? `Sección «${name}» eliminada`
        : `Sección «${name}» eliminada — ${count} agente${count === 1 ? '' : 's'} sin sección`,
    emptyHint: 'Arrastra agentes aquí',
    headingTip: 'Suelta agentes aquí · doble clic para renombrar'
  }
}

export const BOTS_REDESIGN_WORDING: PluginLocaleBundles = {
  en,
  ja,
  zh,
  'zh-hant': zhHant,
  ar,
  ru,
  fr,
  de,
  es
}
