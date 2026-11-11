import { GROUP_REASON_LOCALES, type GroupReasonMessages } from './group-reason-locales'

export interface CanonicalGroupMessages extends GroupReasonMessages {
  members: string
  roomStopped: string
  roomDisbanded: string
  authorityLost: string
  workLimitReached: string
  turnFailed: string
  turnCancelled: string
  turnDeferred: string
  turnNoAnswer: string
  memberUnavailable: string
  openGroup: string
  loadingGroup: string
  loadingGroups: string
  emptyGroups: string
  driverUnavailable: string
  invalidLogCursor: string
  allowOnce: string
  deny: string
  discardUnknown: string
  discardWarning: string
  discardStopPending: string
  retryBusy: string
  discardPeerUnreachable: string
  discardPeerUpdateRequired: string
  confirmDiscard: string
  unconfirmedSend: string
  restoredPendingSend: string
  groupMessage: string
  attachFiles: string
  removeAttachment: string
  uploadFailed: string
  memberWorking: string
  memberQueued: string
  groupWorking: string
  statusWorking: string
  statusStalled: string
  statusReplied: string
  statusPassed: string
  statusFailed: string
  statusCancelled: string
  statusUnavailable: string
  statusIdle: string
  you: string
  roundOf: string
  round: string
  cycleWorking: string
  cycleSettled: string
  cycleBounded: string
  cycleMessageCap: string
  cycleStopped: string
  typing: string
  typingTool: string
  handoff: string
  replyTo: string
  replyingTo: string
  cancelReply: string
  composerPlaceholder: string
  groupDetails: string
  stopGroup: string
  pendingApproval: string
  pendingRetry: string
  pendingDiscard: string
  membersFixed: string
  rowWorking: string
  rowFailed: string
  rowNeedsYou: string
  lead: string
  roomSettings: string
  purposeLabel: string
  purposePlaceholder: string
  notesLabel: string
  notesPlaceholder: string
  deliveryTarget: string
  scheduledPost: string
  routingLabel: string
  routingEveryone: string
  routingLead: string
  routingEveryoneHint: string
  routingLeadHint: string
  leadBadge: string
  assignment: string
  assignmentFallback: string
  routedTo: string
  routedEveryone: string
  routedFallback: string
  pausedAfter: string
  continueDiscussion: string
  continued: string
  routingAuto: string
  routingAutoHint: string
  statusBlocked: string
  statusApprovals: string
  statusAttention: string
  stopped: string
  pendingActionUnconfirmed: string
  sendRefused: string
  sendStopped: string
  sendMaybe: string
  sendSettledElsewhere: string
  journalLoadFailed: string
  journalReload: string
  approvalAction: string
  approvalCommand: string
  approvalDetailsMissing: string
  checkingDriver: string
  legacyRoom: string
  startGatewayGroup: string
  createRefused: string
  createGroupSize: string
  createOtherConnection: string
  createDuplicateMember: string
  createUncertain: string
  createEnded: string
  legacyRoomElsewhere: string
  otherConnections: string
  legacyRoomBlocked: string
  groupsUnsupported: string
  groupDriverStopped: string
  groupsUnreachable: string
  earlierMessages: string
  peerTargetSetup: string
  peerSetupUpdate: string
  peerSetupApiServer: string
  peerSetupRunStorage: string
  peerSetupRoomKey: string
  peerSetupApprovals: string
  peerSetupAddress: string
  peerSetupBackground: string
  peerSetupGeneric: string
  peerJoinFailed: string
  peerJoinUnconfirmed: string
  peerJoinUnreachable: string
  peerJoinChanged: string
  readTimedOut: string
  disbandUnconfirmed: string
  disbandKept: string
  disbandPeerUnreachable: string
  disbandStopPending: string
  disbandMemberHomeUnknown: string
  disbandInProgress: string
  disbandRevokeUnsupported: string
  disbandServiceRequired: string
  peerStartWithoutRetry: string
  peerStartWithoutKept: string
  /** A new group chat whose bots from other computers did not all join; `{names}` lists them. */
  createdNotJoined: string
  peerNotJoinedBadge: string
  peerOfflineBadge: string
  peerRejoinBadge: string
  peerNotJoined: string
  peerRejoin: string
  peerAway: string
  peerRetryFailed: string
  peerJoinConnect: string
  peerStartWithout: string
  peerStartWithoutConfirm: string
}

export const CANONICAL_GROUP_LOCALES = {
  en: {
    ...GROUP_REASON_LOCALES['en'],
    roomStopped: 'Stop requested',
    roomDisbanded: 'Group disbanded',
    authorityLost: 'Group ownership changed',
    workLimitReached: 'Group work limit reached',
    turnFailed: 'Turn failed',
    turnCancelled: 'Turn cancelled',
    turnDeferred: 'Turn deferred',
    turnNoAnswer: 'the bot didn’t answer in time',
    memberUnavailable: 'Member unavailable',
    members: 'Members',
    openGroup: 'Open group chat',
    loadingGroup: 'Loading group…',
    loadingGroups: 'Loading group chats…',
    emptyGroups: 'No group chats found.',
    driverUnavailable: 'This group chat isn’t available. Update or reconnect the gateway that hosts it.',
    invalidLogCursor: "Couldn't load more of this group chat's history.",
    allowOnce: 'Allow once',
    deny: 'Deny',
    discardUnknown: 'Discard turn',
    discardWarning:
      'Side effects may already have occurred. Discarding does not undo them. For a bot on another computer, Discard waits for that computer to confirm the turn has stopped or never started.',
    discardStopPending:
      '{name} has not confirmed that its turn has stopped. The turn has been kept. Try Discard again when its computer is answering.',
    retryBusy: '{name} may still be working on this turn. Try again once it has finished.',
    discardPeerUnreachable:
      '{name}’s computer isn’t answering. The turn has been kept. Reconnect that computer, then try Discard again.',
    discardPeerUpdateRequired:
      'Hermes on {name}’s computer is too old to confirm that this turn never started. The turn has been kept. Update Hermes on that computer, then try Discard again.',
    confirmDiscard: 'Confirm discard',
    unconfirmedSend: 'The previous send is unconfirmed. Retry its original text before sending another message.',
    restoredPendingSend: 'An unconfirmed send was restored. Retry it before sending another message.',
    groupMessage: 'Group message',
    attachFiles: 'Attach files',
    removeAttachment: 'Remove attachment',
    uploadFailed: 'Upload failed',
    memberWorking: 'Working',
    memberQueued: 'queued',
    groupWorking: 'The group is working…',
    statusWorking: 'working',
    statusStalled: 'no progress for {time}',
    statusReplied: 'replied',
    statusPassed: 'passed',
    statusFailed: 'failed',
    statusCancelled: 'cancelled',
    statusUnavailable: 'unavailable',
    statusIdle: 'idle',
    you: 'You',
    roundOf: 'Round {n} of {max}',
    round: 'Round {n}',
    cycleWorking: 'Working: {names}',
    cycleSettled: 'Settled — nobody had more to say',
    cycleBounded: 'Reached the round limit',
    cycleMessageCap: 'Reached the message limit',
    cycleStopped: 'Stopped',
    typing: '{name} is working…',
    typingTool: '{name} is running {tool}…',
    handoff: '{from} asked {to}',
    replyTo: 'Reply to {name}',
    replyingTo: 'Replying to {name}',
    cancelReply: 'Cancel reply',
    composerPlaceholder: 'Message the group — @mention to assign',
    groupDetails: 'Group details',
    stopGroup: 'Stop the group',
    pendingApproval: '{name} is asking for approval. You decide.',
    pendingRetry: '{name}’s turn has not finished. Retry it?',
    pendingDiscard: 'Discard {name}’s turn? Work already done will not be undone.',
    membersFixed: 'Members are set when the group is created.',
    rowWorking: 'Working',
    rowFailed: 'A member failed',
    rowNeedsYou: 'Needs you',
    lead: 'Lead',
    roomSettings: 'Group chat settings',
    purposeLabel: 'Purpose',
    purposePlaceholder: 'What this group chat is for, in one line',
    notesLabel: 'Shared notes',
    notesPlaceholder: 'Facts every member should keep in mind',
    deliveryTarget: 'Scheduled jobs can post here with deliver:',
    scheduledPost: 'Scheduled',
    routingLabel: 'Who answers a message that names nobody',
    routingEveryone: 'Everyone answers',
    routingLead: 'Lead: {name}',
    routingEveryoneHint: 'Every member gets a turn, as today.',
    routingLeadHint: 'The lead picks who takes it; @mentions and @everyone still go straight to those members.',
    leadBadge: 'Lead',
    assignment: '{lead} gave this to {names}',
    assignmentFallback: '{lead} is taking this',
    routedTo: 'Assigned to {names}',
    routedEveryone: 'Assigned to everyone',
    routedFallback: "Couldn't choose who answers, so everyone will",
    pausedAfter: 'Paused after {count} messages',
    continueDiscussion: 'Continue',
    continued: 'Continued',
    routingAuto: 'Pick who answers automatically',
    routingAutoHint:
      'A quick model call picks who a message is for. @mentions still go straight to the people you name.',
    statusBlocked: 'Blocked',
    statusApprovals: '{count} waiting for approval',
    statusAttention: 'Needs attention: {count}',
    stopped: 'Stop requested. Turns awaiting confirmation stay visible; use Retry or Discard where offered.',
    pendingActionUnconfirmed: 'We couldn’t confirm this action. Refresh the group chat to check its status.',
    sendRefused: 'The gateway refused this message. Edit it and send again.',
    sendStopped: 'Stopped before sending. Your message is still here.',
    sendMaybe: 'This message may already have been sent. Retry sends the same message; it will not be posted twice.',
    sendSettledElsewhere:
      'Another Desktop window already finished with this message. Check the conversation before sending it again.',
    journalLoadFailed:
      'Saved messages couldn’t be loaded. Sending is paused so that a saved message is never sent twice.',
    journalReload: 'Try loading again',
    approvalAction: 'Action',
    approvalCommand: 'Command',
    approvalDetailsMissing: 'Action details are unavailable. Check again before allowing this.',
    checkingDriver: 'Checking group chats on this connection…',
    legacyRoom:
      'Desktop ran this group chat itself, so it can’t continue on this gateway. Start a new group with the same bots: it keeps working when Desktop is closed. Earlier messages aren’t copied into it.',
    startGatewayGroup: 'Start new group',
    createRefused: 'The gateway couldn’t create this group chat.',
    createGroupSize: 'A group chat hosted on this gateway needs two to six bots.',
    createOtherConnection: 'A group chat hosted on this gateway can only seat bots on this connection.',
    createDuplicateMember:
      'A group chat hosted on this gateway needs bots with different profiles and handles; @all and @everyone are reserved.',
    createUncertain:
      'The gateway didn’t answer, so this group chat may already exist. Try again to open it. It won’t be created twice.',
    createEnded:
      'The gateway refused this group chat, so it may have been changed or removed somewhere else. Try again to start a new one.',
    legacyRoomElsewhere:
      'This group chat can’t continue here: it has bots on other computers, and a group chat hosted on this gateway seats bots from one connection only. Switch to the connection where those bots run and start a group there, or start a new group chat with bots from this connection. Earlier messages aren’t copied into a new group.',
    otherConnections: 'Bots on other connections: {connections}',
    legacyRoomBlocked:
      'Desktop ran this group chat itself, so it can’t continue on this gateway, and these bots can’t form a group chat hosted on this gateway as they are (the reason is below). Start a new group chat and choose its bots. Earlier messages aren’t copied into it.',
    groupsUnsupported:
      'Group chats can’t open here: Hermes on this connection’s computer is too old for them. Update Hermes on that computer and reconnect, then try again. Nothing is lost: earlier messages stay where they are.',
    groupDriverStopped:
      'Group chats can’t open here right now: group chats aren’t running on this connection’s gateway. Restart or update Hermes on that computer, then try again. Nothing is lost: earlier messages stay where they are.',
    groupsUnreachable:
      'Couldn’t reach this connection’s gateway to open this group chat. Check that Hermes is running there and that you’re connected, then try again. Nothing is lost: earlier messages stay where they are.',
    earlierMessages: 'Earlier messages, read only. They stay on this Desktop and aren’t copied into a new group.',
    peerTargetSetup: '{connection} can’t host a bot for a group chat on another computer yet. {reason}',
    peerSetupUpdate: 'Hermes there is too old for it. Update Hermes on that computer, then check again.',
    peerSetupApiServer:
      'Its API server is off. Turn on the API server there (gateway.platforms.api_server), restart Hermes on that computer, then check again.',
    peerSetupRunStorage:
      'Its API server can’t save its run records, so a message could run twice. Make sure Hermes can write to its data folder on that computer, restart Hermes there, then check again.',
    peerSetupRoomKey:
      'Hermes there can’t create the key that protects access to group chats. Make sure Hermes can write to its data folder on that computer, restart Hermes there, then check again.',
    peerSetupApprovals:
      'Command approvals are off there, or its settings can’t be read. Set approvals to manual or smart on that computer (approvals.mode), then check again.',
    peerSetupAddress:
      'It has no address that the computer running this group chat can reach. Set gateway.room_link_url on that computer, restart Hermes there, then check again.',
    peerSetupBackground:
      'Hermes there stops when its Desktop closes. Run its gateway on its own there (hermes gateway run, or install it as a service), then check again.',
    peerSetupGeneric:
      'Update Hermes there, turn on its API server with an address the computer running this group chat can reach (gateway.room_link_url), and keep approvals on.',
    peerJoinFailed: '{name} hasn’t joined: {reason}',
    peerJoinUnconfirmed:
      'Could not confirm whether {name} joined. Check the group chat when its computer is reachable.',
    peerJoinUnreachable:
      '{name} hasn’t joined: the computer running this group chat couldn’t reach its computer. Check that its computer can be reached from there without Desktop, then try again.',
    peerJoinChanged:
      '{name} can’t join this group chat: its computer’s bot settings changed since the group chat was created. Start a new group chat to include it.',
    readTimedOut:
      'The gateway didn’t answer in time. It may be waiting for a bot’s computer. Choose Refresh to try again.',
    disbandUnconfirmed:
      'We couldn’t confirm that the group chat ended. A bot’s computer may be asleep or offline. Try again once it’s back.',
    disbandKept: 'The group chat has been kept. {reason}',
    disbandPeerUnreachable:
      'Its work was stopped. A member’s computer isn’t answering. Reconnect that computer, then try Disband again.',
    disbandStopPending:
      'The bots in this group chat haven’t all confirmed they’ve stopped. Try Disband again once they have. For a waiting turn, use Retry or Discard if offered in the group chat.',
    disbandMemberHomeUnknown:
      'The group chat may still be working, but one of its bot profiles is no longer configured on the computer running the group chat. Add that profile back to hosted_rooms.profiles so its Stop can be confirmed, then try Disband again.',
    disbandInProgress:
      'Disband could not finish. Wait a moment, check the group chat, then try Disband again if needed.',
    disbandRevokeUnsupported:
      'Its work was stopped. A member cannot remove its access to this group chat. Update Hermes on that member’s computer, then try Disband again.',
    disbandServiceRequired:
      'Hermes must be running on the computer that runs this group chat to stop its bots and remove their access. Start or reconnect Hermes there, then try Disband again.',
    peerStartWithoutRetry: 'Use Retry for these bots in the new group chat.',
    peerStartWithoutKept:
      'The new group chat was created and the original group chat has been kept. {reason} Disband the original when you’re ready.',
    createdNotJoined:
      'The group chat was created, but {names} hasn’t joined yet. The group chat says why; use Retry there.',
    peerNotJoinedBadge: 'not joined',
    peerOfflineBadge: 'offline',
    peerRejoinBadge: 'access expired',
    peerNotJoined: '{name} hasn’t joined this group chat yet.',
    peerRejoin:
      '{name}’s access to this group chat ended. To invite it again, use Retry from a Desktop connected to its computer.',
    peerAway: '{name}’s computer could not be reached. The next message to it will try again.',
    peerRetryFailed: 'We couldn’t reconnect {name}. Check that Hermes is running on its computer, then try again.',
    peerJoinConnect: '{name} hasn’t joined: connect to the computer it runs on, then choose Retry.',
    peerStartWithout: 'Start new group chat without {name}',
    peerStartWithoutConfirm:
      'A new group chat starts with the other bots. The original ends only after they all join and every original member confirms Stop and access revocation. If either step fails, both group chats stay. Earlier messages aren’t copied. Using this action again after creation makes another group chat.'
  },
  ja: {
    ...GROUP_REASON_LOCALES['ja'],
    roomStopped: '停止を要求しました',
    roomDisbanded: 'グループが解散しました',
    authorityLost: 'グループの管理元が変更されました',
    workLimitReached: 'グループの作業上限に達しました',
    turnFailed: 'ターンに失敗しました',
    turnCancelled: 'ターンがキャンセルされました',
    turnDeferred: 'ターンが延期されました',
    turnNoAnswer: 'ボットが時間内に応答しませんでした',
    memberUnavailable: 'メンバーは利用できません',
    members: 'メンバー',
    openGroup: 'グループチャットを開く',
    loadingGroup: 'グループを読み込み中…',
    loadingGroups: 'グループチャットを読み込み中…',
    emptyGroups: 'グループチャットが見つかりません。',
    driverUnavailable: 'このグループチャットは利用できません。実行元のゲートウェイを更新するか、再接続してください。',
    invalidLogCursor: 'グループチャットの履歴の続きを読み込めませんでした。',
    allowOnce: '今回のみ許可',
    deny: '拒否',
    discardUnknown: 'ターンを破棄',
    discardWarning:
      'すでに変更が生じている場合があります。破棄しても元には戻りません。別のコンピューターのボットでは、そのコンピューターがターンの停止または未開始を確認するまで待ちます。',
    discardStopPending:
      '{name} はターンの停止をまだ確認していません。ターンは保持されています。そのコンピューターが応答するようになったら、もう一度破棄してください。',
    retryBusy: '{name} はまだこのターンを実行中の可能性があります。終了してからもう一度お試しください。',
    discardPeerUnreachable:
      '{name} のコンピューターが応答していません。ターンは保持されています。そのコンピューターに再接続してから、もう一度破棄してください。',
    discardPeerUpdateRequired:
      '{name} のコンピューターの Hermes が古いため、このターンが開始されていないことを確認できません。ターンは保持されています。そのコンピューターで Hermes を更新してから、もう一度破棄してください。',
    confirmDiscard: '破棄を確定',
    unconfirmedSend: '前回の送信は未確認です。別のメッセージを送信する前に、元のテキストで再試行してください。',
    restoredPendingSend: '未確認の送信を復元しました。別のメッセージを送信する前に再試行してください。',
    groupMessage: 'グループメッセージ',
    attachFiles: 'ファイルを添付',
    removeAttachment: '添付ファイルを削除',
    uploadFailed: 'アップロードに失敗しました',
    memberWorking: '作業中',
    memberQueued: '待機中',
    groupWorking: 'グループが作業中です…',
    statusWorking: '作業中',
    statusStalled: '{time}進捗なし',
    statusReplied: '返信済み',
    statusPassed: 'パス',
    statusFailed: '失敗',
    statusCancelled: 'キャンセル',
    statusUnavailable: '利用不可',
    statusIdle: '待機中',
    you: 'あなた',
    roundOf: 'ラウンド {n}/{max}',
    round: 'ラウンド {n}',
    cycleWorking: '作業中: {names}',
    cycleSettled: '完了 — 誰も追加の発言はありません',
    cycleBounded: 'ラウンド上限に達しました',
    cycleMessageCap: 'メッセージ上限に達しました',
    cycleStopped: '停止しました',
    typing: '{name} が作業中…',
    typingTool: '{name} が {tool} を実行中…',
    handoff: '{from} が {to} に依頼',
    replyTo: '{name} に返信',
    replyingTo: '{name} に返信中',
    cancelReply: '返信をキャンセル',
    composerPlaceholder: 'グループにメッセージ — @メンションで担当を指名',
    groupDetails: 'グループの詳細',
    stopGroup: 'グループを停止',
    pendingApproval: '{name} が承認を求めています。判断してください。',
    pendingRetry: '{name} のターンはまだ完了していません。再試行しますか？',
    pendingDiscard: '{name} のターンを破棄しますか？すでに行われた作業は元に戻りません。',
    membersFixed: 'メンバーはグループ作成時に決まります。',
    rowWorking: '作業中',
    rowFailed: 'メンバーが失敗しました',
    rowNeedsYou: '対応が必要',
    lead: 'リード',
    roomSettings: 'グループチャット設定',
    purposeLabel: '目的',
    purposePlaceholder: 'このグループチャットの目的を一行で',
    notesLabel: '共有メモ',
    notesPlaceholder: '全メンバーが覚えておくべき事項',
    deliveryTarget: 'スケジュール済みジョブの投稿先 deliver:',
    scheduledPost: 'スケジュール',
    routingLabel: '誰も指名されていないメッセージに答えるのは',
    routingEveryone: '全員が答える',
    routingLead: 'リーダー: {name}',
    routingEveryoneHint: '今と同じく、全メンバーが順に答えます。',
    routingLeadHint: 'リーダーが担当者を選びます。@メンションと @everyone はそのまま宛先のメンバーに届きます。',
    leadBadge: 'リーダー',
    assignment: '{lead} が {names} に割り当てました',
    assignmentFallback: '{lead} が担当します',
    routedTo: '{names} に割り当てました',
    routedEveryone: '全員に割り当てました',
    routedFallback: '回答者を選べなかったため、全員が回答します',
    pausedAfter: '{count} 件のメッセージの後で一時停止しました',
    continueDiscussion: '続ける',
    continued: '再開しました',
    routingAuto: '回答者を自動で選ぶ',
    routingAutoHint: '短いモデル呼び出しで、メッセージの宛先を選びます。@メンションはそのまま指名した相手に届きます。',
    statusBlocked: 'ブロック中',
    statusApprovals: '承認待ち {count} 件',
    statusAttention: '要対応 {count} 件',
    stopped:
      '停止を要求しました。確認待ちのターンは表示されたままです。再試行や破棄が表示されていれば、その操作を使えます。',
    pendingActionUnconfirmed: 'この操作を確認できませんでした。グループチャットを更新して状態を確認してください。',
    sendRefused: 'ゲートウェイがこのメッセージを拒否しました。編集してからもう一度送信してください。',
    sendStopped: '送信前に停止しました。メッセージはそのまま残っています。',
    sendMaybe:
      'このメッセージは既に送信された可能性があります。再試行しても同じメッセージが二重に投稿されることはありません。',
    sendSettledElsewhere:
      'このメッセージは別の Desktop ウィンドウで処理済みです。もう一度送信する前に会話を確認してください。',
    journalLoadFailed:
      '保存済みメッセージを読み込めませんでした。保存済みメッセージを重複して送信しないよう、送信を一時停止しています。',
    journalReload: 'もう一度読み込む',
    approvalAction: '操作',
    approvalCommand: 'コマンド',
    approvalDetailsMissing: '操作の詳細を表示できません。許可する前に、詳細を更新してください。',
    checkingDriver: 'この接続のグループチャットを確認中…',
    legacyRoom:
      'このグループチャットは Desktop 自身が動かしていたため、このゲートウェイでは続けられません。同じボットで新しいグループを始めると、Desktop を閉じても動き続けます。以前のメッセージは新しいグループにコピーされません。',
    startGatewayGroup: '新しいグループを始める',
    createRefused: 'ゲートウェイはこのグループチャットを作成できませんでした。',
    createGroupSize: 'このゲートウェイで運用されるグループチャットには 2〜6 体のボットが必要です。',
    createOtherConnection:
      'このゲートウェイで運用されるグループチャットには、この接続上のボットだけを参加させられます。',
    createDuplicateMember:
      'このゲートウェイで運用されるグループチャットには、プロファイルとハンドルが異なるボットが必要です。@all と @everyone は予約済みです。',
    createUncertain:
      'ゲートウェイから応答がなかったため、このグループチャットはすでに作成されている可能性があります。もう一度試すと開きます。二重に作成されることはありません。',
    createEnded:
      'ゲートウェイがこのグループチャットを拒否しました。別の場所で変更または削除された可能性があります。もう一度試すと新しく作成します。',
    legacyRoomElsewhere:
      'このグループチャットはここでは続けられません。ほかのコンピューター上のボットが含まれていて、このゲートウェイで運用されるグループチャットには 1 つの接続のボットしか参加させられないためです。それらのボットが動いている接続に切り替えてそこでグループを始めるか、この接続のボットで新しいグループチャットを始めてください。以前のメッセージは新しいグループにコピーされません。',
    otherConnections: 'ほかの接続上のボット: {connections}',
    legacyRoomBlocked:
      'このグループチャットは Desktop 自身が動かしていたため、このゲートウェイでは続けられません。また、これらのボットはこのままではこのゲートウェイで運用されるグループチャットを作れません（理由は下記）。新しいグループチャットを始めてボットを選んでください。以前のメッセージはコピーされません。',
    groupsUnsupported:
      'ここではグループチャットを開けません。この接続先のコンピューターの Hermes が古く、グループチャットに対応していないためです。そのコンピューターの Hermes を更新して再接続してから、もう一度お試しください。何も失われません。以前のメッセージはそのまま残ります。',
    groupDriverStopped:
      'いまはここでグループチャットを開けません。この接続のゲートウェイでグループチャットが動いていないためです。そのコンピューターの Hermes を再起動または更新してから、もう一度お試しください。何も失われません。以前のメッセージはそのまま残ります。',
    groupsUnreachable:
      'このグループチャットを開くための、この接続のゲートウェイに接続できませんでした。そのコンピューターで Hermes が動いていて、接続されていることを確認してから、もう一度お試しください。何も失われません。以前のメッセージはそのまま残ります。',
    earlierMessages: '以前のメッセージ（閲覧のみ）。この Desktop に残り、新しいグループにはコピーされません。',
    peerTargetSetup:
      '{connection} は、まだほかのコンピューターのグループチャットにボットを参加させられません。{reason}',
    peerSetupUpdate:
      'そこの Hermes が古いためです。そのコンピューターの Hermes を更新してから、もう一度確認してください。',
    peerSetupApiServer:
      'API サーバーがオフになっています。そこで API サーバー（gateway.platforms.api_server）を有効にし、そのコンピューターの Hermes を再起動してから、もう一度確認してください。',
    peerSetupRunStorage:
      'API サーバーが実行記録を保存できないため、メッセージが二重に実行されるおそれがあります。そのコンピューターで Hermes がデータフォルダーに書き込めることを確認し、Hermes を再起動してから、もう一度確認してください。',
    peerSetupRoomKey:
      'そこの Hermes が、グループチャットへのアクセスを保護するキーを作成できません。そのコンピューターで Hermes がデータフォルダーに書き込めることを確認し、Hermes を再起動してから、もう一度確認してください。',
    peerSetupApprovals:
      'そこではコマンドの承認がオフになっているか、設定を読み取れません。そのコンピューターで承認を manual または smart（approvals.mode）に設定してから、もう一度確認してください。',
    peerSetupAddress:
      'このグループチャットを実行するコンピューターから届くアドレスがありません。そのコンピューターで gateway.room_link_url を設定し、Hermes を再起動してから、もう一度確認してください。',
    peerSetupBackground:
      'そこの Hermes は Desktop を閉じると停止します。そこでゲートウェイを単独で実行してから（hermes gateway run、またはサービスとしてインストール）、もう一度確認してください。',
    peerSetupGeneric:
      'そこで Hermes を更新し、このグループチャットを実行するコンピューターから届くアドレス（gateway.room_link_url）で API サーバーを有効にし、承認をオンのままにしてください。',
    peerJoinFailed: '{name} はまだ参加していません: {reason}',
    peerJoinUnconfirmed:
      '{name} が参加したか確認できませんでした。コンピューターに接続できるようになったら、グループチャットを確認してください。',
    peerJoinUnreachable:
      '{name} はまだ参加していません。このグループチャットを実行するコンピューターから、そのコンピューターに接続できませんでした。Desktop を介さずにそこから届くことを確認してから、もう一度お試しください。',
    peerJoinChanged:
      '{name} はこのグループチャットに参加できません。グループチャットの作成後に、そのコンピューターのボットの設定が変わりました。含めるには新しいグループチャットを始めてください。',
    readTimedOut:
      'ゲートウェイが時間内に応答しませんでした。ボットのコンピューターの応答を待っている可能性があります。「更新」を選んで再試行してください。',
    disbandUnconfirmed:
      'グループチャットの終了を確認できませんでした。ボットのコンピューターがスリープ中かオフラインの可能性があります。復帰してからもう一度お試しください。',
    disbandKept: 'グループチャットは保持されています。{reason}',
    disbandPeerUnreachable:
      '作業は停止しました。メンバーのコンピューターが応答していません。そのコンピューターに再接続してから、もう一度解散してください。',
    disbandStopPending:
      'このグループチャットのすべてのボットから停止の確認が取れたわけではありません。全員の確認が取れたら、もう一度解散してください。保留中のターンには、グループチャットに再試行や破棄が表示されていれば、その操作を使えます。',
    disbandMemberHomeUnknown:
      'このグループチャットの処理はまだ続いている可能性がありますが、ボットのプロファイルの一つがグループチャットを実行しているコンピューターの設定から外れています。そのプロファイルを hosted_rooms.profiles に戻して停止を確認できるようにしてから、もう一度解散してください。',
    disbandInProgress:
      '解散を完了できませんでした。少し待ってグループチャットを確認し、必要であればもう一度解散してください。',
    disbandRevokeUnsupported:
      '作業は停止しました。メンバーがこのグループチャットへのアクセス権を取り消せません。そのメンバーのコンピューターで Hermes を更新してから、もう一度解散してください。',
    disbandServiceRequired:
      'ボットを停止してアクセスを解除するには、このグループチャットを実行するコンピューターで Hermes が動作している必要があります。そこで Hermes を起動するか再接続してから、もう一度解散してください。',
    peerStartWithoutRetry: '新しいグループチャットで、これらのボットの「再試行」を使ってください。',
    peerStartWithoutKept:
      '新しいグループチャットを作成しました。元のグループチャットも保持されています。{reason} 準備ができたら元のグループチャットを解散してください。',
    createdNotJoined:
      'グループチャットを作成しましたが、{names} はまだ参加していません。理由はグループチャットに表示されます。そこで「再試行」を使ってください。',
    peerNotJoinedBadge: '未参加',
    peerOfflineBadge: 'オフライン',
    peerRejoinBadge: 'アクセス期限切れ',
    peerNotJoined: '{name} はまだこのグループチャットに参加していません。',
    peerRejoin:
      '{name} のこのグループチャットへのアクセス期限が切れました。再招待するには、そのコンピューターに接続された Desktop から「再試行」を選んでください。',
    peerAway:
      '{name} のコンピューターに接続できませんでした。次にそのメンバーにメッセージを送ると、もう一度接続を試みます。',
    peerRetryFailed:
      '{name} に再接続できませんでした。そのコンピューターで Hermes が起動していることを確認してから、もう一度お試しください。',
    peerJoinConnect:
      '{name} はまだ参加していません。それが動いているコンピューターに接続してから、「再試行」を選んでください。',
    peerStartWithout: '{name} を除いて新しいグループチャットを始める',
    peerStartWithoutConfirm:
      'ほかのボットで新しいグループチャットを作成します。全員が参加し、元の全メンバーの停止とアクセス取り消しが確認されてから、元のチャットを終了します。どちらかが失敗した場合は両方を保持します。以前のメッセージはコピーされません。 作成後にこの操作をもう一度行うと、別のグループチャットが作成されます。'
  },
  zh: {
    ...GROUP_REASON_LOCALES['zh'],
    roomStopped: '已请求停止',
    roomDisbanded: '群组已解散',
    authorityLost: '群组管理端已更改',
    workLimitReached: '已达到群组工作上限',
    turnFailed: '回合失败',
    turnCancelled: '回合已取消',
    turnDeferred: '回合已推迟',
    turnNoAnswer: '机器人未及时响应',
    memberUnavailable: '成员不可用',
    members: '成员',
    openGroup: '打开群聊',
    loadingGroup: '正在加载群组…',
    loadingGroups: '正在加载群聊…',
    emptyGroups: '未找到群聊。',
    driverUnavailable: '此群聊暂不可用。请更新或重新连接运行它的网关。',
    invalidLogCursor: '无法继续加载此群聊的历史记录。',
    allowOnce: '仅允许一次',
    deny: '拒绝',
    discardUnknown: '丢弃回合',
    discardWarning:
      '可能已经产生了更改，丢弃不会撤销这些更改。对于其他电脑上的机器人，丢弃会等待该电脑确认回合已停止或从未开始。',
    discardStopPending: '{name} 尚未确认回合已停止。回合已保留。请在其电脑恢复响应后再次尝试丢弃。',
    retryBusy: '{name} 可能仍在处理此回合。请在其完成后重试。',
    discardPeerUnreachable: '{name} 所在的电脑没有响应。回合已保留。请重新连接该电脑，然后再次尝试丢弃。',
    discardPeerUpdateRequired:
      '{name} 所在电脑上的 Hermes 版本太旧，无法确认此回合从未开始。回合已保留。请在该电脑上更新 Hermes，然后再次尝试丢弃。',
    confirmDiscard: '确认丢弃',
    unconfirmedSend: '上次发送尚未确认。请先重试发送原始文本，再发送其他消息。',
    restoredPendingSend: '已恢复尚未确认的发送。请先重试，再发送其他消息。',
    groupMessage: '群组消息',
    attachFiles: '附加文件',
    removeAttachment: '移除附件',
    uploadFailed: '上传失败',
    memberWorking: '工作中',
    memberQueued: '排队中',
    groupWorking: '群组正在处理…',
    statusWorking: '工作中',
    statusStalled: '{time}无进展',
    statusReplied: '已回复',
    statusPassed: '已跳过',
    statusFailed: '失败',
    statusCancelled: '已取消',
    statusUnavailable: '不可用',
    statusIdle: '空闲',
    you: '你',
    roundOf: '第 {n}/{max} 轮',
    round: '第 {n} 轮',
    cycleWorking: '工作中：{names}',
    cycleSettled: '已结束 — 没有人还有话要说',
    cycleBounded: '已达到轮次上限',
    cycleMessageCap: '已达到消息上限',
    cycleStopped: '已停止',
    typing: '{name} 正在工作…',
    typingTool: '{name} 正在运行 {tool}…',
    handoff: '{from} 请 {to} 接手',
    replyTo: '回复 {name}',
    replyingTo: '正在回复 {name}',
    cancelReply: '取消回复',
    composerPlaceholder: '发消息给群组 — @提及以指派',
    groupDetails: '群组详情',
    stopGroup: '停止群组',
    pendingApproval: '{name} 正在请求批准，由你决定。',
    pendingRetry: '{name} 的回合尚未完成。要重试吗？',
    pendingDiscard: '丢弃 {name} 的回合？已经完成的工作不会被撤销。',
    membersFixed: '成员在创建群组时确定。',
    rowWorking: '工作中',
    rowFailed: '有成员失败',
    rowNeedsYou: '需要你处理',
    lead: '负责人',
    roomSettings: '群聊设置',
    purposeLabel: '用途',
    purposePlaceholder: '用一句话说明这个群聊的用途',
    notesLabel: '共享笔记',
    notesPlaceholder: '每位成员都应记住的事项',
    deliveryTarget: '定时任务可通过 deliver: 发到这里',
    scheduledPost: '定时',
    routingLabel: '没有点名的消息由谁回答',
    routingEveryone: '所有人回答',
    routingLead: '负责人：{name}',
    routingEveryoneHint: '和现在一样，每位成员都轮流回答。',
    routingLeadHint: '由负责人决定谁来处理；@提及和 @everyone 仍直接发给对应成员。',
    leadBadge: '负责人',
    assignment: '{lead} 把这条交给了 {names}',
    assignmentFallback: '{lead} 亲自处理',
    routedTo: '已分配给 {names}',
    routedEveryone: '已分配给所有人',
    routedFallback: '无法确定由谁回答，所以所有人都会回答',
    pausedAfter: '在 {count} 条消息后暂停',
    continueDiscussion: '继续',
    continued: '已继续',
    routingAuto: '自动选择回答者',
    routingAutoHint: '一次快速的模型调用会判断消息是发给谁的。@提及仍直接发给你点名的人。',
    statusBlocked: '已阻塞',
    statusApprovals: '{count} 项等待批准',
    statusAttention: '{count} 项需要处理',
    stopped: '已请求停止。等待确认的回合会继续显示；如提供重试或丢弃操作，可以使用。',
    pendingActionUnconfirmed: '无法确认此操作。请刷新群聊以检查其状态。',
    sendRefused: '网关拒绝了这条消息。请编辑后重新发送。',
    sendStopped: '已在发送前停止。你的消息仍保留在这里。',
    sendMaybe: '这条消息可能已经发送。重试会发送同一条消息，不会重复发布。',
    sendSettledElsewhere: '另一个 Desktop 窗口已处理完这条消息。再次发送前请先查看对话。',
    journalLoadFailed: '无法加载已保存的消息。发送已暂停，以免重复发送已保存的消息。',
    journalReload: '重新加载',
    approvalAction: '操作',
    approvalCommand: '命令',
    approvalDetailsMissing: '无法显示操作详情。请先刷新详情，再允许此操作。',
    checkingDriver: '正在检查此连接上的群聊…',
    legacyRoom:
      '这个群聊由 Desktop 自己运行，因此无法在此网关上继续。用相同的机器人开始一个新群组：关闭 Desktop 后它仍会继续工作。之前的消息不会复制到新群组。',
    startGatewayGroup: '开始新群组',
    createRefused: '网关无法创建此群聊。',
    createGroupSize: '在此网关上托管的群聊需要 2 到 6 个机器人。',
    createOtherConnection: '在此网关上托管的群聊只能加入此连接上的机器人。',
    createDuplicateMember:
      '在此网关上托管的群聊中的机器人需要使用不同的配置文件和名称标识；@all 和 @everyone 为保留名称。',
    createUncertain: '网关没有响应，此群聊可能已经创建。请重试以打开它，不会重复创建。',
    createEnded: '网关拒绝了此群聊，它可能已在其他地方被更改或移除。请重试以新建一个。',
    legacyRoomElsewhere:
      '这个群聊无法在此继续：它包含其他电脑上的机器人，而在此网关上托管的群聊只能加入同一个连接上的机器人。请切换到这些机器人所在的连接并在那里开始群组，或者用此连接上的机器人开始一个新群聊。之前的消息不会复制到新群组。',
    otherConnections: '其他连接上的机器人：{connections}',
    legacyRoomBlocked:
      '这个群聊由 Desktop 自己运行，因此无法在此网关上继续，而且这些机器人目前无法组成在此网关上托管的群聊（原因见下方）。请开始一个新群聊并选择它的机器人。之前的消息不会复制到新群组。',
    groupsUnsupported:
      '无法在此打开群聊：此连接所在电脑上的 Hermes 版本太旧，不支持群聊。请在那台电脑上更新 Hermes 并重新连接，然后再试一次。不会丢失任何内容：之前的消息仍保留在原处。',
    groupDriverStopped:
      '暂时无法在此打开群聊：此连接的网关没有运行群聊。请在那台电脑上重启或更新 Hermes，然后再试一次。不会丢失任何内容：之前的消息仍保留在原处。',
    groupsUnreachable:
      '无法连接到此连接的网关来打开这个群聊。请确认那台电脑上的 Hermes 正在运行且你已连接，然后再试一次。不会丢失任何内容：之前的消息仍保留在原处。',
    earlierMessages: '之前的消息（只读）。它们保留在此 Desktop 上，不会复制到新群组。',
    peerTargetSetup: '{connection} 目前还无法让机器人加入其他电脑上的群聊。{reason}',
    peerSetupUpdate: '那里的 Hermes 版本太旧。请在那台电脑上更新 Hermes，然后再检查一次。',
    peerSetupApiServer:
      '它的 API 服务器未开启。请在那里开启 API 服务器（gateway.platforms.api_server），重启那台电脑上的 Hermes，然后再检查一次。',
    peerSetupRunStorage:
      '它的 API 服务器无法保存运行记录，消息可能会被执行两次。请确认 Hermes 能写入那台电脑上的数据文件夹，重启那里的 Hermes，然后再检查一次。',
    peerSetupRoomKey:
      '那里的 Hermes 无法创建保护群聊访问的密钥。请确认 Hermes 能写入那台电脑上的数据文件夹，重启那里的 Hermes，然后再检查一次。',
    peerSetupApprovals:
      '那里的命令批准已关闭，或无法读取其设置。请在那台电脑上将批准设为 manual 或 smart（approvals.mode），然后再检查一次。',
    peerSetupAddress:
      '它没有运行此群聊的电脑可以访问的地址。请在那台电脑上设置 gateway.room_link_url，重启那里的 Hermes，然后再检查一次。',
    peerSetupBackground:
      '那里的 Hermes 会在其 Desktop 关闭时停止。请在那里单独运行其网关（hermes gateway run，或将其安装为服务），然后再检查一次。',
    peerSetupGeneric:
      '请在那里更新 Hermes，用运行此群聊的电脑可以访问的地址（gateway.room_link_url）开启其 API 服务器，并保持批准开启。',
    peerJoinFailed: '{name} 尚未加入：{reason}',
    peerJoinUnconfirmed: '无法确认 {name} 是否已加入。请在能够连接到其电脑后检查群聊。',
    peerJoinUnreachable:
      '{name} 尚未加入：运行此群聊的电脑无法访问它所在的电脑。请确认无需 Desktop 也能从那里访问该电脑，然后重试。',
    peerJoinChanged:
      '{name} 无法加入此群聊：自群聊创建以来，它所在电脑上的机器人设置已更改。要加入它，请开始一个新群聊。',
    readTimedOut: '网关未及时响应。它可能正在等待某个机器人的电脑响应。请选择“刷新”重试。',
    disbandUnconfirmed: '无法确认群聊是否已经结束。某个机器人的电脑可能处于休眠或离线状态。请在它恢复连接后重试。',
    disbandKept: '群聊已保留。{reason}',
    disbandPeerUnreachable: '群聊中的工作已停止。某位成员所在的电脑没有响应。请重新连接该电脑，然后再次尝试解散。',
    disbandStopPending:
      '此群聊中的机器人尚未全部确认已停止。请在全部确认后再次尝试解散。如果群聊中为等待处理的回合提供了重试或丢弃操作，可以使用这些操作。',
    disbandMemberHomeUnknown:
      '群聊可能仍在运行，但其中一个机器人的配置档案已不在运行群聊的电脑的配置中。请将该配置档案重新加入 hosted_rooms.profiles，以便确认其已停止，然后再次尝试解散。',
    disbandInProgress: '未能完成解散。请稍等片刻，检查群聊状态，然后根据需要重试解散。',
    disbandRevokeUnsupported:
      '群聊中的工作已停止。某位成员无法撤销其对此群聊的访问权限。请在该成员所在的电脑上更新 Hermes，然后再次尝试解散。',
    disbandServiceRequired:
      '必须在运行此群聊的电脑上启动 Hermes，才能停止机器人并撤销其访问权限。请在那台电脑上启动或重新连接 Hermes，然后重试解散。',
    peerStartWithoutRetry: '请在新群聊中对这些机器人使用“重试”。',
    peerStartWithoutKept: '已创建新群聊，原群聊已保留。{reason} 准备好后可解散原群聊。',
    createdNotJoined: '群聊已创建，但 {names} 尚未加入。原因显示在群聊中，请在那里使用“重试”。',
    peerNotJoinedBadge: '未加入',
    peerOfflineBadge: '离线',
    peerRejoinBadge: '访问权限已过期',
    peerNotJoined: '{name} 尚未加入此群聊。',
    peerRejoin: '{name} 对此群聊的访问权限已过期。要再次邀请它，请在连接到其电脑的 Desktop 中选择重试。',
    peerAway: '未能连接到 {name} 所在的电脑。下次向它发送消息时会重试。',
    peerRetryFailed: '未能重新连接 {name}。请确认 Hermes 正在其电脑上运行，然后重试。',
    peerJoinConnect: '{name} 尚未加入：请连接到它所在的电脑，然后选择“重试”。',
    peerStartWithout: '不含 {name} 开始新群聊',
    peerStartWithoutConfirm:
      '将用其他机器人创建新群聊。只有它们全部加入，且原群聊所有成员确认停止并撤销访问权限后，原群聊才会结束。任一步骤失败都会保留两个群聊。之前的消息不会复制。 创建后再次使用此操作会创建另一个群聊。'
  },
  'zh-hant': {
    ...GROUP_REASON_LOCALES['zh-hant'],
    roomStopped: '已要求停止',
    roomDisbanded: '群組已解散',
    authorityLost: '群組管理端已變更',
    workLimitReached: '已達到群組工作上限',
    turnFailed: '回合失敗',
    turnCancelled: '回合已取消',
    turnDeferred: '回合已延後',
    turnNoAnswer: '機器人未及時回應',
    memberUnavailable: '成員無法使用',
    members: '成員',
    openGroup: '開啟群組聊天',
    loadingGroup: '正在載入群組…',
    loadingGroups: '正在載入群組聊天…',
    emptyGroups: '找不到群組聊天。',
    driverUnavailable: '此群組聊天目前無法使用。請更新或重新連線至執行它的閘道。',
    invalidLogCursor: '無法繼續載入此群組聊天的歷史記錄。',
    allowOnce: '僅允許一次',
    deny: '拒絕',
    discardUnknown: '捨棄回合',
    discardWarning:
      '可能已經產生了變更，捨棄不會復原這些變更。對於其他電腦上的機器人，捨棄會等待該電腦確認回合已停止或從未開始。',
    discardStopPending: '{name} 尚未確認回合已停止。回合已保留。請在其電腦恢復回應後再次嘗試捨棄。',
    retryBusy: '{name} 可能仍在處理此回合。請在完成後再試一次。',
    discardPeerUnreachable: '{name} 所在的電腦沒有回應。回合已保留。請重新連線至該電腦，然後再次嘗試捨棄。',
    discardPeerUpdateRequired:
      '{name} 所在電腦上的 Hermes 版本太舊，無法確認此回合從未開始。回合已保留。請在該電腦上更新 Hermes，然後再次嘗試捨棄。',
    confirmDiscard: '確認捨棄',
    unconfirmedSend: '上次傳送尚未確認。請先重試傳送原始文字，再傳送其他訊息。',
    restoredPendingSend: '已還原尚未確認的傳送。請先重試，再傳送其他訊息。',
    groupMessage: '群組訊息',
    attachFiles: '附加檔案',
    removeAttachment: '移除附件',
    uploadFailed: '上傳失敗',
    memberWorking: '工作中',
    memberQueued: '排隊中',
    groupWorking: '群組正在處理…',
    statusWorking: '工作中',
    statusStalled: '{time}無進展',
    statusReplied: '已回覆',
    statusPassed: '已略過',
    statusFailed: '失敗',
    statusCancelled: '已取消',
    statusUnavailable: '無法使用',
    statusIdle: '閒置',
    you: '你',
    roundOf: '第 {n}/{max} 輪',
    round: '第 {n} 輪',
    cycleWorking: '工作中：{names}',
    cycleSettled: '已結束 — 沒有人還有話要說',
    cycleBounded: '已達輪次上限',
    cycleMessageCap: '已達訊息上限',
    cycleStopped: '已停止',
    typing: '{name} 正在工作…',
    typingTool: '{name} 正在執行 {tool}…',
    handoff: '{from} 請 {to} 接手',
    replyTo: '回覆 {name}',
    replyingTo: '正在回覆 {name}',
    cancelReply: '取消回覆',
    composerPlaceholder: '傳訊息給群組 — @提及以指派',
    groupDetails: '群組詳細資料',
    stopGroup: '停止群組',
    pendingApproval: '{name} 正在請求核准，由你決定。',
    pendingRetry: '{name} 的回合尚未完成。要重試嗎？',
    pendingDiscard: '捨棄 {name} 的回合？已完成的工作不會被復原。',
    membersFixed: '成員在建立群組時決定。',
    rowWorking: '工作中',
    rowFailed: '有成員失敗',
    rowNeedsYou: '需要你處理',
    lead: '負責人',
    roomSettings: '群組聊天設定',
    purposeLabel: '用途',
    purposePlaceholder: '用一句話說明這個群組聊天的用途',
    notesLabel: '共用筆記',
    notesPlaceholder: '每位成員都應記住的事項',
    deliveryTarget: '排程工作可透過 deliver: 張貼到這裡',
    scheduledPost: '排程',
    routingLabel: '沒有點名的訊息由誰回答',
    routingEveryone: '所有人回答',
    routingLead: '負責人：{name}',
    routingEveryoneHint: '和現在一樣，每位成員都輪流回答。',
    routingLeadHint: '由負責人決定誰來處理；@提及和 @everyone 仍直接送給對應成員。',
    leadBadge: '負責人',
    assignment: '{lead} 把這則交給了 {names}',
    assignmentFallback: '{lead} 親自處理',
    routedTo: '已指派給 {names}',
    routedEveryone: '已指派給所有人',
    routedFallback: '無法決定由誰回答，所以所有人都會回答',
    pausedAfter: '在 {count} 則訊息後暫停',
    continueDiscussion: '繼續',
    continued: '已繼續',
    routingAuto: '自動選擇回答者',
    routingAutoHint: '一次快速的模型呼叫會判斷訊息是給誰的。@提及仍直接送給你點名的人。',
    statusBlocked: '已封鎖',
    statusApprovals: '{count} 項等待核准',
    statusAttention: '{count} 項需要處理',
    stopped: '已要求停止。等待確認的回合會繼續顯示；如提供重試或捨棄操作，可以使用。',
    pendingActionUnconfirmed: '無法確認此操作。請重新整理群組聊天以檢查其狀態。',
    sendRefused: '閘道拒絕了這則訊息。請編輯後再傳送。',
    sendStopped: '已在傳送前停止。你的訊息仍保留在這裡。',
    sendMaybe: '這則訊息可能已經傳送。重試會傳送同一則訊息，不會重複張貼。',
    sendSettledElsewhere: '另一個 Desktop 視窗已處理完這則訊息。再次傳送前請先查看對話。',
    journalLoadFailed: '無法載入已儲存的訊息。傳送已暫停，以免重複傳送已儲存的訊息。',
    journalReload: '重新載入',
    approvalAction: '操作',
    approvalCommand: '命令',
    approvalDetailsMissing: '無法顯示操作詳細資訊。請先重新整理詳細資訊，再允許此操作。',
    checkingDriver: '正在檢查此連線上的群組聊天…',
    legacyRoom:
      '這個群組聊天由 Desktop 自行執行，因此無法在此閘道上繼續。用相同的機器人開始新群組：關閉 Desktop 後它仍會繼續運作。先前的訊息不會複製到新群組。',
    startGatewayGroup: '開始新群組',
    createRefused: '閘道無法建立此群組聊天。',
    createGroupSize: '在此閘道上託管的群組聊天需要 2 到 6 個機器人。',
    createOtherConnection: '在此閘道上託管的群組聊天只能加入此連線上的機器人。',
    createDuplicateMember:
      '在此閘道上託管的群組聊天中的機器人需要使用不同的設定檔和代稱；@all 與 @everyone 為保留名稱。',
    createUncertain: '閘道沒有回應，此群組聊天可能已經建立。請再試一次以開啟它，不會重複建立。',
    createEnded: '閘道拒絕了此群組聊天，它可能已在其他地方被變更或移除。請再試一次以建立新的群組聊天。',
    legacyRoomElsewhere:
      '這個群組聊天無法在此繼續：它包含其他電腦上的機器人，而在此閘道上託管的群組聊天只能加入同一個連線上的機器人。請切換到這些機器人所在的連線並在那裡開始群組，或用此連線上的機器人開始新的群組聊天。先前的訊息不會複製到新群組。',
    otherConnections: '其他連線上的機器人：{connections}',
    legacyRoomBlocked:
      '這個群組聊天由 Desktop 自行執行，因此無法在此閘道上繼續，而且這些機器人目前無法組成在此閘道上託管的群組聊天（原因見下方）。請開始新的群組聊天並選擇它的機器人。先前的訊息不會複製到新群組。',
    groupsUnsupported:
      '無法在此開啟群組聊天：此連線所在電腦上的 Hermes 版本太舊，不支援群組聊天。請在那台電腦上更新 Hermes 並重新連線，然後再試一次。不會遺失任何內容：先前的訊息仍保留在原處。',
    groupDriverStopped:
      '暫時無法在此開啟群組聊天：此連線的閘道沒有執行群組聊天。請在那台電腦上重新啟動或更新 Hermes，然後再試一次。不會遺失任何內容：先前的訊息仍保留在原處。',
    groupsUnreachable:
      '無法連上此連線的閘道來開啟這個群組聊天。請確認那台電腦上的 Hermes 正在執行且你已連線，然後再試一次。不會遺失任何內容：先前的訊息仍保留在原處。',
    earlierMessages: '先前的訊息（唯讀）。它們會保留在此 Desktop 上，不會複製到新群組。',
    peerTargetSetup: '{connection} 目前還無法讓機器人加入其他電腦上的群組聊天。{reason}',
    peerSetupUpdate: '那裡的 Hermes 版本太舊。請在那台電腦上更新 Hermes，然後再檢查一次。',
    peerSetupApiServer:
      '它的 API 伺服器未開啟。請在那裡開啟 API 伺服器（gateway.platforms.api_server），重新啟動那台電腦上的 Hermes，然後再檢查一次。',
    peerSetupRunStorage:
      '它的 API 伺服器無法儲存執行記錄，訊息可能會被執行兩次。請確認 Hermes 能寫入那台電腦上的資料夾，重新啟動那裡的 Hermes，然後再檢查一次。',
    peerSetupRoomKey:
      '那裡的 Hermes 無法建立保護群組聊天存取權的金鑰。請確認 Hermes 能寫入那台電腦上的資料夾，重新啟動那裡的 Hermes，然後再檢查一次。',
    peerSetupApprovals:
      '那裡的指令核准已關閉，或無法讀取其設定。請在那台電腦上將核准設為 manual 或 smart（approvals.mode），然後再檢查一次。',
    peerSetupAddress:
      '它沒有執行此群組聊天的電腦可以連到的位址。請在那台電腦上設定 gateway.room_link_url，重新啟動那裡的 Hermes，然後再檢查一次。',
    peerSetupBackground:
      '那裡的 Hermes 會在其 Desktop 關閉時停止。請在那裡單獨執行其閘道（hermes gateway run，或將其安裝為服務），然後再檢查一次。',
    peerSetupGeneric:
      '請在那裡更新 Hermes，以執行此群組聊天的電腦可以連到的位址（gateway.room_link_url）開啟其 API 伺服器，並保持核准開啟。',
    peerJoinFailed: '{name} 尚未加入：{reason}',
    peerJoinUnconfirmed: '無法確認 {name} 是否已加入。請在能夠連線到其電腦後檢查群組聊天。',
    peerJoinUnreachable:
      '{name} 尚未加入：執行此群組聊天的電腦無法連到它所在的電腦。請確認不經 Desktop 也能從那裡連到該電腦，然後再試一次。',
    peerJoinChanged:
      '{name} 無法加入此群組聊天：自群組聊天建立以來，它所在電腦上的機器人設定已變更。若要加入它，請開始新的群組聊天。',
    readTimedOut: '閘道未及時回應。它可能正在等待某個機器人的電腦回應。請選擇「重新整理」重試。',
    disbandUnconfirmed: '無法確認群組聊天是否已經結束。某個機器人的電腦可能處於休眠或離線狀態。請在它恢復連線後重試。',
    disbandKept: '群組聊天已保留。{reason}',
    disbandPeerUnreachable:
      '群組聊天中的工作已停止。某位成員所在的電腦沒有回應。請重新連線至該電腦，然後再次嘗試解散。',
    disbandStopPending:
      '此群組聊天中的機器人尚未全部確認已停止。請在全部確認後再次嘗試解散。如果群組聊天中為等待處理的回合提供了重試或捨棄操作，可以使用這些操作。',
    disbandMemberHomeUnknown:
      '群組聊天可能仍在執行，但其中一個機器人的設定檔已不在執行群組聊天的電腦的設定中。請將該設定檔重新加入 hosted_rooms.profiles，以便確認其已停止，然後再次嘗試解散。',
    disbandInProgress: '未能完成解散。請稍候片刻，檢查群組聊天狀態，再視需要重試解散。',
    disbandRevokeUnsupported:
      '群組聊天中的工作已停止。某位成員無法撤銷其對此群組聊天的存取權。請在該成員所在的電腦上更新 Hermes，然後再次嘗試解散。',
    disbandServiceRequired:
      '必須在執行此群組聊天的電腦上啟動 Hermes，才能停止機器人並撤銷其存取權。請在那台電腦上啟動或重新連線至 Hermes，再重試解散。',
    peerStartWithoutRetry: '請在新的群組聊天中對這些機器人使用「重試」。',
    peerStartWithoutKept: '已建立新的群組聊天，原有的群組聊天也已保留。{reason} 準備好後可解散原有的群組聊天。',
    createdNotJoined: '群組聊天已建立，但 {names} 尚未加入。原因會顯示在群組聊天中，請在那裡使用「重試」。',
    peerNotJoinedBadge: '未加入',
    peerOfflineBadge: '離線',
    peerRejoinBadge: '存取權已過期',
    peerNotJoined: '{name} 尚未加入此群組聊天。',
    peerRejoin: '{name} 對此群組聊天的存取權已過期。若要再次邀請它，請在連線至其電腦的 Desktop 中選擇重試。',
    peerAway: '未能連線至 {name} 所在的電腦。下次向它傳送訊息時會再試一次。',
    peerRetryFailed: '未能重新連線至 {name}。請確認 Hermes 正在其電腦上執行，然後再試一次。',
    peerJoinConnect: '{name} 尚未加入：請連線到它所在的電腦，然後選擇「重試」。',
    peerStartWithout: '不含 {name} 開始新群組聊天',
    peerStartWithoutConfirm:
      '將用其他機器人建立新的群組聊天。只有它們全部加入，且原有群組聊天的所有成員確認停止並撤銷存取權後，原有聊天才會結束。任一步驟失敗都會保留兩個群組聊天。先前的訊息不會複製。 建立後再次使用此操作會建立另一個群組聊天。'
  },
  ar: {
    ...GROUP_REASON_LOCALES['ar'],
    roomStopped: 'طُلب الإيقاف',
    roomDisbanded: 'حُلّت المجموعة',
    authorityLost: 'تغيّر مالك المجموعة',
    workLimitReached: 'بلغت المجموعة حد العمل',
    turnFailed: 'فشلت الجولة',
    turnCancelled: 'أُلغيت الجولة',
    turnDeferred: 'أُجّلت الجولة',
    turnNoAnswer: 'لم يستجب البوت في الوقت المناسب',
    memberUnavailable: 'العضو غير متاح',
    members: 'الأعضاء',
    openGroup: 'فتح محادثة المجموعة',
    loadingGroup: 'جارٍ تحميل المجموعة…',
    loadingGroups: 'جارٍ تحميل محادثات المجموعات…',
    emptyGroups: 'لم يتم العثور على محادثات مجموعات.',
    driverUnavailable: 'محادثة المجموعة هذه غير متاحة. حدّث البوابة التي تستضيفها أو أعد الاتصال بها.',
    invalidLogCursor: 'تعذّر تحميل المزيد من سجل محادثة المجموعة هذه.',
    allowOnce: 'السماح مرة واحدة',
    deny: 'رفض',
    discardUnknown: 'تجاهل الدور',
    discardWarning:
      'ربما حدثت تغييرات بالفعل. لن يؤدي تجاهل الدور إلى التراجع عنها. بالنسبة لروبوت على جهاز آخر، ينتظر التجاهل تأكيد ذلك الجهاز أن الدور توقف أو لم يبدأ أصلًا.',
    discardStopPending: 'لم يؤكد {name} بعد أن دوره توقف. تم الاحتفاظ بالدور. حاول تجاهله مجددًا عندما يستجيب جهازه.',
    retryBusy: 'قد يكون {name} ما زال يعمل على هذا الدور. أعد المحاولة بعد انتهائه.',
    discardPeerUnreachable:
      'جهاز {name} لا يستجيب. تم الاحتفاظ بالدور. أعد الاتصال بذلك الجهاز، ثم حاول تجاهل الدور مجددًا.',
    discardPeerUpdateRequired:
      'إصدار Hermes على جهاز {name} قديم ولا يستطيع تأكيد أن هذا الدور لم يبدأ أصلًا. تم الاحتفاظ بالدور. حدّث Hermes على ذلك الجهاز، ثم حاول تجاهل الدور مجددًا.',
    confirmDiscard: 'تأكيد التجاهل',
    unconfirmedSend: 'الإرسال السابق غير مؤكّد. أعد المحاولة بالنص الأصلي قبل إرسال رسالة أخرى.',
    restoredPendingSend: 'تمت استعادة إرسال غير مؤكّد. أعد محاولته قبل إرسال رسالة أخرى.',
    groupMessage: 'رسالة المجموعة',
    attachFiles: 'إرفاق ملفات',
    removeAttachment: 'إزالة المرفق',
    uploadFailed: 'فشل الرفع',
    memberWorking: 'يعمل',
    memberQueued: 'في الانتظار',
    groupWorking: 'المجموعة تعمل…',
    statusWorking: 'يعمل',
    statusStalled: 'لا تقدّم منذ {time}',
    statusReplied: 'ردّ',
    statusPassed: 'تخطّى',
    statusFailed: 'فشل',
    statusCancelled: 'أُلغي',
    statusUnavailable: 'غير متاح',
    statusIdle: 'خامل',
    you: 'أنت',
    roundOf: 'الجولة {n} من {max}',
    round: 'الجولة {n}',
    cycleWorking: 'يعمل: {names}',
    cycleSettled: 'انتهى — لم يعد لدى أحد ما يقوله',
    cycleBounded: 'بلغت المجموعة حد الجولات',
    cycleMessageCap: 'بلغت المجموعة حد الرسائل',
    cycleStopped: 'توقّف',
    typing: '{name} يعمل…',
    typingTool: '{name} يشغّل {tool}…',
    handoff: '{from} طلب من {to}',
    replyTo: 'الرد على {name}',
    replyingTo: 'الرد على {name}',
    cancelReply: 'إلغاء الرد',
    composerPlaceholder: 'راسل المجموعة — اذكر @بوتًا لتكليفه',
    groupDetails: 'تفاصيل المجموعة',
    stopGroup: 'إيقاف المجموعة',
    pendingApproval: '{name} يطلب موافقة. القرار لك.',
    pendingRetry: 'لم ينتهِ دور {name} بعد. هل تعيد المحاولة؟',
    pendingDiscard: 'هل تريد تجاهل دور {name}؟ لن يتم التراجع عن العمل الذي أُنجز بالفعل.',
    membersFixed: 'يُحدَّد الأعضاء عند إنشاء المجموعة.',
    rowWorking: 'يعمل',
    rowFailed: 'فشل أحد الأعضاء',
    rowNeedsYou: 'بحاجة إليك',
    lead: 'القائد',
    roomSettings: 'إعدادات محادثة المجموعة',
    purposeLabel: 'الغرض',
    purposePlaceholder: 'الغرض من محادثة المجموعة هذه في سطر واحد',
    notesLabel: 'ملاحظات مشتركة',
    notesPlaceholder: 'حقائق يجب أن يتذكرها كل عضو',
    deliveryTarget: 'يمكن للمهام المجدولة النشر هنا باستخدام deliver:',
    scheduledPost: 'مجدول',
    routingLabel: 'من يجيب عن رسالة لا تذكر أحدًا',
    routingEveryone: 'يجيب الجميع',
    routingLead: 'القائد: {name}',
    routingEveryoneHint: 'يحصل كل عضو على دوره كما هو الحال الآن.',
    routingLeadHint: 'يختار القائد من يتولاها؛ الإشارات @ و@everyone تصل مباشرة إلى أصحابها.',
    leadBadge: 'القائد',
    assignment: 'أسند {lead} هذه الرسالة إلى {names}',
    assignmentFallback: 'يتولى {lead} هذه الرسالة',
    routedTo: 'أُسندت إلى {names}',
    routedEveryone: 'أُسندت إلى الجميع',
    routedFallback: 'تعذّر اختيار من يجيب، لذا سيجيب الجميع',
    pausedAfter: 'توقفت مؤقتًا بعد {count} رسالة',
    continueDiscussion: 'متابعة',
    continued: 'تمت المتابعة',
    routingAuto: 'اختيار من يجيب تلقائيًا',
    routingAutoHint: 'يختار استدعاء سريع للنموذج لمن الرسالة. الإشارات @ تصل مباشرة إلى من تسمّيهم.',
    statusBlocked: 'محظور',
    statusApprovals: '{count} بانتظار الموافقة',
    statusAttention: '{count} تحتاج إلى انتباه',
    stopped: 'طُلب الإيقاف. تبقى الأدوار التي تنتظر التأكيد ظاهرة؛ استخدم إعادة المحاولة أو التجاهل حيث يتوفر الخيار.',
    pendingActionUnconfirmed: 'لم نتمكن من تأكيد هذا الإجراء. حدّث محادثة المجموعة للتحقق من حالته.',
    sendRefused: 'رفضت البوابة هذه الرسالة. عدّلها ثم أرسلها مرة أخرى.',
    sendStopped: 'توقّف قبل الإرسال. رسالتك ما زالت هنا.',
    sendMaybe: 'ربما أُرسلت هذه الرسالة بالفعل. تعيد إعادة المحاولة إرسال الرسالة نفسها دون نشرها مرتين.',
    sendSettledElsewhere: 'انتهت نافذة Desktop أخرى من هذه الرسالة بالفعل. راجع المحادثة قبل إرسالها مرة أخرى.',
    journalLoadFailed: 'تعذّر تحميل الرسائل المحفوظة. توقف الإرسال مؤقتًا لمنع إرسال رسالة محفوظة مرتين.',
    journalReload: 'إعادة التحميل',
    approvalAction: 'الإجراء',
    approvalCommand: 'الأمر',
    approvalDetailsMissing: 'تفاصيل الإجراء غير متاحة. حدّثها قبل السماح بهذا الإجراء.',
    checkingDriver: 'جارٍ التحقق من محادثات المجموعات على هذا الاتصال…',
    legacyRoom:
      'شغّل Desktop محادثة المجموعة هذه بنفسه، لذا لا يمكنها المتابعة على هذه البوابة. ابدأ مجموعة جديدة بالروبوتات نفسها: ستستمر في العمل عند إغلاق Desktop. لن تُنسخ الرسائل السابقة إليها.',
    startGatewayGroup: 'بدء مجموعة جديدة',
    createRefused: 'تعذّر على البوابة إنشاء محادثة المجموعة هذه.',
    createGroupSize: 'تحتاج محادثة المجموعة المستضافة على هذه البوابة إلى ما بين اثنين وستة روبوتات.',
    createOtherConnection:
      'لا يمكن لمحادثة المجموعة المستضافة على هذه البوابة أن تضم إلا الروبوتات الموجودة على هذا الاتصال.',
    createDuplicateMember:
      'تحتاج محادثة المجموعة المستضافة على هذه البوابة إلى روبوتات بملفات تعريف ومعرّفات مختلفة؛ ‎@all و‎@everyone محجوزان.',
    createUncertain:
      'لم تستجب البوابة، لذا ربما تكون محادثة المجموعة هذه قد أُنشئت بالفعل. أعد المحاولة لفتحها، ولن تُنشأ مرتين.',
    createEnded:
      'رفضت البوابة محادثة المجموعة هذه، فربما تم تغييرها أو إزالتها في مكان آخر. أعد المحاولة لبدء محادثة جديدة.',
    legacyRoomElsewhere:
      'لا يمكن متابعة محادثة المجموعة هذه هنا: فيها روبوتات تعمل على أجهزة كمبيوتر أخرى، ولا تضم محادثة المجموعة المستضافة على هذه البوابة إلا روبوتات من اتصال واحد. انتقل إلى الاتصال الذي تعمل عليه تلك الروبوتات وابدأ مجموعة هناك، أو ابدأ محادثة مجموعة جديدة بروبوتات من هذا الاتصال. لن تُنسخ الرسائل السابقة إلى مجموعة جديدة.',
    otherConnections: 'روبوتات على اتصالات أخرى: {connections}',
    legacyRoomBlocked:
      'شغّل Desktop محادثة المجموعة هذه بنفسه، لذا لا يمكنها المتابعة على هذه البوابة، ولا تستطيع هذه الروبوتات بدء محادثة مجموعة مستضافة على هذه البوابة كما هي (السبب أدناه). ابدأ محادثة مجموعة جديدة واختر روبوتاتها. لن تُنسخ الرسائل السابقة إليها.',
    groupsUnsupported:
      'لا يمكن فتح محادثات المجموعات هنا: إصدار Hermes على كمبيوتر هذا الاتصال أقدم من أن يدعمها. حدّث Hermes على ذلك الكمبيوتر وأعد الاتصال، ثم حاول مرة أخرى. لن يضيع شيء: تبقى الرسائل السابقة في مكانها.',
    groupDriverStopped:
      'لا يمكن فتح محادثات المجموعات هنا الآن: بوابة هذا الاتصال لا تشغّل محادثات المجموعات. أعد تشغيل Hermes على ذلك الكمبيوتر أو حدّثه، ثم حاول مرة أخرى. لن يضيع شيء: تبقى الرسائل السابقة في مكانها.',
    groupsUnreachable:
      'تعذّر الوصول إلى بوابة هذا الاتصال لفتح محادثة المجموعة هذه. تأكد من أن Hermes يعمل هناك وأنك متصل، ثم حاول مرة أخرى. لن يضيع شيء: تبقى الرسائل السابقة في مكانها.',
    earlierMessages: 'الرسائل السابقة، للقراءة فقط. تبقى على Desktop هذا ولن تُنسخ إلى مجموعة جديدة.',
    peerTargetSetup: 'لا يستطيع {connection} بعد استضافة روبوت لمحادثة مجموعة على جهاز آخر. {reason}',
    peerSetupUpdate: 'إصدار Hermes هناك قديم جدًا. حدّث Hermes على ذلك الجهاز، ثم تحقّق مجددًا.',
    peerSetupApiServer:
      'خادم API الخاص به متوقف. فعّل خادم API هناك (gateway.platforms.api_server)، وأعد تشغيل Hermes على ذلك الجهاز، ثم تحقّق مجددًا.',
    peerSetupRunStorage:
      'لا يستطيع خادم API الخاص به حفظ سجلات التشغيل، لذا قد تُنفَّذ رسالة مرتين. تأكد من أن Hermes يستطيع الكتابة في مجلد بياناته على ذلك الجهاز، وأعد تشغيل Hermes هناك، ثم تحقّق مجددًا.',
    peerSetupRoomKey:
      'لا يستطيع Hermes هناك إنشاء المفتاح الذي يحمي الوصول إلى المحادثات الجماعية. تأكد من أن Hermes يستطيع الكتابة في مجلد بياناته على ذلك الجهاز، وأعد تشغيل Hermes هناك، ثم تحقّق مجددًا.',
    peerSetupApprovals:
      'موافقات الأوامر معطّلة هناك، أو تتعذّر قراءة إعداداته. اضبط الموافقات على manual أو smart على ذلك الجهاز (approvals.mode)، ثم تحقّق مجددًا.',
    peerSetupAddress:
      'ليس له عنوان يستطيع الجهاز الذي يشغّل محادثة المجموعة هذه الوصول إليه. اضبط gateway.room_link_url على ذلك الجهاز، وأعد تشغيل Hermes هناك، ثم تحقّق مجددًا.',
    peerSetupBackground:
      'يتوقف Hermes هناك عند إغلاق Desktop الخاص به. شغّل بوابته وحدها هناك (hermes gateway run، أو ثبّتها كخدمة)، ثم تحقّق مجددًا.',
    peerSetupGeneric:
      'حدّث Hermes هناك، وفعّل خادم API الخاص به بعنوان يستطيع الجهاز الذي يشغّل محادثة المجموعة هذه الوصول إليه (gateway.room_link_url)، وأبقِ الموافقات مفعّلة.',
    peerJoinFailed: 'لم ينضم {name} بعد: {reason}',
    peerJoinUnconfirmed: 'تعذر تأكيد انضمام {name}. تحقق من محادثة المجموعة عندما يمكن الاتصال بجهازها.',
    peerJoinUnreachable:
      'لم ينضم {name} بعد: تعذّر على الجهاز الذي يشغّل محادثة المجموعة هذه الوصول إلى جهازه. تأكد من إمكانية الوصول إلى جهازه من هناك دون Desktop، ثم حاول مرة أخرى.',
    peerJoinChanged:
      'لا يمكن لـ {name} الانضمام إلى محادثة المجموعة هذه: تغيّرت إعدادات الروبوت على جهازه منذ إنشاء محادثة المجموعة. ابدأ محادثة مجموعة جديدة لتضمينه.',
    readTimedOut:
      'لم تستجب البوابة في الوقت المحدد. ربما تنتظر استجابة جهاز أحد البوتات. اختر «تحديث» لإعادة المحاولة.',
    disbandUnconfirmed:
      'لم نتمكن من تأكيد انتهاء محادثة المجموعة. قد يكون جهاز أحد البوتات في وضع السكون أو غير متصل. أعد المحاولة بعد عودته للاتصال.',
    disbandKept: 'تم الاحتفاظ بمحادثة المجموعة. {reason}',
    disbandPeerUnreachable:
      'توقف العمل فيها. جهاز أحد الأعضاء لا يستجيب. أعد الاتصال بذلك الجهاز، ثم حاول حل محادثة المجموعة مجددًا.',
    disbandStopPending:
      'لم تؤكد جميع البوتات في محادثة المجموعة هذه أنها توقفت. حاول حل محادثة المجموعة مجددًا بعد تأكيدها جميعًا. للدور المنتظر، استخدم إعادة المحاولة أو التجاهل إذا كان الخيار متاحًا في محادثة المجموعة.',
    disbandMemberHomeUnknown:
      'قد يستمر العمل في محادثة المجموعة، لكن أحد ملفات تعريف البوتات لم يعد مضافًا إلى إعدادات الجهاز الذي يشغّل محادثة المجموعة. أعد إضافته إلى hosted_rooms.profiles حتى يمكن تأكيد توقفه، ثم حاول حل محادثة المجموعة مجددًا.',
    disbandInProgress: 'تعذّر إكمال حل محادثة المجموعة. انتظر قليلًا وتحقّق منها، ثم حاول حلها مجددًا إذا لزم الأمر.',
    disbandRevokeUnsupported:
      'توقف العمل فيها. لا يستطيع أحد الأعضاء إلغاء وصوله إلى محادثة المجموعة هذه. حدّث Hermes على جهاز ذلك العضو، ثم حاول حل محادثة المجموعة مجددًا.',
    disbandServiceRequired:
      'يجب تشغيل Hermes على الجهاز الذي يشغّل محادثة المجموعة هذه لإيقاف روبوتاتها وإلغاء وصولها. شغّل Hermes هناك أو أعد الاتصال به، ثم حاول حل المحادثة مجددًا.',
    peerStartWithoutRetry: 'استخدم إعادة المحاولة لهذه البوتات في محادثة المجموعة الجديدة.',
    peerStartWithoutKept:
      'تم إنشاء محادثة المجموعة الجديدة والاحتفاظ بمحادثة المجموعة الأصلية. {reason} يمكنك حل الدردشة الأصلية عندما تكون مستعدًا.',
    createdNotJoined:
      'تم إنشاء محادثة المجموعة، لكن {names} لم ينضم بعد. تعرض محادثة المجموعة السبب؛ استخدم إعادة المحاولة هناك.',
    peerNotJoinedBadge: 'لم ينضم',
    peerOfflineBadge: 'غير متصل',
    peerRejoinBadge: 'انتهت صلاحية الوصول',
    peerNotJoined: 'لم ينضم {name} إلى محادثة المجموعة هذه بعد.',
    peerRejoin:
      'انتهت صلاحية وصول {name} إلى محادثة المجموعة هذه. لدعوته مجددًا، اختر إعادة المحاولة من Desktop متصل بجهازه.',
    peerAway: 'تعذّر الوصول إلى جهاز {name}. ستحاول الرسالة التالية الموجّهة إليه الاتصال مجددًا.',
    peerRetryFailed: 'تعذّر إعادة الاتصال بـ {name}. تأكد من تشغيل Hermes على جهازه، ثم حاول مجددًا.',
    peerJoinConnect: 'لم ينضم {name} بعد: اتصل بالجهاز الذي يعمل عليه، ثم اختر إعادة المحاولة.',
    peerStartWithout: 'بدء محادثة مجموعة جديدة بدون {name}',
    peerStartWithoutConfirm:
      'تبدأ دردشة جماعية جديدة بالبوتات الأخرى. لا تنتهي الأصلية إلا بعد انضمامها جميعًا وتأكيد إيقاف كل عضو أصلي وإلغاء وصوله. إذا فشلت أي خطوة، تبقى الدردشتان. لن تُنسخ الرسائل السابقة. استخدام هذا الإجراء مجددًا بعد الإنشاء ينشئ دردشة جماعية أخرى.'
  },
  ru: {
    ...GROUP_REASON_LOCALES['ru'],
    roomStopped: 'Остановка запрошена',
    roomDisbanded: 'Группа распущена',
    authorityLost: 'Владелец группы изменился',
    workLimitReached: 'Достигнут лимит работы группы',
    turnFailed: 'Ход завершился ошибкой',
    turnCancelled: 'Ход отменён',
    turnDeferred: 'Ход отложен',
    turnNoAnswer: 'бот не ответил вовремя',
    memberUnavailable: 'Участник недоступен',
    members: 'Участники',
    openGroup: 'Открыть групповой чат',
    loadingGroup: 'Загрузка группы…',
    loadingGroups: 'Загрузка групповых чатов…',
    emptyGroups: 'Групповые чаты не найдены.',
    driverUnavailable:
      'Этот групповой чат недоступен. Обновите шлюз, на котором он работает, или подключитесь к нему заново.',
    invalidLogCursor: 'Не удалось загрузить следующую часть истории этого группового чата.',
    allowOnce: 'Разрешить один раз',
    deny: 'Отклонить',
    discardUnknown: 'Отбросить ход',
    discardWarning:
      'Изменения уже могли произойти. Отмена хода не отменяет их. Для бота на другом компьютере нужно дождаться подтверждения, что ход остановлен или не начинался.',
    discardStopPending:
      '{name} ещё не подтвердил остановку своего хода. Ход сохранён. Попробуйте отменить его снова, когда его компьютер будет отвечать.',
    retryBusy: '{name}, возможно, ещё выполняет этот ход. Повторите попытку, когда он завершится.',
    discardPeerUnreachable:
      'Компьютер {name} не отвечает. Ход сохранён. Восстановите подключение к этому компьютеру и попробуйте отменить ход снова.',
    discardPeerUpdateRequired:
      'Версия Hermes на компьютере {name} слишком старая, чтобы подтвердить, что этот ход не начинался. Ход сохранён. Обновите Hermes на этом компьютере и попробуйте отменить ход снова.',
    confirmDiscard: 'Подтвердить отмену работы',
    unconfirmedSend:
      'Предыдущая отправка не подтверждена. Повторите её с исходным текстом, прежде чем отправлять другое сообщение.',
    restoredPendingSend:
      'Восстановлена неподтверждённая отправка. Повторите её, прежде чем отправлять другое сообщение.',
    groupMessage: 'Сообщение группе',
    attachFiles: 'Прикрепить файлы',
    removeAttachment: 'Удалить вложение',
    uploadFailed: 'Не удалось загрузить файл',
    memberWorking: 'Работает',
    memberQueued: 'в очереди',
    groupWorking: 'Группа работает…',
    statusWorking: 'работает',
    statusStalled: 'нет прогресса {time}',
    statusReplied: 'ответил',
    statusPassed: 'пропустил',
    statusFailed: 'ошибка',
    statusCancelled: 'отменено',
    statusUnavailable: 'недоступен',
    statusIdle: 'ожидает',
    you: 'Вы',
    roundOf: 'Раунд {n} из {max}',
    round: 'Раунд {n}',
    cycleWorking: 'Работают: {names}',
    cycleSettled: 'Завершено — больше никому нечего сказать',
    cycleBounded: 'Достигнут лимит раундов',
    cycleMessageCap: 'Достигнут лимит сообщений',
    cycleStopped: 'Остановлено',
    typing: '{name} работает…',
    typingTool: '{name} выполняет {tool}…',
    handoff: '{from} попросил {to}',
    replyTo: 'Ответить {name}',
    replyingTo: 'Ответ для {name}',
    cancelReply: 'Отменить ответ',
    composerPlaceholder: 'Сообщение группе — @упомяните, чтобы поручить',
    groupDetails: 'Сведения о группе',
    stopGroup: 'Остановить группу',
    pendingApproval: '{name} просит одобрения. Решать вам.',
    pendingRetry: 'Ход {name} ещё не завершён. Повторить?',
    pendingDiscard: 'Отбросить ход {name}? Уже выполненная работа не будет отменена.',
    membersFixed: 'Участники задаются при создании группы.',
    rowWorking: 'Работает',
    rowFailed: 'Ошибка участника',
    rowNeedsYou: 'Нужны вы',
    lead: 'Ведущий',
    roomSettings: 'Настройки группового чата',
    purposeLabel: 'Назначение',
    purposePlaceholder: 'Для чего нужен этот групповой чат, одной строкой',
    notesLabel: 'Общие заметки',
    notesPlaceholder: 'Что должен помнить каждый участник',
    deliveryTarget: 'Задания по расписанию могут писать сюда через deliver:',
    scheduledPost: 'По расписанию',
    routingLabel: 'Кто отвечает на сообщение без упоминаний',
    routingEveryone: 'Отвечают все',
    routingLead: 'Ведущий: {name}',
    routingEveryoneHint: 'Каждый участник отвечает по очереди, как сейчас.',
    routingLeadHint:
      'Ведущий выбирает, кому передать сообщение; @упоминания и @everyone по-прежнему идут прямо адресатам.',
    leadBadge: 'Ведущий',
    assignment: '{lead} передал(а) это: {names}',
    assignmentFallback: '{lead} берёт это на себя',
    routedTo: 'Назначено: {names}',
    routedEveryone: 'Назначено всем',
    routedFallback: 'Не удалось выбрать, кто ответит, поэтому ответят все',
    pausedAfter: 'Пауза после сообщений: {count}',
    continueDiscussion: 'Продолжить',
    continued: 'Продолжено',
    routingAuto: 'Выбирать отвечающих автоматически',
    routingAutoHint:
      'Быстрый вызов модели определяет, кому адресовано сообщение. @упоминания по-прежнему идут прямо тем, кого вы назвали.',
    statusBlocked: 'Заблокировано',
    statusApprovals: 'Ждут подтверждения: {count}',
    statusAttention: 'Требуется внимание: {count}',
    stopped:
      'Остановка запрошена. Ходы, ожидающие подтверждения, остаются видимыми; используйте повтор или отмену, если они доступны.',
    pendingActionUnconfirmed:
      'Не удалось подтвердить это действие. Обновите групповой чат, чтобы проверить его состояние.',
    sendRefused: 'Шлюз отклонил это сообщение. Измените его и отправьте снова.',
    sendStopped: 'Остановлено до отправки. Ваше сообщение сохранено.',
    sendMaybe: 'Сообщение, возможно, уже отправлено. Повтор отправит то же сообщение, дважды оно не появится.',
    sendSettledElsewhere:
      'Другое окно Desktop уже завершило работу с этим сообщением. Проверьте беседу, прежде чем отправлять его снова.',
    journalLoadFailed:
      'Не удалось загрузить сохранённые сообщения. Отправка приостановлена, чтобы не отправить сохранённое сообщение дважды.',
    journalReload: 'Загрузить ещё раз',
    approvalAction: 'Действие',
    approvalCommand: 'Команда',
    approvalDetailsMissing: 'Подробности действия недоступны. Обновите их перед тем, как разрешить действие.',
    checkingDriver: 'Проверка групповых чатов этого подключения…',
    legacyRoom:
      'Этим групповым чатом управлял сам Desktop, поэтому он не может продолжиться на этом шлюзе. Начните новую группу с теми же ботами: она продолжит работать, когда Desktop закрыт. Прежние сообщения в неё не копируются.',
    startGatewayGroup: 'Начать новую группу',
    createRefused: 'Шлюз не смог создать этот групповой чат.',
    createGroupSize: 'В групповом чате, размещённом на этом шлюзе, должно быть от двух до шести ботов.',
    createOtherConnection: 'В групповой чат, размещённый на этом шлюзе, можно добавить только ботов этого подключения.',
    createDuplicateMember:
      'Для группового чата, размещённого на этом шлюзе, нужны боты с разными профилями и именами; @all и @everyone зарезервированы.',
    createUncertain:
      'Шлюз не ответил, поэтому этот групповой чат, возможно, уже создан. Повторите попытку, чтобы открыть его. Второй чат создан не будет.',
    createEnded:
      'Шлюз отклонил этот групповой чат: возможно, он был изменён или удалён в другом месте. Повторите попытку, чтобы создать новый.',
    legacyRoomElsewhere:
      'Этот групповой чат не может продолжиться здесь: в нём есть боты на других компьютерах, а в групповой чат, размещённый на этом шлюзе, можно добавить ботов только одного подключения. Переключитесь на подключение, где работают эти боты, и начните группу там или начните новый групповой чат с ботами этого подключения. Прежние сообщения в новую группу не копируются.',
    otherConnections: 'Боты на других подключениях: {connections}',
    legacyRoomBlocked:
      'Этим групповым чатом управлял сам Desktop, поэтому он не может продолжиться на этом шлюзе, а из этих ботов в нынешнем составе нельзя создать групповой чат, размещённый на этом шлюзе (причина ниже). Начните новый групповой чат и выберите для него ботов. Прежние сообщения в него не копируются.',
    groupsUnsupported:
      'Групповые чаты здесь не открываются: Hermes на компьютере этого подключения слишком старый для них. Обновите Hermes на том компьютере, подключитесь заново и повторите попытку. Ничего не потеряно: прежние сообщения остаются на месте.',
    groupDriverStopped:
      'Сейчас групповые чаты здесь не открываются: шлюз этого подключения не запустил групповые чаты. Перезапустите или обновите Hermes на том компьютере и повторите попытку. Ничего не потеряно: прежние сообщения остаются на месте.',
    groupsUnreachable:
      'Не удалось связаться со шлюзом этого подключения, чтобы открыть этот групповой чат. Убедитесь, что Hermes там запущен и подключение есть, и повторите попытку. Ничего не потеряно: прежние сообщения остаются на месте.',
    earlierMessages:
      'Прежние сообщения, только для чтения. Они остаются в этом Desktop и не копируются в новую группу.',
    peerTargetSetup: '{connection} пока не может предоставить бота групповому чату на другом компьютере. {reason}',
    peerSetupUpdate: 'Hermes там слишком старый. Обновите Hermes на том компьютере и проверьте снова.',
    peerSetupApiServer:
      'Его API-сервер выключен. Включите там API-сервер (gateway.platforms.api_server), перезапустите Hermes на том компьютере и проверьте снова.',
    peerSetupRunStorage:
      'Его API-сервер не может сохранять записи о запусках, поэтому сообщение может выполниться дважды. Убедитесь, что Hermes может записывать в свою папку данных на том компьютере, перезапустите там Hermes и проверьте снова.',
    peerSetupRoomKey:
      'Hermes там не может создать ключ, защищающий доступ к групповым чатам. Убедитесь, что Hermes может записывать в свою папку данных на том компьютере, перезапустите там Hermes и проверьте снова.',
    peerSetupApprovals:
      'Подтверждения команд там выключены, или его настройки не читаются. Установите подтверждения в manual или smart на том компьютере (approvals.mode) и проверьте снова.',
    peerSetupAddress:
      'У него нет адреса, доступного компьютеру, на котором работает этот групповой чат. Задайте gateway.room_link_url на том компьютере, перезапустите там Hermes и проверьте снова.',
    peerSetupBackground:
      'Hermes там останавливается, когда закрывается его Desktop. Запустите там его шлюз отдельно (hermes gateway run или установите его как службу) и проверьте снова.',
    peerSetupGeneric:
      'Обновите там Hermes, включите его API-сервер с адресом, доступным компьютеру, на котором работает этот групповой чат (gateway.room_link_url), и оставьте подтверждения включёнными.',
    peerJoinFailed: '{name} ещё не присоединился: {reason}',
    peerJoinUnconfirmed:
      'Не удалось подтвердить, присоединился ли {name}. Проверьте групповой чат, когда его компьютер станет доступен.',
    peerJoinUnreachable:
      '{name} ещё не присоединился: компьютер, на котором работает этот групповой чат, не смог связаться с его компьютером. Убедитесь, что этот компьютер доступен оттуда без Desktop, и попробуйте снова.',
    peerJoinChanged:
      '{name} не может присоединиться к этому групповому чату: настройки бота на его компьютере изменились после создания группового чата. Чтобы добавить его, начните новый групповой чат.',
    readTimedOut:
      'Шлюз не ответил вовремя. Возможно, он ждёт, пока ответит компьютер бота. Нажмите «Обновить», чтобы повторить попытку.',
    disbandUnconfirmed:
      'Не удалось подтвердить завершение группового чата. Компьютер одного из ботов может быть в спящем режиме или не в сети. Повторите попытку, когда он снова будет доступен.',
    disbandKept: 'Групповой чат сохранён. {reason}',
    disbandPeerUnreachable:
      'Работа в нём остановлена. Компьютер одного из участников не отвечает. Восстановите подключение к этому компьютеру и попробуйте распустить групповой чат снова.',
    disbandStopPending:
      'Не все боты в этом групповом чате подтвердили остановку. Повторите роспуск группового чата, когда все подтвердят её. Для ожидающего хода используйте повтор или отмену, если эти действия доступны в групповом чате.',
    disbandMemberHomeUnknown:
      'Работа в групповом чате может продолжаться, но один из профилей ботов больше не настроен на компьютере, на котором работает групповой чат. Верните этот профиль в hosted_rooms.profiles, чтобы подтвердить его остановку, затем повторите роспуск группового чата.',
    disbandInProgress:
      'Не удалось завершить роспуск. Немного подождите, проверьте групповой чат и при необходимости повторите роспуск.',
    disbandRevokeUnsupported:
      'Работа в нём остановлена. Один из участников не может отозвать свой доступ к этому групповому чату. Обновите Hermes на компьютере этого участника и попробуйте распустить групповой чат снова.',
    disbandServiceRequired:
      'Чтобы остановить ботов и отозвать их доступ, Hermes должен работать на компьютере, который ведёт этот групповой чат. Запустите Hermes там или восстановите подключение к нему, затем повторите роспуск.',
    peerStartWithoutRetry: 'Нажмите «Повторить» для этих ботов в новом групповом чате.',
    peerStartWithoutKept:
      'Новый групповой чат создан, а исходный групповой чат сохранён. {reason} Когда будете готовы, распустите исходный групповой чат.',
    createdNotJoined:
      'Групповой чат создан, но {names} ещё не присоединился. Причина показана в групповом чате; нажмите там «Повторить».',
    peerNotJoinedBadge: 'не присоединился',
    peerOfflineBadge: 'не в сети',
    peerRejoinBadge: 'доступ истёк',
    peerNotJoined: '{name} ещё не присоединился к этому групповому чату.',
    peerRejoin:
      'Доступ {name} к этому групповому чату истёк. Чтобы пригласить его снова, нажмите «Повторить» в Desktop, подключённом к его компьютеру.',
    peerAway: 'Не удалось связаться с компьютером {name}. Следующее сообщение ему повторит попытку подключения.',
    peerRetryFailed:
      'Не удалось восстановить подключение к {name}. Убедитесь, что на его компьютере работает Hermes, и попробуйте снова.',
    peerJoinConnect:
      '{name} ещё не присоединился: подключитесь к компьютеру, на котором он работает, и выберите «Повторить».',
    peerStartWithout: 'Начать новый групповой чат без {name}',
    peerStartWithoutConfirm:
      'Создаётся новый групповой чат с остальными ботами. Исходный завершится только после их присоединения и подтверждения остановки и отзыва доступа каждого исходного участника. При сбое любого шага оба чата сохраняются. Прежние сообщения не копируются. Повторное действие после создания создаст ещё один групповой чат.'
  },
  fr: {
    ...GROUP_REASON_LOCALES['fr'],
    roomStopped: 'Arrêt demandé',
    roomDisbanded: 'Groupe dissous',
    authorityLost: 'Gestionnaire du groupe modifié',
    workLimitReached: 'Limite de travail du groupe atteinte',
    turnFailed: 'Tour échoué',
    turnCancelled: 'Tour annulé',
    turnDeferred: 'Tour différé',
    turnNoAnswer: 'le bot n’a pas répondu à temps',
    memberUnavailable: 'Membre indisponible',
    members: 'Membres',
    openGroup: 'Ouvrir la discussion de groupe',
    loadingGroup: 'Chargement du groupe…',
    loadingGroups: 'Chargement des discussions de groupe…',
    emptyGroups: 'Aucune discussion de groupe trouvée.',
    driverUnavailable:
      'Cette discussion de groupe est indisponible. Mettez à jour ou reconnectez la passerelle qui l’héberge.',
    invalidLogCursor: 'Impossible de charger la suite de l’historique de cette discussion de groupe.',
    allowOnce: 'Autoriser une fois',
    deny: 'Refuser',
    discardUnknown: 'Abandonner le tour',
    discardWarning:
      'Des changements ont peut-être déjà eu lieu. Abandonner le tour ne les annule pas. Pour un bot sur un autre ordinateur, il faut attendre que cet ordinateur confirme que le tour est arrêté ou n’a jamais démarré.',
    discardStopPending:
      '{name} n’a pas encore confirmé l’arrêt de son tour. Le tour a été conservé. Réessayez de l’abandonner lorsque son ordinateur répondra.',
    retryBusy: '{name} travaille peut-être encore sur ce tour. Réessayez une fois qu’il est terminé.',
    discardPeerUnreachable:
      'L’ordinateur de {name} ne répond pas. Le tour a été conservé. Reconnectez cet ordinateur, puis réessayez d’abandonner le tour.',
    discardPeerUpdateRequired:
      'Hermes sur l’ordinateur de {name} est trop ancien pour confirmer que ce tour n’a jamais démarré. Le tour a été conservé. Mettez Hermes à jour sur cet ordinateur, puis réessayez d’abandonner le tour.',
    confirmDiscard: "Confirmer l'abandon",
    unconfirmedSend:
      'L’envoi précédent n’est pas confirmé. Réessayez d’envoyer le message d’origine avant d’en envoyer un autre.',
    restoredPendingSend: "Un envoi non confirmé a été restauré. Réessayez-le avant d'envoyer un autre message.",
    groupMessage: 'Message de groupe',
    attachFiles: 'Joindre des fichiers',
    removeAttachment: 'Retirer la pièce jointe',
    uploadFailed: 'Échec du téléversement',
    memberWorking: 'En cours',
    memberQueued: 'en attente',
    groupWorking: 'Le groupe travaille…',
    statusWorking: 'au travail',
    statusStalled: 'aucun progrès depuis {time}',
    statusReplied: 'a répondu',
    statusPassed: 'a passé',
    statusFailed: 'échec',
    statusCancelled: 'annulé',
    statusUnavailable: 'indisponible',
    statusIdle: 'inactif',
    you: 'Vous',
    roundOf: 'Tour {n} sur {max}',
    round: 'Tour {n}',
    cycleWorking: 'Au travail : {names}',
    cycleSettled: 'Terminé — personne n’avait plus rien à dire',
    cycleBounded: 'Limite de tours atteinte',
    cycleMessageCap: 'Limite de messages atteinte',
    cycleStopped: 'Arrêté',
    typing: '{name} travaille…',
    typingTool: '{name} exécute {tool}…',
    handoff: '{from} a sollicité {to}',
    replyTo: 'Répondre à {name}',
    replyingTo: 'Réponse à {name}',
    cancelReply: 'Annuler la réponse',
    composerPlaceholder: 'Message au groupe — @mentionnez pour assigner',
    groupDetails: 'Détails du groupe',
    stopGroup: 'Arrêter le groupe',
    pendingApproval: '{name} demande une approbation. À vous de décider.',
    pendingRetry: 'Le tour de {name} n’est pas terminé. Réessayer ?',
    pendingDiscard: 'Abandonner le tour de {name} ? Le travail déjà effectué ne sera pas annulé.',
    membersFixed: 'Les membres sont fixés à la création du groupe.',
    rowWorking: 'Au travail',
    rowFailed: 'Un membre a échoué',
    rowNeedsYou: 'Vous êtes attendu',
    lead: 'Responsable',
    roomSettings: 'Paramètres de la discussion de groupe',
    purposeLabel: 'Objectif',
    purposePlaceholder: 'À quoi sert cette discussion de groupe, en une ligne',
    notesLabel: 'Notes partagées',
    notesPlaceholder: 'Ce que chaque membre doit garder en tête',
    deliveryTarget: 'Les tâches planifiées peuvent publier ici avec deliver:',
    scheduledPost: 'Planifié',
    routingLabel: 'Qui répond à un message qui ne nomme personne',
    routingEveryone: 'Tout le monde répond',
    routingLead: 'Responsable : {name}',
    routingEveryoneHint: "Chaque membre a son tour, comme aujourd'hui.",
    routingLeadHint:
      "Le responsable choisit qui s'en charge ; les @mentions et @everyone vont toujours directement aux membres visés.",
    leadBadge: 'Responsable',
    assignment: '{lead} a confié ceci à {names}',
    assignmentFallback: "{lead} s'en charge",
    routedTo: 'Attribué à {names}',
    routedEveryone: 'Attribué à tout le monde',
    routedFallback: 'Impossible de choisir qui répond, donc tout le monde répondra',
    pausedAfter: 'En pause après {count} messages',
    continueDiscussion: 'Continuer',
    continued: 'Reprise',
    routingAuto: 'Choisir automatiquement qui répond',
    routingAutoHint:
      "Un appel rapide au modèle choisit à qui s'adresse un message. Les @mentions vont toujours directement aux personnes nommées.",
    statusBlocked: 'Bloqué',
    statusApprovals: '{count} en attente d’approbation',
    statusAttention: '{count} à traiter',
    stopped:
      'Arrêt demandé. Les tours en attente de confirmation restent visibles ; utilisez Réessayer ou Ignorer si ces actions sont proposées.',
    pendingActionUnconfirmed:
      'Cette action n’a pas pu être confirmée. Actualisez la discussion pour vérifier son état.',
    sendRefused: 'La passerelle a refusé ce message. Modifiez-le puis renvoyez-le.',
    sendStopped: 'Arrêté avant l’envoi. Votre message est toujours là.',
    sendMaybe: 'Ce message a peut-être déjà été envoyé. Réessayer renvoie le même message, sans le publier deux fois.',
    sendSettledElsewhere:
      'Une autre fenêtre Desktop a déjà traité ce message. Vérifiez la conversation avant de le renvoyer.',
    journalLoadFailed:
      'Impossible de charger les messages enregistrés. L’envoi est suspendu pour éviter d’envoyer deux fois un message enregistré.',
    journalReload: 'Réessayer le chargement',
    approvalAction: 'Opération',
    approvalCommand: 'Commande',
    approvalDetailsMissing:
      'Les détails de l’action sont indisponibles. Actualisez-les avant d’autoriser cette action.',
    checkingDriver: 'Vérification des discussions de groupe sur cette connexion…',
    legacyRoom:
      'Desktop gérait lui-même cette discussion de groupe, elle ne peut donc pas continuer sur cette passerelle. Démarrez un nouveau groupe avec les mêmes bots : il continue de fonctionner quand Desktop est fermé. Les messages précédents n’y sont pas copiés.',
    startGatewayGroup: 'Démarrer un nouveau groupe',
    createRefused: 'La passerelle n’a pas pu créer cette discussion de groupe.',
    createGroupSize: 'Une discussion de groupe hébergée sur cette passerelle nécessite de deux à six bots.',
    createOtherConnection:
      'Une discussion de groupe hébergée sur cette passerelle ne peut accueillir que des bots de cette connexion.',
    createDuplicateMember:
      'Une discussion de groupe hébergée sur cette passerelle nécessite des bots aux profils et identifiants distincts ; @all et @everyone sont réservés.',
    createUncertain:
      'La passerelle n’a pas répondu : cette discussion de groupe existe peut-être déjà. Réessayez pour l’ouvrir. Elle ne sera pas créée deux fois.',
    createEnded:
      'La passerelle a refusé cette discussion de groupe : elle a peut-être été modifiée ou supprimée ailleurs. Réessayez pour en créer une nouvelle.',
    legacyRoomElsewhere:
      'Cette discussion de groupe ne peut pas continuer ici : elle comprend des bots sur d’autres ordinateurs, et une discussion de groupe hébergée sur cette passerelle n’accueille que les bots d’une seule connexion. Passez à la connexion où ces bots fonctionnent et démarrez-y un groupe, ou démarrez une nouvelle discussion de groupe avec des bots de cette connexion. Les messages précédents ne sont pas copiés dans un nouveau groupe.',
    otherConnections: 'Bots sur d’autres connexions : {connections}',
    legacyRoomBlocked:
      'Desktop gérait lui-même cette discussion de groupe, elle ne peut donc pas continuer sur cette passerelle, et ces bots ne peuvent pas former une discussion de groupe hébergée sur cette passerelle tels quels (la raison figure ci-dessous). Démarrez une nouvelle discussion de groupe et choisissez ses bots. Les messages précédents n’y sont pas copiés.',
    groupsUnsupported:
      'Les discussions de groupe ne peuvent pas s’ouvrir ici : la version de Hermes sur l’ordinateur de cette connexion est trop ancienne. Mettez Hermes à jour sur cet ordinateur et reconnectez-vous, puis réessayez. Rien n’est perdu : les messages précédents restent où ils sont.',
    groupDriverStopped:
      'Les discussions de groupe ne peuvent pas s’ouvrir ici pour le moment : la passerelle de cette connexion n’exécute pas les discussions de groupe. Redémarrez ou mettez à jour Hermes sur cet ordinateur, puis réessayez. Rien n’est perdu : les messages précédents restent où ils sont.',
    groupsUnreachable:
      'Impossible de joindre la passerelle de cette connexion pour ouvrir cette discussion de groupe. Vérifiez que Hermes y fonctionne et que vous êtes connecté, puis réessayez. Rien n’est perdu : les messages précédents restent où ils sont.',
    earlierMessages:
      'Messages précédents, en lecture seule. Ils restent sur ce Desktop et ne sont pas copiés dans un nouveau groupe.',
    peerTargetSetup:
      '{connection} ne peut pas encore accueillir de bot pour une discussion de groupe sur un autre ordinateur. {reason}',
    peerSetupUpdate:
      'La version de Hermes sur cet ordinateur est trop ancienne. Mettez Hermes à jour sur cet ordinateur, puis vérifiez de nouveau.',
    peerSetupApiServer:
      'Son serveur API est désactivé. Activez-y le serveur API (gateway.platforms.api_server), redémarrez Hermes sur cet ordinateur, puis vérifiez de nouveau.',
    peerSetupRunStorage:
      'Son serveur API ne peut pas enregistrer ses exécutions : un message pourrait s’exécuter deux fois. Vérifiez que Hermes peut écrire dans son dossier de données sur cet ordinateur, redémarrez-y Hermes, puis vérifiez de nouveau.',
    peerSetupRoomKey:
      'Hermes ne peut pas y créer la clé qui protège l’accès aux discussions de groupe. Vérifiez que Hermes peut écrire dans son dossier de données sur cet ordinateur, redémarrez-y Hermes, puis vérifiez de nouveau.',
    peerSetupApprovals:
      'Les approbations de commandes y sont désactivées, ou ses réglages sont illisibles. Réglez les approbations sur manual ou smart sur cet ordinateur (approvals.mode), puis vérifiez de nouveau.',
    peerSetupAddress:
      'Il n’a aucune adresse que l’ordinateur qui gère cette discussion de groupe peut joindre. Définissez gateway.room_link_url sur cet ordinateur, redémarrez-y Hermes, puis vérifiez de nouveau.',
    peerSetupBackground:
      'Hermes s’y arrête quand son Desktop se ferme. Lancez-y sa passerelle seule (hermes gateway run, ou installez-la comme service), puis vérifiez de nouveau.',
    peerSetupGeneric:
      'Mettez Hermes à jour là-bas, activez son serveur API avec une adresse que l’ordinateur qui gère cette discussion de groupe peut joindre (gateway.room_link_url), et laissez les approbations activées.',
    peerJoinFailed: '{name} n’a pas encore rejoint la discussion de groupe : {reason}',
    peerJoinUnconfirmed:
      'Impossible de confirmer si {name} a rejoint la discussion de groupe. Vérifiez la discussion lorsque son ordinateur sera joignable.',
    peerJoinUnreachable:
      '{name} n’a pas encore rejoint la discussion de groupe : l’ordinateur qui gère cette discussion de groupe n’a pas pu joindre son ordinateur. Vérifiez qu’on peut le joindre depuis là-bas sans Desktop, puis réessayez.',
    peerJoinChanged:
      '{name} ne peut pas rejoindre cette discussion de groupe : les réglages du bot sur son ordinateur ont changé depuis la création de la discussion de groupe. Démarrez une nouvelle discussion de groupe pour l’inclure.',
    readTimedOut:
      'La passerelle n’a pas répondu à temps. Elle attend peut-être l’ordinateur d’un bot. Choisissez Actualiser pour réessayer.',
    disbandUnconfirmed:
      'La fin de la discussion de groupe n’a pas pu être confirmée. L’ordinateur d’un bot est peut-être en veille ou hors ligne. Réessayez lorsqu’il sera de nouveau disponible.',
    disbandKept: 'La discussion de groupe a été conservée. {reason}',
    disbandPeerUnreachable:
      'Le travail a été arrêté. L’ordinateur d’un membre ne répond pas. Reconnectez cet ordinateur, puis réessayez de dissoudre la discussion de groupe.',
    disbandStopPending:
      'Les bots de cette discussion de groupe n’ont pas tous confirmé leur arrêt. Réessayez de dissoudre la discussion une fois que tous l’auront confirmé. Pour un tour en attente, utilisez Réessayer ou Ignorer si ces actions sont proposées dans la discussion de groupe.',
    disbandMemberHomeUnknown:
      'La discussion de groupe peut encore être en cours, mais un profil de bot n’est plus configuré sur l’ordinateur qui exécute la discussion de groupe. Ajoutez à nouveau ce profil à hosted_rooms.profiles pour pouvoir confirmer son arrêt, puis réessayez de dissoudre la discussion de groupe.',
    disbandInProgress:
      'La dissolution n’a pas pu se terminer. Patientez un instant, vérifiez la discussion de groupe, puis réessayez si nécessaire.',
    disbandRevokeUnsupported:
      'Le travail a été arrêté. Un membre ne peut pas révoquer son accès à cette discussion de groupe. Mettez Hermes à jour sur son ordinateur, puis réessayez de dissoudre la discussion de groupe.',
    disbandServiceRequired:
      'Hermes doit fonctionner sur l’ordinateur qui exécute cette discussion de groupe pour arrêter ses bots et révoquer leurs accès. Démarrez ou reconnectez Hermes sur cet ordinateur, puis réessayez de dissoudre la discussion de groupe.',
    peerStartWithoutRetry: 'Utilisez Réessayer pour ces bots dans la nouvelle discussion de groupe.',
    peerStartWithoutKept:
      'La nouvelle discussion de groupe a été créée et celle d’origine a été conservée. {reason} Vous pourrez dissoudre la discussion d’origine quand vous serez prêt.',
    createdNotJoined:
      'La discussion de groupe a été créée, mais {names} ne l’a pas encore rejointe. La discussion de groupe indique pourquoi ; utilisez Réessayer depuis celle-ci.',
    peerNotJoinedBadge: 'pas encore là',
    peerOfflineBadge: 'hors ligne',
    peerRejoinBadge: 'accès expiré',
    peerNotJoined: '{name} n’a pas encore rejoint cette discussion de groupe.',
    peerRejoin:
      'L’accès de {name} à cette discussion de groupe a expiré. Pour l’inviter à nouveau, choisissez Réessayer depuis un Desktop connecté à son ordinateur.',
    peerAway: 'Impossible de joindre l’ordinateur de {name}. Le prochain message qui lui sera adressé réessaiera.',
    peerRetryFailed:
      'Impossible de reconnecter {name}. Vérifiez que Hermes fonctionne sur son ordinateur, puis réessayez.',
    peerJoinConnect:
      '{name} n’a pas encore rejoint la discussion de groupe : connectez-vous à l’ordinateur où il fonctionne, puis choisissez Réessayer.',
    peerStartWithout: 'Démarrer une nouvelle discussion de groupe sans {name}',
    peerStartWithoutConfirm:
      'Une nouvelle discussion de groupe est créée avec les autres bots. L’originale ne se termine qu’après leur arrivée et la confirmation de l’arrêt et de la révocation des accès de chaque membre d’origine. Si une étape échoue, les deux discussions sont conservées. Les messages précédents ne sont pas copiés. Réutiliser cette action après la création crée une autre discussion de groupe.'
  },
  de: {
    ...GROUP_REASON_LOCALES['de'],
    roomStopped: 'Stopp angefordert',
    roomDisbanded: 'Gruppe aufgelöst',
    authorityLost: 'Gruppenverwaltung gewechselt',
    workLimitReached: 'Arbeitslimit der Gruppe erreicht',
    turnFailed: 'Zug fehlgeschlagen',
    turnCancelled: 'Zug abgebrochen',
    turnDeferred: 'Zug verschoben',
    turnNoAnswer: 'der Bot hat nicht rechtzeitig geantwortet',
    memberUnavailable: 'Mitglied nicht verfügbar',
    members: 'Mitglieder',
    openGroup: 'Gruppenchat öffnen',
    loadingGroup: 'Gruppe wird geladen…',
    loadingGroups: 'Gruppenchats werden geladen…',
    emptyGroups: 'Keine Gruppenchats gefunden.',
    driverUnavailable:
      'Dieser Gruppenchat ist nicht verfügbar. Aktualisieren Sie das Gateway, auf dem er läuft, oder verbinden Sie es neu.',
    invalidLogCursor: 'Der weitere Verlauf dieses Gruppenchats konnte nicht geladen werden.',
    allowOnce: 'Einmal erlauben',
    deny: 'Ablehnen',
    discardUnknown: 'Turn verwerfen',
    discardWarning:
      'Es können bereits Änderungen erfolgt sein. Verwerfen macht sie nicht rückgängig. Bei einem Bot auf einem anderen Computer wird gewartet, bis dieser Computer bestätigt, dass der Turn gestoppt wurde oder nie begonnen hat.',
    discardStopPending:
      '{name} hat noch nicht bestätigt, dass der Turn gestoppt wurde. Der Turn bleibt erhalten. Versuchen Sie erneut, ihn zu verwerfen, sobald sein Computer antwortet.',
    retryBusy: '{name} arbeitet möglicherweise noch an diesem Zug. Versuchen Sie es erneut, sobald er beendet ist.',
    discardPeerUnreachable:
      'Der Computer von {name} antwortet nicht. Der Turn bleibt erhalten. Stellen Sie die Verbindung zu diesem Computer wieder her und versuchen Sie dann erneut, den Turn zu verwerfen.',
    discardPeerUpdateRequired:
      'Hermes auf dem Computer von {name} ist zu alt, um zu bestätigen, dass dieser Turn nie begonnen hat. Der Turn bleibt erhalten. Aktualisieren Sie Hermes auf diesem Computer und versuchen Sie dann erneut, den Turn zu verwerfen.',
    confirmDiscard: 'Verwerfen bestätigen',
    unconfirmedSend:
      'Der vorherige Versand wurde noch nicht bestätigt. Versuchen Sie erneut, die ursprüngliche Nachricht zu senden, bevor Sie eine weitere Nachricht senden.',
    restoredPendingSend:
      'Ein unbestätigter Versand wurde wiederhergestellt. Wiederholen Sie ihn, bevor Sie eine weitere Nachricht senden.',
    groupMessage: 'Gruppennachricht',
    attachFiles: 'Dateien anhängen',
    removeAttachment: 'Anhang entfernen',
    uploadFailed: 'Hochladen fehlgeschlagen',
    memberWorking: 'Arbeitet',
    memberQueued: 'in der Warteschlange',
    groupWorking: 'Die Gruppe arbeitet…',
    statusWorking: 'arbeitet',
    statusStalled: 'kein Fortschritt seit {time}',
    statusReplied: 'hat geantwortet',
    statusPassed: 'hat gepasst',
    statusFailed: 'fehlgeschlagen',
    statusCancelled: 'abgebrochen',
    statusUnavailable: 'nicht verfügbar',
    statusIdle: 'untätig',
    you: 'Du',
    roundOf: 'Runde {n} von {max}',
    round: 'Runde {n}',
    cycleWorking: 'Arbeitet: {names}',
    cycleSettled: 'Beendet — niemand hatte mehr zu sagen',
    cycleBounded: 'Rundenlimit erreicht',
    cycleMessageCap: 'Nachrichtenlimit erreicht',
    cycleStopped: 'Gestoppt',
    typing: '{name} arbeitet…',
    typingTool: '{name} führt {tool} aus…',
    handoff: '{from} hat {to} gefragt',
    replyTo: '{name} antworten',
    replyingTo: 'Antwort an {name}',
    cancelReply: 'Antwort abbrechen',
    composerPlaceholder: 'Nachricht an die Gruppe — @erwähnen, um zuzuweisen',
    groupDetails: 'Gruppendetails',
    stopGroup: 'Gruppe stoppen',
    pendingApproval: '{name} bittet um Freigabe. Du entscheidest.',
    pendingRetry: 'Der Turn von {name} ist noch nicht abgeschlossen. Erneut versuchen?',
    pendingDiscard: 'Den Turn von {name} verwerfen? Bereits ausgeführte Arbeit wird nicht rückgängig gemacht.',
    membersFixed: 'Mitglieder werden beim Erstellen der Gruppe festgelegt.',
    rowWorking: 'Arbeitet',
    rowFailed: 'Ein Mitglied ist fehlgeschlagen',
    rowNeedsYou: 'Braucht dich',
    lead: 'Leitung',
    roomSettings: 'Gruppenchat-Einstellungen',
    purposeLabel: 'Zweck',
    purposePlaceholder: 'Wofür dieser Gruppenchat da ist, in einer Zeile',
    notesLabel: 'Gemeinsame Notizen',
    notesPlaceholder: 'Was jedes Mitglied im Blick behalten soll',
    deliveryTarget: 'Geplante Aufgaben können hier mit deliver: posten',
    scheduledPost: 'Geplant',
    routingLabel: 'Wer antwortet auf eine Nachricht ohne Erwähnung',
    routingEveryone: 'Alle antworten',
    routingLead: 'Leitung: {name}',
    routingEveryoneHint: 'Jedes Mitglied ist an der Reihe, wie bisher.',
    routingLeadHint:
      'Die Leitung wählt, wer übernimmt; @Erwähnungen und @everyone gehen weiterhin direkt an diese Mitglieder.',
    leadBadge: 'Leitung',
    assignment: '{lead} hat das an {names} gegeben',
    assignmentFallback: '{lead} übernimmt das',
    routedTo: 'Zugewiesen an {names}',
    routedEveryone: 'Allen zugewiesen',
    routedFallback: 'Konnte nicht wählen, wer antwortet, daher antworten alle',
    pausedAfter: 'Pausiert nach {count} Nachrichten',
    continueDiscussion: 'Fortsetzen',
    continued: 'Fortgesetzt',
    routingAuto: 'Automatisch wählen, wer antwortet',
    routingAutoHint:
      'Ein kurzer Modellaufruf wählt, an wen eine Nachricht gerichtet ist. @Erwähnungen gehen weiterhin direkt an die genannten Personen.',
    statusBlocked: 'Blockiert',
    statusApprovals: 'Freigabe ausstehend: {count}',
    statusAttention: 'Aufmerksamkeit erforderlich: {count}',
    stopped:
      'Stopp angefordert. Turns, deren Bestätigung aussteht, bleiben sichtbar; nutzen Sie „Erneut versuchen“ oder „Verwerfen“, wenn angeboten.',
    pendingActionUnconfirmed:
      'Diese Aktion konnte nicht bestätigt werden. Aktualisieren Sie den Gruppenchat, um ihren Status zu prüfen.',
    sendRefused: 'Das Gateway hat diese Nachricht abgelehnt. Bearbeiten Sie sie und senden Sie sie erneut.',
    sendStopped: 'Vor dem Senden gestoppt. Ihre Nachricht ist noch da.',
    sendMaybe:
      'Diese Nachricht wurde möglicherweise schon gesendet. Erneut versuchen sendet dieselbe Nachricht, ohne sie doppelt zu posten.',
    sendSettledElsewhere:
      'Ein anderes Desktop-Fenster hat diese Nachricht bereits abgeschlossen. Prüfen Sie die Unterhaltung, bevor Sie sie erneut senden.',
    journalLoadFailed:
      'Gespeicherte Nachrichten konnten nicht geladen werden. Das Senden ist pausiert, damit eine gespeicherte Nachricht nicht doppelt gesendet wird.',
    journalReload: 'Erneut laden',
    approvalAction: 'Aktion',
    approvalCommand: 'Befehl',
    approvalDetailsMissing:
      'Die Aktionsdetails sind nicht verfügbar. Aktualisieren Sie sie, bevor Sie diese Aktion erlauben.',
    checkingDriver: 'Gruppenchats dieser Verbindung werden geprüft…',
    legacyRoom:
      'Desktop hat diesen Gruppenchat selbst ausgeführt, daher kann er auf diesem Gateway nicht weiterlaufen. Starten Sie eine neue Gruppe mit denselben Bots: Sie arbeitet weiter, wenn Desktop geschlossen ist. Frühere Nachrichten werden nicht in sie kopiert.',
    startGatewayGroup: 'Neue Gruppe starten',
    createRefused: 'Das Gateway konnte diesen Gruppenchat nicht erstellen.',
    createGroupSize: 'Ein auf diesem Gateway gehosteter Gruppenchat braucht zwei bis sechs Bots.',
    createOtherConnection: 'Ein auf diesem Gateway gehosteter Gruppenchat kann nur Bots dieser Verbindung aufnehmen.',
    createDuplicateMember:
      'Ein auf diesem Gateway gehosteter Gruppenchat braucht Bots mit unterschiedlichen Profilen und Handles; @all und @everyone sind reserviert.',
    createUncertain:
      'Das Gateway hat nicht geantwortet, daher gibt es diesen Gruppenchat möglicherweise schon. Versuchen Sie es erneut, um ihn zu öffnen. Er wird nicht doppelt erstellt.',
    createEnded:
      'Das Gateway hat diesen Gruppenchat abgelehnt. Er wurde möglicherweise an anderer Stelle geändert oder entfernt. Versuchen Sie es erneut, um einen neuen zu erstellen.',
    legacyRoomElsewhere:
      'Dieser Gruppenchat kann hier nicht weiterlaufen: Er enthält Bots auf anderen Computern, und ein auf diesem Gateway gehosteter Gruppenchat nimmt nur Bots einer einzigen Verbindung auf. Wechseln Sie zu der Verbindung, auf der diese Bots laufen, und starten Sie dort eine Gruppe, oder starten Sie einen neuen Gruppenchat mit Bots dieser Verbindung. Frühere Nachrichten werden nicht in eine neue Gruppe kopiert.',
    otherConnections: 'Bots auf anderen Verbindungen: {connections}',
    legacyRoomBlocked:
      'Desktop hat diesen Gruppenchat selbst ausgeführt, daher kann er auf diesem Gateway nicht weiterlaufen, und diese Bots können so keinen auf diesem Gateway gehosteten Gruppenchat bilden (den Grund finden Sie unten). Starten Sie einen neuen Gruppenchat und wählen Sie seine Bots. Frühere Nachrichten werden nicht in ihn kopiert.',
    groupsUnsupported:
      'Gruppenchats lassen sich hier nicht öffnen: Hermes auf dem Computer dieser Verbindung ist dafür zu alt. Aktualisieren Sie Hermes auf diesem Computer, verbinden Sie sich neu und versuchen Sie es dann erneut. Nichts geht verloren: Frühere Nachrichten bleiben, wo sie sind.',
    groupDriverStopped:
      'Gruppenchats lassen sich hier gerade nicht öffnen: Das Gateway dieser Verbindung führt keine Gruppenchats aus. Starten Sie Hermes auf diesem Computer neu oder aktualisieren Sie es, und versuchen Sie es dann erneut. Nichts geht verloren: Frühere Nachrichten bleiben, wo sie sind.',
    groupsUnreachable:
      'Das Gateway dieser Verbindung war nicht erreichbar, um diesen Gruppenchat zu öffnen. Prüfen Sie, ob Hermes dort läuft und Sie verbunden sind, und versuchen Sie es dann erneut. Nichts geht verloren: Frühere Nachrichten bleiben, wo sie sind.',
    earlierMessages:
      'Frühere Nachrichten, nur lesbar. Sie bleiben auf diesem Desktop und werden nicht in eine neue Gruppe kopiert.',
    peerTargetSetup:
      '{connection} kann noch keinen Bot für einen Gruppenchat auf einem anderen Computer bereitstellen. {reason}',
    peerSetupUpdate:
      'Hermes ist dort zu alt dafür. Aktualisieren Sie Hermes auf diesem Computer und prüfen Sie es dann erneut.',
    peerSetupApiServer:
      'Sein API-Server ist ausgeschaltet. Schalten Sie dort den API-Server ein (gateway.platforms.api_server), starten Sie Hermes auf diesem Computer neu und prüfen Sie es dann erneut.',
    peerSetupRunStorage:
      'Sein API-Server kann keine Ausführungsprotokolle speichern, daher könnte eine Nachricht doppelt ausgeführt werden. Stellen Sie sicher, dass Hermes auf diesem Computer in seinen Datenordner schreiben kann, starten Sie Hermes dort neu und prüfen Sie es dann erneut.',
    peerSetupRoomKey:
      'Hermes kann dort den Schlüssel nicht erstellen, der den Zugriff auf Gruppenchats schützt. Stellen Sie sicher, dass Hermes auf diesem Computer in seinen Datenordner schreiben kann, starten Sie Hermes dort neu und prüfen Sie es dann erneut.',
    peerSetupApprovals:
      'Befehlsfreigaben sind dort ausgeschaltet, oder seine Einstellungen lassen sich nicht lesen. Setzen Sie die Freigaben auf diesem Computer auf manual oder smart (approvals.mode) und prüfen Sie es dann erneut.',
    peerSetupAddress:
      'Dort ist keine Adresse eingerichtet, die der Computer erreichen kann, auf dem dieser Gruppenchat läuft. Legen Sie auf diesem Computer gateway.room_link_url fest, starten Sie Hermes dort neu und prüfen Sie es dann erneut.',
    peerSetupBackground:
      'Hermes stoppt dort, wenn sein Desktop geschlossen wird. Starten Sie dort sein Gateway eigenständig (hermes gateway run oder als Dienst installiert) und prüfen Sie es dann erneut.',
    peerSetupGeneric:
      'Aktualisieren Sie Hermes dort, aktivieren Sie seinen API-Server mit einer Adresse, die der Computer erreichen kann, auf dem dieser Gruppenchat läuft (gateway.room_link_url), und lassen Sie Freigaben eingeschaltet.',
    peerJoinFailed: '{name} ist noch nicht beigetreten: {reason}',
    peerJoinUnconfirmed:
      'Es konnte nicht bestätigt werden, ob {name} beigetreten ist. Prüfen Sie den Gruppenchat, sobald sein Computer erreichbar ist.',
    peerJoinUnreachable:
      '{name} ist noch nicht beigetreten: Der Computer, auf dem dieser Gruppenchat läuft, konnte seinen Computer nicht erreichen. Prüfen Sie, ob dieser Computer von dort ohne Desktop erreichbar ist, und versuchen Sie es erneut.',
    peerJoinChanged:
      '{name} kann diesem Gruppenchat nicht beitreten: Die Bot-Einstellungen auf seinem Computer haben sich seit dem Erstellen des Gruppenchats geändert. Starten Sie einen neuen Gruppenchat, um ihn aufzunehmen.',
    readTimedOut:
      'Das Gateway hat nicht rechtzeitig geantwortet. Möglicherweise wartet es auf den Computer eines Bots. Wählen Sie „Aktualisieren“, um es erneut zu versuchen.',
    disbandUnconfirmed:
      'Das Ende des Gruppenchats konnte nicht bestätigt werden. Der Computer eines Bots ist möglicherweise im Ruhezustand oder offline. Versuchen Sie es erneut, sobald er wieder erreichbar ist.',
    disbandKept: 'Der Gruppenchat bleibt erhalten. {reason}',
    disbandPeerUnreachable:
      'Die Arbeit darin wurde gestoppt. Der Computer eines Mitglieds antwortet nicht. Stellen Sie die Verbindung zu diesem Computer wieder her und versuchen Sie erneut, den Gruppenchat aufzulösen.',
    disbandStopPending:
      'Die Bots in diesem Gruppenchat haben noch nicht alle bestätigt, dass sie gestoppt wurden. Versuchen Sie erneut, den Gruppenchat aufzulösen, sobald alle dies bestätigt haben. Nutzen Sie bei einem wartenden Turn „Erneut versuchen“ oder „Verwerfen“, wenn diese Aktionen im Gruppenchat angeboten werden.',
    disbandMemberHomeUnknown:
      'Der Gruppenchat arbeitet möglicherweise noch, aber eines seiner Bot-Profile ist auf dem Computer, auf dem der Gruppenchat läuft, nicht mehr konfiguriert. Fügen Sie das Profil wieder zu hosted_rooms.profiles hinzu, damit sein Stopp bestätigt werden kann. Versuchen Sie dann erneut, den Gruppenchat aufzulösen.',
    disbandInProgress:
      'Die Auflösung konnte nicht abgeschlossen werden. Warten Sie kurz, prüfen Sie den Gruppenchat und versuchen Sie es bei Bedarf erneut.',
    disbandRevokeUnsupported:
      'Die Arbeit darin wurde gestoppt. Ein Mitglied kann seinen Zugriff auf diesen Gruppenchat nicht aufheben. Aktualisieren Sie Hermes auf dem Computer dieses Mitglieds und versuchen Sie dann erneut, den Gruppenchat aufzulösen.',
    disbandServiceRequired:
      'Hermes muss auf dem Computer laufen, der diesen Gruppenchat ausführt, um die Bots zu stoppen und ihren Zugriff aufzuheben. Starten Sie Hermes dort oder verbinden Sie es erneut und versuchen Sie dann erneut, den Gruppenchat aufzulösen.',
    peerStartWithoutRetry: 'Verwenden Sie für diese Bots im neuen Gruppenchat „Erneut versuchen“.',
    peerStartWithoutKept:
      'Der neue Gruppenchat wurde erstellt und der ursprüngliche Gruppenchat bleibt erhalten. {reason} Lösen Sie den ursprünglichen Gruppenchat auf, wenn Sie bereit sind.',
    createdNotJoined:
      'Der Gruppenchat wurde erstellt, aber {names} ist noch nicht beigetreten. Der Gruppenchat zeigt den Grund; verwenden Sie dort „Erneut versuchen“.',
    peerNotJoinedBadge: 'nicht beigetreten',
    peerOfflineBadge: 'nicht erreichbar',
    peerRejoinBadge: 'Zugriff abgelaufen',
    peerNotJoined: '{name} ist diesem Gruppenchat noch nicht beigetreten.',
    peerRejoin:
      'Der Zugriff von {name} auf diesen Gruppenchat ist abgelaufen. Um das Mitglied erneut einzuladen, wählen Sie „Erneut versuchen“ in einem Desktop, der mit seinem Computer verbunden ist.',
    peerAway:
      'Der Computer von {name} konnte nicht erreicht werden. Bei der nächsten Nachricht an dieses Mitglied wird es erneut versucht.',
    peerRetryFailed:
      'Die Verbindung zu {name} konnte nicht wiederhergestellt werden. Prüfen Sie, ob Hermes auf seinem Computer läuft, und versuchen Sie es erneut.',
    peerJoinConnect:
      '{name} ist noch nicht beigetreten: Verbinden Sie sich mit dem Computer, auf dem er läuft, und wählen Sie dann Erneut versuchen.',
    peerStartWithout: 'Neuen Gruppenchat ohne {name} starten',
    peerStartWithoutConfirm:
      'Ein neuer Gruppenchat wird mit den anderen Bots erstellt. Der ursprüngliche endet erst, wenn alle beigetreten sind und Stopp und Zugriffsentzug aller ursprünglichen Mitglieder bestätigt wurden. Schlägt ein Schritt fehl, bleiben beide Gruppenchats erhalten. Frühere Nachrichten werden nicht kopiert. Nach der Erstellung legt diese Aktion beim nächsten Mal einen weiteren Gruppenchat an.'
  },
  es: {
    ...GROUP_REASON_LOCALES['es'],
    roomStopped: 'Parada solicitada',
    roomDisbanded: 'Grupo disuelto',
    authorityLost: 'Propietario del grupo cambiado',
    workLimitReached: 'Límite de trabajo del grupo alcanzado',
    turnFailed: 'Turno fallido',
    turnCancelled: 'Turno cancelado',
    turnDeferred: 'Turno aplazado',
    turnNoAnswer: 'el bot no respondió a tiempo',
    memberUnavailable: 'Miembro no disponible',
    members: 'Miembros',
    openGroup: 'Abrir chat de grupo',
    loadingGroup: 'Cargando grupo…',
    loadingGroups: 'Cargando chats de grupo…',
    emptyGroups: 'No se encontraron chats de grupo.',
    driverUnavailable: 'Este chat de grupo no está disponible. Actualiza o vuelve a conectar la pasarela que lo aloja.',
    invalidLogCursor: 'No se pudo cargar más historial de este chat de grupo.',
    allowOnce: 'Permitir una vez',
    deny: 'Denegar',
    discardUnknown: 'Descartar turno',
    discardWarning:
      'Es posible que ya se hayan producido cambios. Descartar no los deshace. Si el bot está en otro equipo, se espera a que ese equipo confirme que el turno se ha detenido o nunca llegó a empezar.',
    discardStopPending:
      '{name} aún no ha confirmado que su turno se haya detenido. El turno se ha conservado. Vuelve a intentar descartarlo cuando su equipo responda.',
    retryBusy: 'Puede que {name} siga trabajando en este turno. Inténtalo de nuevo cuando termine.',
    discardPeerUnreachable:
      'El equipo de {name} no responde. El turno se ha conservado. Reconecta ese equipo y vuelve a intentar descartar el turno.',
    discardPeerUpdateRequired:
      'La versión de Hermes en el equipo de {name} es demasiado antigua para confirmar que este turno nunca llegó a empezar. El turno se ha conservado. Actualiza Hermes en ese equipo y vuelve a intentar descartar el turno.',
    confirmDiscard: 'Confirmar descarte',
    unconfirmedSend:
      'El envío anterior no está confirmado. Reintenta el envío del mensaje original antes de enviar otro mensaje.',
    restoredPendingSend: 'Se restauró un envío sin confirmar. Reinténtalo antes de enviar otro mensaje.',
    groupMessage: 'Mensaje de grupo',
    attachFiles: 'Adjuntar archivos',
    removeAttachment: 'Quitar adjunto',
    uploadFailed: 'Error al subir',
    memberWorking: 'Trabajando',
    memberQueued: 'en cola',
    groupWorking: 'El grupo está trabajando…',
    statusWorking: 'trabajando',
    statusStalled: 'sin avances desde hace {time}',
    statusReplied: 'respondió',
    statusPassed: 'pasó',
    statusFailed: 'falló',
    statusCancelled: 'cancelado',
    statusUnavailable: 'no disponible',
    statusIdle: 'inactivo',
    you: 'Tú',
    roundOf: 'Ronda {n} de {max}',
    round: 'Ronda {n}',
    cycleWorking: 'Trabajando: {names}',
    cycleSettled: 'Terminado — nadie tenía nada más que decir',
    cycleBounded: 'Se alcanzó el límite de rondas',
    cycleMessageCap: 'Se alcanzó el límite de mensajes',
    cycleStopped: 'Detenido',
    typing: '{name} está trabajando…',
    typingTool: '{name} está ejecutando {tool}…',
    handoff: '{from} pidió a {to}',
    replyTo: 'Responder a {name}',
    replyingTo: 'Respondiendo a {name}',
    cancelReply: 'Cancelar respuesta',
    composerPlaceholder: 'Mensaje al grupo — @menciona para asignar',
    groupDetails: 'Detalles del grupo',
    stopGroup: 'Detener el grupo',
    pendingApproval: '{name} pide aprobación. Tú decides.',
    pendingRetry: 'El turno de {name} no ha terminado. ¿Reintentar?',
    pendingDiscard: '¿Descartar el turno de {name}? El trabajo ya realizado no se deshará.',
    membersFixed: 'Los miembros se fijan al crear el grupo.',
    rowWorking: 'Trabajando',
    rowFailed: 'Un miembro falló',
    rowNeedsYou: 'Te necesita',
    lead: 'Responsable',
    roomSettings: 'Ajustes del chat de grupo',
    purposeLabel: 'Propósito',
    purposePlaceholder: 'Para qué sirve este chat de grupo, en una línea',
    notesLabel: 'Notas compartidas',
    notesPlaceholder: 'Lo que cada miembro debe tener presente',
    deliveryTarget: 'Las tareas programadas pueden publicar aquí con deliver:',
    scheduledPost: 'Programado',
    routingLabel: 'Quién responde a un mensaje que no nombra a nadie',
    routingEveryone: 'Responden todos',
    routingLead: 'Líder: {name}',
    routingEveryoneHint: 'Cada miembro tiene su turno, como ahora.',
    routingLeadHint:
      'El líder elige quién se encarga; las @menciones y @everyone siguen yendo directamente a esos miembros.',
    leadBadge: 'Líder',
    assignment: '{lead} se lo dio a {names}',
    assignmentFallback: '{lead} se encarga',
    routedTo: 'Asignado a {names}',
    routedEveryone: 'Asignado a todos',
    routedFallback: 'No se pudo elegir quién responde, así que responderán todos',
    pausedAfter: 'En pausa tras {count} mensajes',
    continueDiscussion: 'Continuar',
    continued: 'Reanudado',
    routingAuto: 'Elegir automáticamente quién responde',
    routingAutoHint:
      'Una llamada rápida al modelo elige a quién va dirigido un mensaje. Las @menciones siguen yendo directamente a quienes nombras.',
    statusBlocked: 'Bloqueado',
    statusApprovals: '{count} esperando aprobación',
    statusAttention: 'Atención necesaria: {count}',
    stopped:
      'Parada solicitada. Los turnos pendientes de confirmación siguen visibles; usa Reintentar o Descartar cuando se ofrezcan.',
    pendingActionUnconfirmed: 'No pudimos confirmar esta acción. Actualiza el chat para comprobar su estado.',
    sendRefused: 'La pasarela rechazó este mensaje. Edítalo y envíalo de nuevo.',
    sendStopped: 'Se detuvo antes de enviar. Tu mensaje sigue aquí.',
    sendMaybe:
      'Es posible que este mensaje ya se haya enviado. Reintentar envía el mismo mensaje sin publicarlo dos veces.',
    sendSettledElsewhere:
      'Otra ventana de Desktop ya terminó con este mensaje. Revisa la conversación antes de volver a enviarlo.',
    journalLoadFailed:
      'No se pudieron cargar los mensajes guardados. El envío está en pausa para evitar enviar un mensaje guardado dos veces.',
    journalReload: 'Volver a cargar',
    approvalAction: 'Acción',
    approvalCommand: 'Comando',
    approvalDetailsMissing: 'Los detalles de la acción no están disponibles. Actualízalos antes de permitirla.',
    checkingDriver: 'Comprobando los chats de grupo de esta conexión…',
    legacyRoom:
      'Desktop ejecutaba este chat de grupo por sí mismo, así que no puede continuar en esta pasarela. Inicia un grupo nuevo con los mismos bots: seguirá funcionando cuando Desktop esté cerrado. Los mensajes anteriores no se copian en él.',
    startGatewayGroup: 'Iniciar nuevo grupo',
    createRefused: 'La pasarela no pudo crear este chat de grupo.',
    createGroupSize: 'Un chat de grupo alojado en esta pasarela necesita de dos a seis bots.',
    createOtherConnection: 'Un chat de grupo alojado en esta pasarela solo puede incluir bots de esta conexión.',
    createDuplicateMember:
      'Un chat de grupo alojado en esta pasarela necesita bots con perfiles e identificadores distintos; @all y @everyone están reservados.',
    createUncertain:
      'La pasarela no respondió, así que este chat de grupo puede que ya exista. Inténtalo de nuevo para abrirlo. No se creará dos veces.',
    createEnded:
      'La pasarela rechazó este chat de grupo; puede que se haya cambiado o eliminado en otro lugar. Inténtalo de nuevo para crear uno nuevo.',
    legacyRoomElsewhere:
      'Este chat de grupo no puede continuar aquí: tiene bots en otros equipos, y un chat de grupo alojado en esta pasarela solo incluye bots de una conexión. Cambia a la conexión donde funcionan esos bots e inicia un grupo allí, o inicia un chat de grupo nuevo con bots de esta conexión. Los mensajes anteriores no se copian en un grupo nuevo.',
    otherConnections: 'Bots en otras conexiones: {connections}',
    legacyRoomBlocked:
      'Desktop ejecutaba este chat de grupo por sí mismo, así que no puede continuar en esta pasarela, y estos bots no pueden formar un chat de grupo alojado en esta pasarela tal como están (el motivo está abajo). Inicia un chat de grupo nuevo y elige sus bots. Los mensajes anteriores no se copian en él.',
    groupsUnsupported:
      'Los chats de grupo no se pueden abrir aquí: Hermes en el equipo de esta conexión es demasiado antiguo para ellos. Actualiza Hermes en ese equipo y vuelve a conectarte; después, inténtalo de nuevo. No se pierde nada: los mensajes anteriores siguen donde están.',
    groupDriverStopped:
      'Los chats de grupo no se pueden abrir aquí ahora mismo: la pasarela de esta conexión no está ejecutando los chats de grupo. Reinicia o actualiza Hermes en ese equipo y vuelve a intentarlo. No se pierde nada: los mensajes anteriores siguen donde están.',
    groupsUnreachable:
      'No se pudo contactar con la pasarela de esta conexión para abrir este chat de grupo. Comprueba que Hermes se esté ejecutando allí y que estés conectado, y vuelve a intentarlo. No se pierde nada: los mensajes anteriores siguen donde están.',
    earlierMessages: 'Mensajes anteriores, solo lectura. Se quedan en este Desktop y no se copian en un grupo nuevo.',
    peerTargetSetup: '{connection} todavía no puede aportar un bot a un chat de grupo de otro equipo. {reason}',
    peerSetupUpdate: 'Hermes es demasiado antiguo allí. Actualiza Hermes en ese equipo y vuelve a comprobarlo.',
    peerSetupApiServer:
      'Su servidor API está desactivado. Activa allí el servidor API (gateway.platforms.api_server), reinicia Hermes en ese equipo y vuelve a comprobarlo.',
    peerSetupRunStorage:
      'Su servidor API no puede guardar los registros de ejecución, así que un mensaje podría ejecutarse dos veces. Comprueba que Hermes pueda escribir en su carpeta de datos en ese equipo, reinicia Hermes allí y vuelve a comprobarlo.',
    peerSetupRoomKey:
      'Hermes no puede crear allí la clave que protege el acceso a los chats de grupo. Comprueba que Hermes pueda escribir en su carpeta de datos en ese equipo, reinicia Hermes allí y vuelve a comprobarlo.',
    peerSetupApprovals:
      'Las aprobaciones de comandos están desactivadas allí, o no se puede leer su configuración. Configura las aprobaciones como manual o smart en ese equipo (approvals.mode) y vuelve a comprobarlo.',
    peerSetupAddress:
      'No tiene una dirección que el equipo que gestiona este chat de grupo pueda alcanzar. Configura gateway.room_link_url en ese equipo, reinicia Hermes allí y vuelve a comprobarlo.',
    peerSetupBackground:
      'Hermes se detiene allí cuando se cierra su Desktop. Ejecuta allí su pasarela por sí sola (hermes gateway run, o instálala como servicio) y vuelve a comprobarlo.',
    peerSetupGeneric:
      'Actualiza Hermes allí, activa su servidor API con una dirección que el equipo que gestiona este chat de grupo pueda alcanzar (gateway.room_link_url) y mantén activadas las aprobaciones.',
    peerJoinFailed: '{name} aún no se ha unido: {reason}',
    peerJoinUnconfirmed:
      'No se pudo confirmar si {name} se ha unido. Revisa el chat de grupo cuando su equipo esté disponible.',
    peerJoinUnreachable:
      '{name} aún no se ha unido: el equipo que gestiona este chat de grupo no pudo alcanzar su equipo. Comprueba que se pueda alcanzar desde allí sin Desktop y vuelve a intentarlo.',
    peerJoinChanged:
      '{name} no puede unirse a este chat de grupo: la configuración del bot en su equipo cambió desde que se creó el chat de grupo. Inicia un chat de grupo nuevo para incluirlo.',
    readTimedOut:
      'La pasarela no respondió a tiempo. Puede que esté esperando al equipo de un bot. Elige Actualizar para reintentarlo.',
    disbandUnconfirmed:
      'No pudimos confirmar que el chat de grupo haya terminado. El equipo de un bot puede estar suspendido o sin conexión. Reinténtalo cuando vuelva a estar disponible.',
    disbandKept: 'El chat de grupo se ha conservado. {reason}',
    disbandPeerUnreachable:
      'El trabajo se ha detenido. El equipo de un miembro no responde. Reconecta ese equipo y vuelve a intentar disolver el chat de grupo.',
    disbandStopPending:
      'Los bots de este chat de grupo aún no han confirmado todos que se hayan detenido. Vuelve a intentar disolver el chat cuando todos lo confirmen. Para un turno pendiente, usa Reintentar o Descartar si esas acciones se ofrecen en el chat de grupo.',
    disbandMemberHomeUnknown:
      'El chat de grupo puede seguir trabajando, pero uno de sus perfiles de bot ya no está configurado en el equipo que ejecuta el chat de grupo. Vuelve a añadir ese perfil a hosted_rooms.profiles para poder confirmar su parada y vuelve a intentar disolver el chat de grupo.',
    disbandInProgress:
      'No se pudo completar la disolución. Espera un momento, comprueba el chat de grupo y vuelve a intentarlo si es necesario.',
    disbandRevokeUnsupported:
      'El trabajo se ha detenido. Un miembro no puede revocar su acceso a este chat de grupo. Actualiza Hermes en el equipo de ese miembro y vuelve a intentar disolver el chat de grupo.',
    disbandServiceRequired:
      'Hermes debe estar en ejecución en el equipo que gestiona este chat de grupo para detener sus bots y revocar sus accesos. Inicia o reconecta Hermes allí y vuelve a intentar disolver el chat de grupo.',
    peerStartWithoutRetry: 'Usa Reintentar para estos bots en el nuevo chat de grupo.',
    peerStartWithoutKept:
      'El nuevo chat de grupo se ha creado y el original se ha conservado. {reason} Disuelve el chat original cuando estés listo.',
    createdNotJoined:
      'El chat de grupo se ha creado, pero {names} aún no se ha unido. El chat de grupo indica el motivo; usa Reintentar allí.',
    peerNotJoinedBadge: 'sin unirse',
    peerOfflineBadge: 'sin conexión',
    peerRejoinBadge: 'acceso caducado',
    peerNotJoined: '{name} aún no se ha unido a este chat de grupo.',
    peerRejoin:
      'El acceso de {name} a este chat de grupo ha caducado. Para volver a invitarlo, elige Reintentar desde un Desktop conectado a su equipo.',
    peerAway:
      'No se pudo conectar con el equipo de {name}. El próximo mensaje dirigido a ese miembro lo volverá a intentar.',
    peerRetryFailed:
      'No se pudo reconectar a {name}. Comprueba que Hermes se esté ejecutando en su equipo y vuelve a intentarlo.',
    peerJoinConnect: '{name} aún no se ha unido: conéctate al equipo donde funciona y elige Reintentar.',
    peerStartWithout: 'Iniciar nuevo chat de grupo sin {name}',
    peerStartWithoutConfirm:
      'Se crea un nuevo chat de grupo con los demás bots. El original solo termina cuando todos se unen y cada miembro original confirma la detención y la revocación de acceso. Si falla algún paso, se conservan ambos chats. Los mensajes anteriores no se copian. Usar esta acción de nuevo tras la creación genera otro chat de grupo.'
  }
} satisfies Record<string, CanonicalGroupMessages>
