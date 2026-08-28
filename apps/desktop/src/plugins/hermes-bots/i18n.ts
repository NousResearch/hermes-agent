/**
 * Plugin-scoped i18n for Bot Mode — bundles registered under the plugin id via
 * `ctx.i18n.register`, never touching core `en.ts`. Mirrors the kanban plugin:
 * `usePluginI18n` returns a stringly-typed `t(key, …)`, and `useBots()` binds it
 * to the message SHAPE so components keep typed `b.roster.search` access.
 *
 * Only strings Bot Mode OWNS live here. Generic verbs (Cancel, Delete, Retry,
 * Close, Loading…) and shared concepts (Scheduled jobs) resolve against core's
 * `useI18n()` instead — core already carries them in every locale, so
 * duplicating them here would be a second, worse translation.
 *
 * Locales follow kanban: `en` / `ja` / `zh` / `zh-hant`. Arabic falls through
 * the resolution chain (active locale → this plugin's `en` → the key) the
 * same way a missing string in any locale does. Nouns match core: ボット /
 * 机器人 / 機器人, プロファイル / 配置档案 / 設定檔, ゲートウェイ / 网关 / 閘道.
 */

import { type PluginLocaleBundles, type PluginTranslate, usePluginI18n } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

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
  /** Creating, editing and removing a bot. */
  bot: {
    newTitle: string
    editTitle: string
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
  }
  /** Group chats: the room, its composer, threads and activity feed. */
  group: {
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
    needsYourInput: string
    pictureGenerationFailed: string
  }
  /** Skills hub + MCP setup surfaces embedded in the bot editor. */
  tools: {
    skillsHub: string
    filterSkills: string
    searchHub: string
    noMcpServers: string
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
  bot: {
    newTitle: 'New Bot',
    editTitle: 'Edit Profile',
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
    chatEmpty: 'Say something to get started.'
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
    savedLocallyDescriptionFailed: 'Saved look locally; description update failed'
  },
  group: {
    newTitle: 'New Group Chat',
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
    deleteAction: 'Delete Group',
    composerPlaceholder: 'Say something — every bot in this group hears the room.',
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
    needsYourInput: 'A bot in this group chat needs your input',
    pictureGenerationFailed: 'Group picture generation failed'
  },
  tools: {
    skillsHub: 'Hermes Skills Hub',
    filterSkills: 'Filter skills…',
    searchHub: 'Search the hub (community + well-known sources)…',
    noMcpServers: 'No MCP servers configured or in the catalog.'
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
  bot: {
    newTitle: '新しいボット',
    editTitle: 'プロファイルを編集',
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
    chatEmpty: '何か書いて始めましょう。'
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
    savedLocallyDescriptionFailed: '見た目はローカルに保存されましたが、説明の更新に失敗しました'
  },
  group: {
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
    deleteAction: 'グループを削除',
    composerPlaceholder: '何か書いてください — このグループのすべてのボットが部屋の内容を受け取ります。',
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
    needsYourInput: 'このグループチャットのボットが入力を待っています',
    pictureGenerationFailed: 'グループ画像の生成に失敗しました'
  },
  tools: {
    skillsHub: 'Hermes スキルハブ',
    filterSkills: 'スキルを絞り込み…',
    searchHub: 'ハブを検索（コミュニティと既知のソース）…',
    noMcpServers: '設定済みまたはカタログ内の MCP サーバーはありません。'
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
  bot: {
    newTitle: '新建机器人',
    editTitle: '编辑配置档案',
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
    chatEmpty: '说点什么开始吧。'
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
    savedLocallyDescriptionFailed: '外观已保存在本地；描述更新失败'
  },
  group: {
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
    deleteAction: '删除群组',
    composerPlaceholder: '说点什么 — 这个群里的每个机器人都会听到。',
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
    needsYourInput: '此群聊中有机器人需要你输入',
    pictureGenerationFailed: '群组图片生成失败'
  },
  tools: {
    skillsHub: 'Hermes 技能中心',
    filterSkills: '筛选技能…',
    searchHub: '搜索技能中心（社区和常见来源）…',
    noMcpServers: '未配置 MCP 服务器，目录中也没有。'
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
  bot: {
    newTitle: '新增機器人',
    editTitle: '編輯設定檔',
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
    chatEmpty: '說點什麼開始吧。'
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
    savedLocallyDescriptionFailed: '外觀已儲存在本機；描述更新失敗'
  },
  group: {
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
    deleteAction: '刪除群組',
    composerPlaceholder: '說點什麼 — 這個群組裡的每個機器人都會聽到。',
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
    needsYourInput: '此群組聊天中有機器人需要您的輸入',
    pictureGenerationFailed: '群組圖片產生失敗'
  },
  tools: {
    skillsHub: 'Hermes 技能中心',
    filterSkills: '篩選技能…',
    searchHub: '搜尋技能中心（社群和常見來源）…',
    noMcpServers: '未設定 MCP 伺服器，目錄中也沒有。'
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
