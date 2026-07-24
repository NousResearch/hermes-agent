/**
 * Plugin-scoped i18n for kanban — bundles shipped under the plugin id via
 * ctx.i18n.register (#67303), never touching core en.ts. usePluginI18n('kanban')
 * returns a stringly-typed t(key, …); `useKanban()` binds it to the message
 * SHAPE so components keep typed `k.newTask` / `k.moveTo(label)` access.
 */

import { type PluginLocaleBundles, type PluginTranslate, usePluginI18n } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

type KanbanMessages = {
    nav: string
    openBoard: string
    countTip: (running: number, ready: number) => string
    col: Record<
      'archived' | 'blocked' | 'done' | 'ready' | 'review' | 'running' | 'scheduled' | 'todo' | 'triage',
      { label: string; help: string }
    >
    locked: { review: string; running: string; scheduled: string }
    arcRunning: string
    arcStale: string
    title: string
    orchestrationSettings: string
    newTask: string
    filterCards: string
    noMatch: string
    noTasks: string
    open: string
    select: string
    deselect: string
    moveTo: (label: string) => string
    delete: string
    reviewChecking: string
    attachedTip: (name: string) => string
    orchestratorTip: (name: string) => string
    autoAssignTip: (name: string) => string
    wontRun: string
    wontRunTip: string
    noHeartbeat: string
    expand: (label: string) => string
    collapse: (label: string) => string
    newTaskIn: (label: string) => string
    empty: string
    unassigned: string
    filters: string
    allProfiles: string
    allTenants: string
    showArchived: string
    groupRunning: string
    nSelected: (n: number) => string
    moveToShort: string
    assign: string
    unassignAction: string
    archive: string
    clearSelection: string
    refused: string
    bulkFailed: (failed: number, total: number, err: string) => string
    titlePlaceholderTriage: string
    titlePlaceholder: string
    descPlaceholder: string
    priority: string
    workspace: string
    boardDefaultSuffix: string
    workspaceOverride: string
    workspaceInherit: string
    workspaceInheritDir: (dir: string) => string
    workspaceInheritGeneric: string
    assignee: string
    defaultOption: (name: string) => string
    parkedOption: string
    skills: string
    skillsPlaceholder: string
    parent: string
    noParent: string
    goalMode: string
    creating: string
    createTask: string
    cancel: string
    save: string
    estimate: string
    estimateEffort: string
    estimating: string
    reEstimate: string
    makesModelCall: string
    estimateTip: string
    estimateTipLong: string
    roughEstimate: string
    tokUnit: string
    couldNotEstimate: string
    complexity: Record<'L' | 'M' | 'S', string>
    introBody: string
    introGotIt: string
    // drawer — activity prose
    evtCreated: (where: string, assignee: string) => string
    evtMovedTo: (col: string) => string
    evtParentReopened: (parent: string) => string
    evtAssignedTo: (assignee: string) => string
    evtUnassigned: string
    evtCommentBy: (author: string) => string
    evtClaimedReview: string
    evtClaimedWorker: string
    evtWorkerStarted: string
    evtCompleted: string
    evtBlocked: string
    evtUnblocked: (col: string) => string
    evtReclaimed: string
    evtSpecified: string
    evtPromoted: string
    evtScheduled: string
    evtArchived: string
    evtReprioritized: (priority: string) => string
    someone: string
    // drawer — meta + sections
    metaPriority: string
    metaTenant: string
    metaCreatedBy: string
    metaCreated: string
    metaWorkerPid: string
    readyUnassignedTitle: string
    readyUnassignedBody: string
    diagnosticsN: (n: number) => string
    commandCopied: string
    description: string
    editDescription: string
    cancelEdit: string
    noDescription: string
    result: string
    latestSummary: string
    dependencies: string
    blockedBy: string
    blocks: string
    comments: (n: number) => string
    commentsHelpRunning: string
    commentsHelp: string
    send: string
    comment: string
    messageWorker: string
    addComment: string
    deliveredLive: string
    requeueWithNote: string
    notePosted: string
    activity: (n: number) => string
    runs: (n: number) => string
    workerLog: string
    workerLogTail: string
    attachments: (n: number) => string
    noAttachments: string
    uploadAttachment: string
    taskActions: string
    copyTaskId: string
    copyTitle: string
    copiedId: (id: string) => string
    copiedTitle: string
    archiveTask: string
    deleteTask: string
    close: string
    working: string
    // board switcher
    board: string
    newBoard: string
    newBoardDots: string
    boardSettings: string
    boardSettingsFor: (name: string) => string
    name: string
    boardNamePlaceholder: string
    slug: (slug: string) => string
    project: string
    noProject: string
    projectHintPre: string
    projectHintCmd: string
    createBoard: string
    // orchestration
    orchestratorProfile: string
    defaultAssignee: string
    defaultParen: string
    autoDecompose: string
    profileDescriptions: string
    profileDescriptionsHint: string
    profileGoodAt: string
    auto: string
  }

const en: KanbanMessages = {
    nav: 'Kanban',
    openBoard: 'Kanban: Open board',
    countTip: (running, ready) => `Kanban — ${running} running, ${ready} ready`,
    col: {
      triage: { label: 'Triage', help: 'Raw ideas — a specifier fleshes out the spec.' },
      todo: { label: 'Todo', help: 'Waiting on dependencies, or unassigned.' },
      scheduled: { label: 'Scheduled', help: 'Waiting for a scheduled time to arrive.' },
      ready: { label: 'Ready', help: 'Dependencies satisfied — assign a profile and the dispatcher runs it.' },
      running: { label: 'Running', help: 'Claimed by a worker — an agent is on it. Set by the dispatcher.' },
      blocked: { label: 'Blocked', help: 'The worker asked for human input.' },
      review: { label: 'Review', help: 'A review agent is checking the work. Set by the dispatcher.' },
      done: { label: 'Done', help: 'Completed; dependent children become ready.' },
      archived: { label: 'Archived', help: 'Hidden from the default board view.' }
    },
    locked: {
      review: 'Review is entered by the dispatcher when a review agent takes the card.',
      running: 'Running is set by the dispatcher when a worker claims the card.',
      scheduled: 'Scheduled needs a wake-up time — agents set it; it can’t be dragged into.'
    },
    arcRunning: 'An agent is working on this now.',
    arcStale: 'Claimed, but no worker heartbeat for 2+ minutes — the dispatcher will reclaim it.',
    title: 'Kanban',
    orchestrationSettings: 'Orchestration settings',
    newTask: 'New task',
    filterCards: 'Filter cards…',
    noMatch: 'No tasks match the filters',
    noTasks: 'No tasks on this board',
    open: 'Open',
    select: 'Select (⌘-click)',
    deselect: 'Deselect',
    moveTo: label => `Move to ${label}`,
    delete: 'Delete',
    reviewChecking: 'A review agent is checking the completed work.',
    attachedTip: name => `${name} is attached — the dispatcher hands this over on its next tick (≤1m).`,
    orchestratorTip: name => `${name} (the orchestrator) picks this up on the next tick and writes the spec.`,
    autoAssignTip: name => `Auto-assigns to “${name}” (kanban.default_assignee) on the next dispatch tick.`,
    wontRun: "won't run",
    wontRunTip:
      'Ready cards only run once a profile is assigned. Open the card and set an assignee, or configure a default assignee in orchestration settings.',
    noHeartbeat: 'no heartbeat',
    expand: label => `Expand ${label}`,
    collapse: label => `Collapse ${label}`,
    newTaskIn: label => `New task in ${label}`,
    empty: 'Empty',
    unassigned: 'unassigned',
    filters: 'Filters',
    allProfiles: 'All profiles',
    allTenants: 'All tenants',
    showArchived: 'Show archived',
    groupRunning: 'Group Running by profile',
    nSelected: n => `${n} selected`,
    moveToShort: 'Move to',
    assign: 'Assign',
    unassignAction: 'Unassign',
    archive: 'Archive',
    clearSelection: 'Clear selection (Esc)',
    refused: 'refused',
    bulkFailed: (failed, total, err) => `${failed} of ${total} failed — ${err}. Failed cards stay selected.`,
    titlePlaceholderTriage: 'Rough idea — a specifier will flesh it out',
    titlePlaceholder: 'Title',
    descPlaceholder: 'Description (optional)',
    priority: 'Priority',
    workspace: 'Workspace',
    boardDefaultSuffix: ' · board default',
    workspaceOverride: 'Workspace path (optional override)',
    workspaceInherit: 'Inherits the board’s project directory',
    workspaceInheritDir: dir => `Leave empty to inherit ${dir}`,
    workspaceInheritGeneric: 'Leave empty to inherit the board’s project directory.',
    assignee: 'Assignee',
    defaultOption: name => `${name} (default)`,
    parkedOption: "unassigned (parked — won't run)",
    skills: 'Skills (comma-separated)',
    skillsPlaceholder: 'translation, github',
    parent: "Parent (blocks until it's done)",
    noParent: '— no parent —',
    goalMode: "Goal mode (worker loops until a judge agrees it's done)",
    creating: 'Creating…',
    createTask: 'Create task',
    cancel: 'Cancel',
    save: 'Save',
    estimate: 'Estimate',
    estimateEffort: 'Estimate effort',
    estimating: 'Estimating…',
    reEstimate: 'Re-estimate',
    makesModelCall: 'makes a model call',
    estimateTip: 'Rough token + complexity estimate from the auxiliary model — makes a model call.',
    estimateTipLong: 'Runs a quick auxiliary-model call to estimate tokens + complexity. A rough guide, not a bill.',
    roughEstimate: 'Rough estimate',
    tokUnit: 'tok',
    couldNotEstimate: 'Could not estimate',
    complexity: { S: 'Small', M: 'Medium', L: 'Large' },
    introBody:
      'You don’t run the cards — agents do. Put a card in Ready with an assignee and an agent picks it up within a minute. No assignee, no run. Triage: an agent rewrites the idea into a proper task first. Todo: waiting on other cards. Scheduled: waiting on a timer. Running and Review: the agents’ lanes, hands off. Blocked: it’s waiting on you. Results come back on the card.',
    introGotIt: 'Got it',
    evtCreated: (where, assignee) => `created${where ? ` in ${where}` : ''}${assignee ? ` · assigned to ${assignee}` : ''}`,
    evtMovedTo: col => `moved to ${col}`,
    evtParentReopened: parent => `parent ${parent} reopened`,
    evtAssignedTo: assignee => `assigned to ${assignee}`,
    evtUnassigned: 'unassigned',
    evtCommentBy: author => `comment by ${author}`,
    evtClaimedReview: 'claimed by a review agent',
    evtClaimedWorker: 'claimed by a worker',
    evtWorkerStarted: 'worker started',
    evtCompleted: 'completed',
    evtBlocked: 'blocked — needs human input',
    evtUnblocked: col => `unblocked${col ? ` → ${col}` : ' → Ready'}`,
    evtReclaimed: 'reclaimed — returned to the queue',
    evtSpecified: 'spec written by the triage agent',
    evtPromoted: 'dependencies done — promoted to Ready',
    evtScheduled: 'scheduled for later',
    evtArchived: 'archived',
    evtReprioritized: priority => `priority set to ${priority}`,
    someone: 'someone',
    metaPriority: 'Priority',
    metaTenant: 'Tenant',
    metaCreatedBy: 'Created by',
    metaCreated: 'Created',
    metaWorkerPid: 'Worker pid',
    readyUnassignedTitle: 'Ready, but unassigned — this card will never run.',
    readyUnassignedBody:
      'The dispatcher only claims Ready cards that have an assignee. Pick a profile in the Assignee field above (or set a default assignee in the orchestration settings) and it runs within a minute.',
    diagnosticsN: n => `Diagnostics · ${n}`,
    commandCopied: 'Command copied',
    description: 'Description',
    editDescription: 'Edit description',
    cancelEdit: 'Cancel edit',
    noDescription: 'No description yet.',
    result: 'Result',
    latestSummary: 'Latest summary',
    dependencies: 'Dependencies',
    blockedBy: 'Blocked by',
    blocks: 'Blocks',
    comments: n => `Comments · ${n}`,
    commentsHelpRunning:
      'This task is running. Your note is folded into the worker’s current turn within a few seconds — no block/unblock dance. “Requeue with note” instead restarts the task from scratch with your note in context.',
    commentsHelp:
      'Comments are added to the task thread. When a worker picks the task up it reads them as part of its context.',
    send: 'Send',
    comment: 'Comment',
    messageWorker: 'Message the running worker…',
    addComment: 'Add a comment…',
    deliveredLive: 'Delivered to the running worker within a few seconds.',
    requeueWithNote: 'Requeue with note',
    notePosted: 'Note posted — worker requeued',
    activity: n => `Activity · ${n}`,
    runs: n => `Runs · ${n}`,
    workerLog: 'Worker log',
    workerLogTail: 'Worker log · tail',
    attachments: n => `Attachments · ${n}`,
    noAttachments: 'No attachments yet.',
    uploadAttachment: 'Upload attachment',
    taskActions: 'Task actions',
    copyTaskId: 'Copy task id',
    copyTitle: 'Copy title',
    copiedId: id => `Copied ${id}`,
    copiedTitle: 'Copied title',
    archiveTask: 'Archive task',
    deleteTask: 'Delete task',
    close: 'Close',
    working: 'working',
    board: 'Board',
    newBoard: 'New board',
    newBoardDots: 'New board…',
    boardSettings: 'Board settings…',
    boardSettingsFor: name => `Board settings — ${name}`,
    name: 'Name',
    boardNamePlaceholder: 'Board name',
    slug: slug => `slug: ${slug}`,
    project: 'Project',
    noProject: 'No project (scratch sandboxes)',
    projectHintPre:
      'New tasks run in the project’s repo (a worktree per task); each task can still override its workspace at creation. Manage projects with ',
    projectHintCmd: 'hermes project',
    createBoard: 'Create board',
    orchestratorProfile: 'Orchestrator profile',
    defaultAssignee: 'Default assignee',
    defaultParen: '(default)',
    autoDecompose: 'Auto-decompose triage tasks',
    profileDescriptions: 'Profile descriptions',
    profileDescriptionsHint:
      'Descriptions guide the decomposer’s routing. Auto-generate with the auxiliary model, or write your own.',
    profileGoodAt: 'What is this profile good at?',
    auto: 'Auto'
  }

const ja: KanbanMessages = {
    nav: 'カンバン',
    openBoard: 'カンバン: ボードを開く',
    countTip: (running, ready) => `カンバン — 実行中 ${running}、待機 ${ready}`,
    col: {
      triage: { label: 'トリアージ', help: '生のアイデア — スペシファイアが仕様に整えます。' },
      todo: { label: 'Todo', help: '依存関係の待ち、または未割り当て。' },
      scheduled: { label: 'スケジュール', help: '予定時刻を待っています。' },
      ready: { label: 'Ready', help: '依存関係が解決済み — プロフィールを割り当てるとディスパッチャが実行します。' },
      running: { label: '実行中', help: 'ワーカーが取得済み — エージェントが作業中。ディスパッチャが設定します。' },
      blocked: { label: 'ブロック', help: 'ワーカーが人間の入力を求めています。' },
      review: { label: 'レビュー', help: 'レビューエージェントが作業を確認中。ディスパッチャが設定します。' },
      done: { label: '完了', help: '完了。依存する子タスクが Ready になります。' },
      archived: { label: 'アーカイブ', help: 'デフォルトのボード表示から非表示。' }
    },
    locked: {
      review: 'レビューは、レビューエージェントがカードを取得するとディスパッチャによって設定されます。',
      running: '実行中は、ワーカーがカードを取得するとディスパッチャによって設定されます。',
      scheduled: 'スケジュールには起動時刻が必要です — エージェントが設定します。ドラッグでは移動できません。'
    },
    arcRunning: 'エージェントが現在作業中です。',
    arcStale: '取得済みですが、2分以上ワーカーのハートビートがありません — ディスパッチャが再取得します。',
    title: 'カンバン',
    orchestrationSettings: 'オーケストレーション設定',
    newTask: '新しいタスク',
    filterCards: 'カードを絞り込み…',
    noMatch: 'フィルタに一致するタスクはありません',
    noTasks: 'このボードにタスクはありません',
    open: '開く',
    select: '選択（⌘クリック）',
    deselect: '選択解除',
    moveTo: label => `${label} へ移動`,
    delete: '削除',
    reviewChecking: 'レビューエージェントが完了した作業を確認中です。',
    attachedTip: name => `${name} が担当 — ディスパッチャが次のティック（≤1分）で引き渡します。`,
    orchestratorTip: name => `${name}（オーケストレーター）が次のティックでこれを取得し、仕様を書きます。`,
    autoAssignTip: name => `次のディスパッチティックで「${name}」（kanban.default_assignee）に自動割り当てされます。`,
    wontRun: '実行されません',
    wontRunTip:
      'Ready のカードはプロフィールが割り当てられて初めて実行されます。カードを開いて担当を設定するか、オーケストレーション設定でデフォルトの担当を設定してください。',
    noHeartbeat: 'ハートビートなし',
    expand: label => `${label} を展開`,
    collapse: label => `${label} を折りたたむ`,
    newTaskIn: label => `${label} に新しいタスク`,
    empty: '空',
    unassigned: '未割り当て',
    filters: 'フィルタ',
    allProfiles: 'すべてのプロフィール',
    allTenants: 'すべてのテナント',
    showArchived: 'アーカイブを表示',
    groupRunning: '実行中をプロフィールでグループ化',
    nSelected: n => `${n} 件選択中`,
    moveToShort: '移動',
    assign: '割り当て',
    unassignAction: '割り当て解除',
    archive: 'アーカイブ',
    clearSelection: '選択をクリア（Esc）',
    refused: '拒否されました',
    bulkFailed: (failed, total, err) => `${total} 件中 ${failed} 件が失敗 — ${err}。失敗したカードは選択されたままです。`,
    titlePlaceholderTriage: '大まかなアイデア — スペシファイアが具体化します',
    titlePlaceholder: 'タイトル',
    descPlaceholder: '説明（任意）',
    priority: '優先度',
    workspace: 'ワークスペース',
    boardDefaultSuffix: '・ボード既定',
    workspaceOverride: 'ワークスペースパス（任意の上書き）',
    workspaceInherit: 'ボードのプロジェクトディレクトリを継承',
    workspaceInheritDir: dir => `空欄にすると ${dir} を継承します`,
    workspaceInheritGeneric: '空欄にするとボードのプロジェクトディレクトリを継承します。',
    assignee: '担当',
    defaultOption: name => `${name}（既定）`,
    parkedOption: '未割り当て（保留 — 実行されません）',
    skills: 'スキル（カンマ区切り）',
    skillsPlaceholder: 'translation, github',
    parent: '親（完了するまでブロック）',
    noParent: '— 親なし —',
    goalMode: 'ゴールモード（ジャッジが完了と認めるまでワーカーがループ）',
    creating: '作成中…',
    createTask: 'タスクを作成',
    cancel: 'キャンセル',
    save: '保存',
    estimate: '見積もり',
    estimateEffort: '工数を見積もり',
    estimating: '見積もり中…',
    reEstimate: '再見積もり',
    makesModelCall: 'モデル呼び出しあり',
    estimateTip: '補助モデルによるトークン数と複雑度の概算 — モデル呼び出しを行います。',
    estimateTipLong: '補助モデルを呼び出してトークン数と複雑度を概算します。目安であり、請求ではありません。',
    roughEstimate: '概算',
    tokUnit: 'tok',
    couldNotEstimate: '見積もりできませんでした',
    complexity: { S: '小', M: '中', L: '大' },
    introBody:
      'カードはあなたではなくエージェントが実行します。担当を設定したカードを Ready に置くと、1分以内にエージェントが取得します。担当がなければ実行されません。トリアージ: エージェントがまずアイデアを適切なタスクに書き直します。Todo: 他のカード待ち。スケジュール: タイマー待ち。実行中とレビュー: エージェントのレーンなので手を出さないでください。ブロック: あなたの対応待ちです。結果はカードに戻ってきます。',
    introGotIt: '了解',
    evtCreated: (where, assignee) =>
      `作成${where ? `（${where}）` : ''}${assignee ? `・${assignee} に割り当て` : ''}`,
    evtMovedTo: col => `${col} へ移動`,
    evtParentReopened: parent => `親 ${parent} が再オープン`,
    evtAssignedTo: assignee => `${assignee} に割り当て`,
    evtUnassigned: '割り当て解除',
    evtCommentBy: author => `${author} のコメント`,
    evtClaimedReview: 'レビューエージェントが取得',
    evtClaimedWorker: 'ワーカーが取得',
    evtWorkerStarted: 'ワーカー開始',
    evtCompleted: '完了',
    evtBlocked: 'ブロック — 人間の入力が必要',
    evtUnblocked: col => `ブロック解除${col ? ` → ${col}` : ' → Ready'}`,
    evtReclaimed: '再取得 — キューに戻しました',
    evtSpecified: 'トリアージエージェントが仕様を作成',
    evtPromoted: '依存関係が完了 — Ready に昇格',
    evtScheduled: '後で実行するようスケジュール',
    evtArchived: 'アーカイブ済み',
    evtReprioritized: priority => `優先度を ${priority} に設定`,
    someone: '誰か',
    metaPriority: '優先度',
    metaTenant: 'テナント',
    metaCreatedBy: '作成者',
    metaCreated: '作成',
    metaWorkerPid: 'ワーカー PID',
    readyUnassignedTitle: 'Ready ですが未割り当て — このカードは実行されません。',
    readyUnassignedBody:
      'ディスパッチャは担当のある Ready カードのみ取得します。上の担当フィールドでプロフィールを選ぶ（またはオーケストレーション設定でデフォルトの担当を設定する）と、1分以内に実行されます。',
    diagnosticsN: n => `診断・${n}`,
    commandCopied: 'コマンドをコピーしました',
    description: '説明',
    editDescription: '説明を編集',
    cancelEdit: '編集をキャンセル',
    noDescription: 'まだ説明はありません。',
    result: '結果',
    latestSummary: '最新のサマリー',
    dependencies: '依存関係',
    blockedBy: 'ブロック元',
    blocks: 'ブロック先',
    comments: n => `コメント・${n}`,
    commentsHelpRunning:
      'このタスクは実行中です。あなたのメモは数秒以内にワーカーの現在のターンに取り込まれます — ブロック/解除の操作は不要です。「メモを付けて再キュー」を選ぶと、メモを文脈に含めてタスクを最初からやり直します。',
    commentsHelp: 'コメントはタスクのスレッドに追加されます。ワーカーがタスクを取得すると、文脈の一部として読み込みます。',
    send: '送信',
    comment: 'コメント',
    messageWorker: '実行中のワーカーにメッセージ…',
    addComment: 'コメントを追加…',
    deliveredLive: '数秒以内に実行中のワーカーへ届きます。',
    requeueWithNote: 'メモを付けて再キュー',
    notePosted: 'メモを投稿しました — ワーカーを再キューしました',
    activity: n => `アクティビティ・${n}`,
    runs: n => `実行・${n}`,
    workerLog: 'ワーカーログ',
    workerLogTail: 'ワーカーログ・末尾',
    attachments: n => `添付・${n}`,
    noAttachments: 'まだ添付はありません。',
    uploadAttachment: '添付をアップロード',
    taskActions: 'タスクの操作',
    copyTaskId: 'タスク ID をコピー',
    copyTitle: 'タイトルをコピー',
    copiedId: id => `${id} をコピーしました`,
    copiedTitle: 'タイトルをコピーしました',
    archiveTask: 'タスクをアーカイブ',
    deleteTask: 'タスクを削除',
    close: '閉じる',
    working: '作業中',
    board: 'ボード',
    newBoard: '新しいボード',
    newBoardDots: '新しいボード…',
    boardSettings: 'ボード設定…',
    boardSettingsFor: name => `ボード設定 — ${name}`,
    name: '名前',
    boardNamePlaceholder: 'ボード名',
    slug: slug => `slug: ${slug}`,
    project: 'プロジェクト',
    noProject: 'プロジェクトなし（スクラッチのサンドボックス）',
    projectHintPre:
      '新しいタスクはプロジェクトのリポジトリで実行されます（タスクごとに worktree）。各タスクは作成時にワークスペースを上書きできます。プロジェクトの管理は ',
    projectHintCmd: 'hermes project',
    createBoard: 'ボードを作成',
    orchestratorProfile: 'オーケストレータープロフィール',
    defaultAssignee: 'デフォルトの担当',
    defaultParen: '（既定）',
    autoDecompose: 'トリアージタスクを自動分解',
    profileDescriptions: 'プロフィールの説明',
    profileDescriptionsHint:
      '説明はデコンポーザーのルーティングを導きます。補助モデルで自動生成するか、自分で書いてください。',
    profileGoodAt: 'このプロフィールの得意分野は？',
    auto: '自動'
  }

const zh: KanbanMessages = {
    nav: '看板',
    openBoard: '看板：打开面板',
    countTip: (running, ready) => `看板 — 运行中 ${running}、就绪 ${ready}`,
    col: {
      triage: { label: '分诊', help: '原始想法 — 由细化代理整理出规格。' },
      todo: { label: '待办', help: '等待依赖，或未分配。' },
      scheduled: { label: '已排期', help: '等待预定时间到来。' },
      ready: { label: '就绪', help: '依赖已满足 — 分配一个配置档，调度器即会运行它。' },
      running: { label: '运行中', help: '已被工作单元领取 — 有代理在处理。由调度器设置。' },
      blocked: { label: '受阻', help: '工作单元需要人工输入。' },
      review: { label: '审查', help: '审查代理正在检查工作。由调度器设置。' },
      done: { label: '完成', help: '已完成；依赖它的子任务变为就绪。' },
      archived: { label: '已归档', help: '从默认面板视图中隐藏。' }
    },
    locked: {
      review: '审查状态由调度器在审查代理领取卡片时设置。',
      running: '运行中由调度器在工作单元领取卡片时设置。',
      scheduled: '排期需要唤醒时间 — 由代理设置；无法拖入。'
    },
    arcRunning: '有代理正在处理它。',
    arcStale: '已领取，但超过 2 分钟没有工作单元心跳 — 调度器将重新领取。',
    title: '看板',
    orchestrationSettings: '编排设置',
    newTask: '新建任务',
    filterCards: '筛选卡片…',
    noMatch: '没有符合筛选条件的任务',
    noTasks: '此面板暂无任务',
    open: '打开',
    select: '选择（⌘点击）',
    deselect: '取消选择',
    moveTo: label => `移动到 ${label}`,
    delete: '删除',
    reviewChecking: '审查代理正在检查已完成的工作。',
    attachedTip: name => `${name} 已接手 — 调度器将在下一个周期（≤1 分钟）移交。`,
    orchestratorTip: name => `${name}（编排者）将在下一个周期领取并撰写规格。`,
    autoAssignTip: name => `将在下一个调度周期自动分配给“${name}”（kanban.default_assignee）。`,
    wontRun: '不会运行',
    wontRunTip: '就绪卡片只有在分配了配置档后才会运行。打开卡片设置负责人，或在编排设置中配置默认负责人。',
    noHeartbeat: '无心跳',
    expand: label => `展开 ${label}`,
    collapse: label => `折叠 ${label}`,
    newTaskIn: label => `在 ${label} 新建任务`,
    empty: '空',
    unassigned: '未分配',
    filters: '筛选',
    allProfiles: '所有配置档',
    allTenants: '所有租户',
    showArchived: '显示已归档',
    groupRunning: '按配置档分组运行中',
    nSelected: n => `已选择 ${n} 个`,
    moveToShort: '移动到',
    assign: '分配',
    unassignAction: '取消分配',
    archive: '归档',
    clearSelection: '清除选择（Esc）',
    refused: '被拒绝',
    bulkFailed: (failed, total, err) => `${total} 个中有 ${failed} 个失败 — ${err}。失败的卡片仍保持选中。`,
    titlePlaceholderTriage: '大致想法 — 细化代理会补全',
    titlePlaceholder: '标题',
    descPlaceholder: '描述（可选）',
    priority: '优先级',
    workspace: '工作区',
    boardDefaultSuffix: '・面板默认',
    workspaceOverride: '工作区路径（可选覆盖）',
    workspaceInherit: '继承面板的项目目录',
    workspaceInheritDir: dir => `留空则继承 ${dir}`,
    workspaceInheritGeneric: '留空则继承面板的项目目录。',
    assignee: '负责人',
    defaultOption: name => `${name}（默认）`,
    parkedOption: '未分配（搁置 — 不会运行）',
    skills: '技能（逗号分隔）',
    skillsPlaceholder: 'translation, github',
    parent: '父任务（完成前会阻塞）',
    noParent: '— 无父任务 —',
    goalMode: '目标模式（工作单元循环直到评判代理认可完成）',
    creating: '创建中…',
    createTask: '创建任务',
    cancel: '取消',
    save: '保存',
    estimate: '估算',
    estimateEffort: '估算工作量',
    estimating: '估算中…',
    reEstimate: '重新估算',
    makesModelCall: '会调用模型',
    estimateTip: '由辅助模型对令牌数和复杂度的粗略估算 — 会调用模型。',
    estimateTipLong: '快速调用辅助模型来估算令牌数和复杂度。仅供参考，并非账单。',
    roughEstimate: '粗略估算',
    tokUnit: 'tok',
    couldNotEstimate: '无法估算',
    complexity: { S: '小', M: '中', L: '大' },
    introBody:
      '卡片不由你运行，而是由代理运行。把带有负责人的卡片放入“就绪”，代理会在一分钟内领取。没有负责人就不会运行。分诊：代理先把想法改写成合适的任务。待办：等待其他卡片。已排期：等待计时器。运行中与审查：这是代理的通道，请勿插手。受阻：正在等你。结果会回到卡片上。',
    introGotIt: '知道了',
    evtCreated: (where, assignee) =>
      `已创建${where ? `（${where}）` : ''}${assignee ? `・分配给 ${assignee}` : ''}`,
    evtMovedTo: col => `移动到 ${col}`,
    evtParentReopened: parent => `父任务 ${parent} 已重新打开`,
    evtAssignedTo: assignee => `分配给 ${assignee}`,
    evtUnassigned: '取消分配',
    evtCommentBy: author => `${author} 的评论`,
    evtClaimedReview: '被审查代理领取',
    evtClaimedWorker: '被工作单元领取',
    evtWorkerStarted: '工作单元已启动',
    evtCompleted: '已完成',
    evtBlocked: '受阻 — 需要人工输入',
    evtUnblocked: col => `已解除阻塞${col ? ` → ${col}` : ' → 就绪'}`,
    evtReclaimed: '已重新领取 — 已放回队列',
    evtSpecified: '分诊代理已撰写规格',
    evtPromoted: '依赖已完成 — 提升为就绪',
    evtScheduled: '已排期稍后运行',
    evtArchived: '已归档',
    evtReprioritized: priority => `优先级设为 ${priority}`,
    someone: '某人',
    metaPriority: '优先级',
    metaTenant: '租户',
    metaCreatedBy: '创建者',
    metaCreated: '创建于',
    metaWorkerPid: '工作单元 PID',
    readyUnassignedTitle: '就绪但未分配 — 这张卡片永远不会运行。',
    readyUnassignedBody:
      '调度器只领取有负责人的就绪卡片。在上面的负责人字段选择一个配置档（或在编排设置中设置默认负责人），它会在一分钟内运行。',
    diagnosticsN: n => `诊断・${n}`,
    commandCopied: '命令已复制',
    description: '描述',
    editDescription: '编辑描述',
    cancelEdit: '取消编辑',
    noDescription: '暂无描述。',
    result: '结果',
    latestSummary: '最新摘要',
    dependencies: '依赖关系',
    blockedBy: '受阻于',
    blocks: '阻塞',
    comments: n => `评论・${n}`,
    commentsHelpRunning:
      '此任务正在运行。你的备注会在几秒内融入工作单元当前的回合 — 无需阻塞/解除操作。选择“附带备注重新入队”则会带着你的备注从头重跑任务。',
    commentsHelp: '评论会添加到任务讨论串。工作单元领取任务时会将其作为上下文的一部分读取。',
    send: '发送',
    comment: '评论',
    messageWorker: '给运行中的工作单元发消息…',
    addComment: '添加评论…',
    deliveredLive: '几秒内送达运行中的工作单元。',
    requeueWithNote: '附带备注重新入队',
    notePosted: '备注已发布 — 工作单元已重新入队',
    activity: n => `活动・${n}`,
    runs: n => `运行・${n}`,
    workerLog: '工作单元日志',
    workerLogTail: '工作单元日志・末尾',
    attachments: n => `附件・${n}`,
    noAttachments: '暂无附件。',
    uploadAttachment: '上传附件',
    taskActions: '任务操作',
    copyTaskId: '复制任务 ID',
    copyTitle: '复制标题',
    copiedId: id => `已复制 ${id}`,
    copiedTitle: '已复制标题',
    archiveTask: '归档任务',
    deleteTask: '删除任务',
    close: '关闭',
    working: '进行中',
    board: '面板',
    newBoard: '新建面板',
    newBoardDots: '新建面板…',
    boardSettings: '面板设置…',
    boardSettingsFor: name => `面板设置 — ${name}`,
    name: '名称',
    boardNamePlaceholder: '面板名称',
    slug: slug => `slug: ${slug}`,
    project: '项目',
    noProject: '无项目（临时沙箱）',
    projectHintPre: '新任务将在项目的仓库中运行（每个任务一个 worktree）；每个任务在创建时仍可覆盖其工作区。管理项目请使用 ',
    projectHintCmd: 'hermes project',
    createBoard: '创建面板',
    orchestratorProfile: '编排者配置档',
    defaultAssignee: '默认负责人',
    defaultParen: '（默认）',
    autoDecompose: '自动分解分诊任务',
    profileDescriptions: '配置档说明',
    profileDescriptionsHint: '说明用于引导分解器的路由。可用辅助模型自动生成，或自行填写。',
    profileGoodAt: '这个配置档擅长什么？',
    auto: '自动'
  }

const zhHant: KanbanMessages = {
    nav: '看板',
    openBoard: '看板：開啟面板',
    countTip: (running, ready) => `看板 — 執行中 ${running}、就緒 ${ready}`,
    col: {
      triage: { label: '分類', help: '原始想法 — 由細化代理整理出規格。' },
      todo: { label: '待辦', help: '等待相依項目，或未指派。' },
      scheduled: { label: '已排程', help: '等待預定時間到來。' },
      ready: { label: '就緒', help: '相依項目已滿足 — 指派一個設定檔，排程器便會執行它。' },
      running: { label: '執行中', help: '已被工作單元領取 — 有代理在處理。由排程器設定。' },
      blocked: { label: '受阻', help: '工作單元需要人工輸入。' },
      review: { label: '審查', help: '審查代理正在檢查工作。由排程器設定。' },
      done: { label: '完成', help: '已完成；相依它的子任務變為就緒。' },
      archived: { label: '已封存', help: '從預設面板檢視中隱藏。' }
    },
    locked: {
      review: '審查狀態由排程器在審查代理領取卡片時設定。',
      running: '執行中由排程器在工作單元領取卡片時設定。',
      scheduled: '排程需要喚醒時間 — 由代理設定；無法拖入。'
    },
    arcRunning: '有代理正在處理它。',
    arcStale: '已領取，但超過 2 分鐘沒有工作單元心跳 — 排程器將重新領取。',
    title: '看板',
    orchestrationSettings: '編排設定',
    newTask: '新增任務',
    filterCards: '篩選卡片…',
    noMatch: '沒有符合篩選條件的任務',
    noTasks: '此面板尚無任務',
    open: '開啟',
    select: '選取（⌘點擊）',
    deselect: '取消選取',
    moveTo: label => `移至 ${label}`,
    delete: '刪除',
    reviewChecking: '審查代理正在檢查已完成的工作。',
    attachedTip: name => `${name} 已接手 — 排程器將在下一個週期（≤1 分鐘）移交。`,
    orchestratorTip: name => `${name}（編排者）將在下一個週期領取並撰寫規格。`,
    autoAssignTip: name => `將在下一個排程週期自動指派給「${name}」（kanban.default_assignee）。`,
    wontRun: '不會執行',
    wontRunTip: '就緒卡片只有在指派了設定檔後才會執行。開啟卡片設定負責人，或在編排設定中設定預設負責人。',
    noHeartbeat: '無心跳',
    expand: label => `展開 ${label}`,
    collapse: label => `摺疊 ${label}`,
    newTaskIn: label => `在 ${label} 新增任務`,
    empty: '空',
    unassigned: '未指派',
    filters: '篩選',
    allProfiles: '所有設定檔',
    allTenants: '所有租戶',
    showArchived: '顯示已封存',
    groupRunning: '依設定檔分組執行中',
    nSelected: n => `已選取 ${n} 個`,
    moveToShort: '移至',
    assign: '指派',
    unassignAction: '取消指派',
    archive: '封存',
    clearSelection: '清除選取（Esc）',
    refused: '被拒絕',
    bulkFailed: (failed, total, err) => `${total} 個中有 ${failed} 個失敗 — ${err}。失敗的卡片仍保持選取。`,
    titlePlaceholderTriage: '大致想法 — 細化代理會補全',
    titlePlaceholder: '標題',
    descPlaceholder: '描述（選填）',
    priority: '優先順序',
    workspace: '工作區',
    boardDefaultSuffix: '・面板預設',
    workspaceOverride: '工作區路徑（選填覆寫）',
    workspaceInherit: '繼承面板的專案目錄',
    workspaceInheritDir: dir => `留空則繼承 ${dir}`,
    workspaceInheritGeneric: '留空則繼承面板的專案目錄。',
    assignee: '負責人',
    defaultOption: name => `${name}（預設）`,
    parkedOption: '未指派（擱置 — 不會執行）',
    skills: '技能（以逗號分隔）',
    skillsPlaceholder: 'translation, github',
    parent: '父任務（完成前會阻擋）',
    noParent: '— 無父任務 —',
    goalMode: '目標模式（工作單元循環直到評判代理認可完成）',
    creating: '建立中…',
    createTask: '建立任務',
    cancel: '取消',
    save: '儲存',
    estimate: '估算',
    estimateEffort: '估算工作量',
    estimating: '估算中…',
    reEstimate: '重新估算',
    makesModelCall: '會呼叫模型',
    estimateTip: '由輔助模型對 token 數與複雜度的粗略估算 — 會呼叫模型。',
    estimateTipLong: '快速呼叫輔助模型來估算 token 數與複雜度。僅供參考，並非帳單。',
    roughEstimate: '粗略估算',
    tokUnit: 'tok',
    couldNotEstimate: '無法估算',
    complexity: { S: '小', M: '中', L: '大' },
    introBody:
      '卡片不由你執行，而是由代理執行。把有負責人的卡片放入「就緒」，代理會在一分鐘內領取。沒有負責人就不會執行。分類：代理先把想法改寫成合適的任務。待辦：等待其他卡片。已排程：等待計時器。執行中與審查：這是代理的通道，請勿插手。受阻：正在等你。結果會回到卡片上。',
    introGotIt: '知道了',
    evtCreated: (where, assignee) =>
      `已建立${where ? `（${where}）` : ''}${assignee ? `・指派給 ${assignee}` : ''}`,
    evtMovedTo: col => `移至 ${col}`,
    evtParentReopened: parent => `父任務 ${parent} 已重新開啟`,
    evtAssignedTo: assignee => `指派給 ${assignee}`,
    evtUnassigned: '取消指派',
    evtCommentBy: author => `${author} 的留言`,
    evtClaimedReview: '被審查代理領取',
    evtClaimedWorker: '被工作單元領取',
    evtWorkerStarted: '工作單元已啟動',
    evtCompleted: '已完成',
    evtBlocked: '受阻 — 需要人工輸入',
    evtUnblocked: col => `已解除阻擋${col ? ` → ${col}` : ' → 就緒'}`,
    evtReclaimed: '已重新領取 — 已放回佇列',
    evtSpecified: '分類代理已撰寫規格',
    evtPromoted: '相依項目已完成 — 提升為就緒',
    evtScheduled: '已排程稍後執行',
    evtArchived: '已封存',
    evtReprioritized: priority => `優先順序設為 ${priority}`,
    someone: '某人',
    metaPriority: '優先順序',
    metaTenant: '租戶',
    metaCreatedBy: '建立者',
    metaCreated: '建立於',
    metaWorkerPid: '工作單元 PID',
    readyUnassignedTitle: '就緒但未指派 — 這張卡片永遠不會執行。',
    readyUnassignedBody:
      '排程器只領取有負責人的就緒卡片。在上方的負責人欄位選擇一個設定檔（或在編排設定中設定預設負責人），它會在一分鐘內執行。',
    diagnosticsN: n => `診斷・${n}`,
    commandCopied: '指令已複製',
    description: '描述',
    editDescription: '編輯描述',
    cancelEdit: '取消編輯',
    noDescription: '尚無描述。',
    result: '結果',
    latestSummary: '最新摘要',
    dependencies: '相依關係',
    blockedBy: '受阻於',
    blocks: '阻擋',
    comments: n => `留言・${n}`,
    commentsHelpRunning:
      '此任務正在執行。你的備註會在幾秒內融入工作單元目前的回合 — 無需阻擋/解除操作。選擇「附上備註重新排入佇列」則會帶著你的備註從頭重跑任務。',
    commentsHelp: '留言會加入任務討論串。工作單元領取任務時會將其作為脈絡的一部分讀取。',
    send: '傳送',
    comment: '留言',
    messageWorker: '傳訊給執行中的工作單元…',
    addComment: '新增留言…',
    deliveredLive: '幾秒內送達執行中的工作單元。',
    requeueWithNote: '附上備註重新排入佇列',
    notePosted: '備註已發布 — 工作單元已重新排入佇列',
    activity: n => `活動・${n}`,
    runs: n => `執行・${n}`,
    workerLog: '工作單元日誌',
    workerLogTail: '工作單元日誌・末尾',
    attachments: n => `附件・${n}`,
    noAttachments: '尚無附件。',
    uploadAttachment: '上傳附件',
    taskActions: '任務操作',
    copyTaskId: '複製任務 ID',
    copyTitle: '複製標題',
    copiedId: id => `已複製 ${id}`,
    copiedTitle: '已複製標題',
    archiveTask: '封存任務',
    deleteTask: '刪除任務',
    close: '關閉',
    working: '進行中',
    board: '面板',
    newBoard: '新增面板',
    newBoardDots: '新增面板…',
    boardSettings: '面板設定…',
    boardSettingsFor: name => `面板設定 — ${name}`,
    name: '名稱',
    boardNamePlaceholder: '面板名稱',
    slug: slug => `slug: ${slug}`,
    project: '專案',
    noProject: '無專案（暫存沙箱）',
    projectHintPre: '新任務將在專案的儲存庫中執行（每個任務一個 worktree）；每個任務在建立時仍可覆寫其工作區。管理專案請使用 ',
    projectHintCmd: 'hermes project',
    createBoard: '建立面板',
    orchestratorProfile: '編排者設定檔',
    defaultAssignee: '預設負責人',
    defaultParen: '（預設）',
    autoDecompose: '自動分解分類任務',
    profileDescriptions: '設定檔說明',
    profileDescriptionsHint: '說明用於引導分解器的路由。可用輔助模型自動產生，或自行填寫。',
    profileGoodAt: '這個設定檔擅長什麼？',
    auto: '自動'
  }

/** Registered via `ctx.i18n.register` at plugin load (disposer tracked). */
export const KANBAN_LOCALES: PluginLocaleBundles = { en, ja, zh, 'zh-hant': zhHant }

// Bind the message SHAPE to a plugin translator: string leaves resolve now,
// function leaves forward their args through t(path, …). One tiny generic
// instead of a hand-written accessor per key.
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

export type KanbanText = Bound<KanbanMessages>

/** The kanban strings for the active locale — one hook every component reads. */
export function useKanban(): KanbanText {
  const t = usePluginI18n('kanban')

  return useMemo(() => bind(t, en), [t])
}

// Column labels/help live in i18n; unknown backend statuses fall back to the id.
export const columnLabel = (k: KanbanText, name: string) => k.col[name as keyof KanbanText['col']]?.label ?? name
export const columnHelp = (k: KanbanText, name: string) => k.col[name as keyof KanbanText['col']]?.help ?? ''
export const lockedReason = (k: KanbanText, name: string) => k.locked[name as keyof KanbanText['locked']] ?? ''
