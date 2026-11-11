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
  /** Command label — shows in the ⌘K palette AND as the keybind panel row,
   *  so it carries the "Kanban: " prefix the palette convention wants. */
  newTaskCommand: string
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
  openSession: (sessionId: string) => string
  select: (modifier: string) => string
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
  model: string
  modelInherit: string
  modelClear: string
  modelHint: string
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
  /** #124391 — blocked-card detail: why it blocked and what it cost. */
  blockKindTip: (kind: string) => string
  blockReason: string
  blockRecurrences: string
  blockRecurrencesTip: string
  consecutiveFailures: string
  lastFailureError: string
  unblockedMessage: (id: string) => string
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
  close: string
  working: string
  // board switcher
  board: string
  newBoard: string
  /** Tooltip on the page-header trigger — names the ACTION, since the visible
   *  text is the board's own name and reads as a static label otherwise. */
  switchBoard: string
  pinBoard: string
  unpinBoard: string
  /** Accessible name of the pinned-tab strip. */
  pinnedBoards: string
  pinnedBoardsHint: string
  newBoardDots: string
  // Menu labels are bare verbs — the board they act on is the one named in the
  // switcher's trigger. The nouns come back for the native file-dialog and
  // in-app dialog titles, which stand alone.
  exportDots: string
  importDots: string
  renameDots: string
  settingsDots: string
  exportBoardTitle: string
  importBoardTitle: string
  boardExported: (path: string) => string
  boardImported: (name: string) => string
  boardImportedAs: (slug: string) => string
  renameBoardTitle: string
  deleteBoardTitle: (name: string) => string
  deleteBoardConfirm: string
  boardArchived: (path: string) => string
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
  // gc
  gc: string
  gcRunning: string
  gcSuccess: (ws: number, events: number, logs: number) => string
  gcFailed: string
  // native/toast notifications for terminal worker events (completion-notify)
  notify: {
    completedTitle: string
    blockedTitle: string
    blockLoopTitle: string
    gaveUpTitle: string
    /** Body for gave_up — the raw worker error rides in the toast `detail`. */
    gaveUpBody: string
    crashedTitle: string
    timedOutTitle: string
    openKanban: string
    artifacts: (n: number) => string
  }
}

export const en: KanbanMessages = {
  nav: 'Kanban',
  openBoard: 'Kanban: Open board',
  newTaskCommand: 'Kanban: New task',
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
  openSession: sessionId => `Open session (${sessionId.slice(-8)})`,
  select: modifier => `Select (${modifier}-click)`,
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
  model: 'Model',
  modelInherit: 'Profile default',
  modelClear: 'Clear model override',
  modelHint: 'Runs this task on a specific model and thinking depth. Unset uses the assigned profile’s own.',
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
  evtCreated: (where, assignee) =>
    `created${where ? ` in ${where}` : ''}${assignee ? ` · assigned to ${assignee}` : ''}`,
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
  blockKindTip: kind =>
    kind === 'needs_input'
      ? 'The worker asked for human input.'
      : kind === 'capability'
        ? 'The worker hit a capability wall it cannot get past.'
        : kind === 'transient'
          ? 'The worker failed on a transient error.'
          : 'The worker reported a dependency it must wait on.',
  blockReason: 'Block kind',
  blockRecurrences: 'Block recurrences',
  blockRecurrencesTip: 'Times this task re-blocked for the same reason after a human unblock.',
  consecutiveFailures: 'Consecutive failures',
  lastFailureError: 'Last failure',
  unblockedMessage: id => `Unblocked ${id}. Task is ready for the next tick.`,
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
  close: 'Close',
  working: 'working',
  board: 'Board',
  newBoard: 'New board',
  switchBoard: 'Switch board',
  pinBoard: 'Pin board',
  unpinBoard: 'Unpin board',
  pinnedBoards: 'Pinned boards',
  pinnedBoardsHint: 'Drag to reorder, or press Alt + Left/Right',
  newBoardDots: 'New board…',
  exportDots: 'Export…',
  importDots: 'Import…',
  renameDots: 'Rename…',
  settingsDots: 'Settings…',
  exportBoardTitle: 'Export board…',
  importBoardTitle: 'Import board…',
  boardExported: path => `Board exported to ${path}`,
  boardImported: name => `Imported ${name}`,
  boardImportedAs: slug => `That name was taken — imported as ${slug}`,
  renameBoardTitle: 'Rename board',
  deleteBoardTitle: name => `Delete "${name}"?`,
  deleteBoardConfirm: 'The board is archived, not erased — its tasks and attachments stay on disk and can be restored.',
  boardArchived: path => `Board archived to ${path}`,
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
  auto: 'Auto',
  // gc
  gc: 'Garbage collect',
  gcRunning: 'Running GC sweep…',
  gcSuccess: (ws, events, logs) => `GC complete: ${ws} workspace(s), ${events} event(s), ${logs} log(s) removed`,
  gcFailed: 'GC failed — check the logs',
  notify: {
    completedTitle: 'Task completed',
    blockedTitle: 'Task blocked — needs your input',
    blockLoopTitle: 'Task routed to triage — needs a decision',
    gaveUpTitle: 'Task stopped',
    gaveUpBody: 'Hermes couldn’t finish this task. Open Kanban to see why and reassign it.',
    crashedTitle: 'Task hit a problem — Hermes will retry it automatically',
    timedOutTitle: 'Task took too long — Hermes will retry it automatically',
    openKanban: 'Open Kanban',
    artifacts: (n: number) => `${n} artifacts`
  }
}

const ja: KanbanMessages = {
  nav: 'カンバン',
  openBoard: 'カンバン: ボードを開く',
  newTaskCommand: 'カンバン: 新しいタスク',
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
  openSession: sessionId => `セッションを開く (${sessionId.slice(-8)})`,
  select: modifier => `選択（${modifier}クリック）`,
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
  model: 'モデル',
  modelInherit: 'プロファイル既定',
  modelClear: 'モデル指定を解除',
  modelHint: 'このタスクを特定のモデルと思考深度で実行します。未設定なら担当プロファイルの設定を使用します。',
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
  evtCreated: (where, assignee) => `作成${where ? `（${where}）` : ''}${assignee ? `・${assignee} に割り当て` : ''}`,
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
  blockKindTip: kind =>
    kind === 'needs_input'
      ? 'ワーカーが人間の入力を求めています。'
      : kind === 'capability'
        ? 'ワーカーが越えられない能力の壁に当たりました。'
        : kind === 'transient'
          ? 'ワーカーが一時的なエラーで失敗しました。'
          : 'ワーカーが待機すべき依存関係を報告しました。',
  blockReason: 'ブロック種別',
  blockRecurrences: 'ブロック回数',
  blockRecurrencesTip: '人間がブロック解除した後、同じ理由で再ブロックされた回数。',
  consecutiveFailures: '連続失敗',
  lastFailureError: '直近の失敗',
  unblockedMessage: id => `${id} をブロック解除しました。次のティックで実行できます。`,
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
  commentsHelp:
    'コメントはタスクのスレッドに追加されます。ワーカーがタスクを取得すると、文脈の一部として読み込みます。',
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
  close: '閉じる',
  working: '作業中',
  board: 'ボード',
  newBoard: '新しいボード',
  switchBoard: 'ボードを切り替え',
  pinBoard: 'ボードを固定',
  unpinBoard: 'ボードの固定を解除',
  pinnedBoards: '固定したボード',
  pinnedBoardsHint: 'ドラッグ、または Alt + 左右キーで並べ替え',
  newBoardDots: '新しいボード…',
  exportDots: 'エクスポート…',
  importDots: 'インポート…',
  renameDots: '名前を変更…',
  settingsDots: '設定…',
  exportBoardTitle: 'ボードをエクスポート…',
  importBoardTitle: 'ボードをインポート…',
  boardExported: path => `ボードを ${path} にエクスポートしました`,
  boardImported: name => `${name} をインポートしました`,
  boardImportedAs: slug => `その名前は使用中のため ${slug} としてインポートしました`,
  renameBoardTitle: 'ボード名を変更',
  deleteBoardTitle: name => `「${name}」を削除しますか？`,
  deleteBoardConfirm: 'ボードは消去されずアーカイブされます。タスクと添付ファイルはディスクに残り、復元できます。',
  boardArchived: path => `ボードを ${path} にアーカイブしました`,
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
  auto: '自動',
  // gc
  gc: 'ごみ収集',
  gcRunning: 'GC スイープを実行中…',
  gcSuccess: (ws, events, logs) => `GC完了: ${ws} 件のワークスペース、${events} 件の中イベント、${logs} 件のログを削除`,
  gcFailed: 'GC に失敗しました — ログを確認してください',
  notify: {
    completedTitle: 'タスク完了',
    blockedTitle: 'タスクがブロック中 — 入力が必要です',
    blockLoopTitle: 'タスクをトリアージへ移動 — 判断が必要です',
    gaveUpTitle: 'タスクが停止しました',
    gaveUpBody: 'Hermes はこのタスクを完了できませんでした。かんばんを開いて原因を確認し、再割り当てしてください。',
    crashedTitle: 'タスクで問題が発生 — Hermes が自動で再試行します',
    timedOutTitle: 'タスクに時間がかかりすぎました — Hermes が自動で再試行します',
    openKanban: 'かんばんを開く',
    artifacts: (n: number) => `成果物 ${n} 件`
  }
}

/** Registered via `ctx.i18n.register` at plugin load (disposer tracked). */
export const KANBAN_LOCALES: PluginLocaleBundles = { en, ja }

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
