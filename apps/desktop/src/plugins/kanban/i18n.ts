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

const ko: KanbanMessages = {
  switchBoard: '보드 전환',
  blockKindTip: kind =>
    kind === 'needs_input'
      ? '작업자가 사용자 입력을 기다리고 있습니다.'
      : kind === 'capability'
        ? '작업자가 현재 기능으로 해결할 수 없는 문제에 부딪혔습니다.'
        : kind === 'transient'
          ? '일시적인 오류로 작업이 실패했습니다.'
          : '작업자가 선행 조건이 충족되기를 기다리고 있습니다.',
  blockReason: '진행 중단 유형',
  blockRecurrences: '같은 원인으로 중단된 횟수',
  blockRecurrencesTip: '사용자가 중단을 해제한 뒤 같은 원인으로 다시 중단된 횟수입니다.',
  consecutiveFailures: '연속 실패 횟수',
  lastFailureError: '최근 실패 원인',
  unblockedMessage: id => `${id}의 중단을 해제했습니다. 다음 작업 확인 주기에 실행할 준비가 되었습니다.`,

  nav: '칸반',
  openBoard: '칸반: 보드 열기',
  newTaskCommand: '칸반: 새 작업',
  countTip: (running, ready) => `칸반 — 실행 중 ${running}개, 준비 ${ready}개`,
  col: {
    triage: { label: '분류', help: '아직 구체화되지 않은 아이디어입니다. 명세 작성 에이전트가 내용을 보완합니다.' },
    todo: { label: '할 일', help: '선행 작업을 기다리거나 담당자가 지정되지 않은 작업입니다.' },
    scheduled: { label: '예약됨', help: '예약한 실행 시각을 기다리는 작업입니다.' },
    ready: { label: '준비', help: '선행 작업이 완료되었습니다. 프로필을 지정하면 작업 배정기가 실행합니다.' },
    running: {
      label: '실행 중',
      help: '작업 실행기가 작업을 맡아 에이전트가 처리 중입니다. 작업 배정기가 설정하는 상태입니다.'
    },
    blocked: { label: '막힘', help: '작업 실행기가 사용자 입력을 요청했습니다.' },
    review: { label: '검토', help: '검토 에이전트가 결과를 확인 중입니다. 작업 배정기가 설정하는 상태입니다.' },
    done: { label: '완료', help: '완료된 작업입니다. 이 작업을 기다리던 하위 작업이 준비 상태가 됩니다.' },
    archived: { label: '보관됨', help: '기본 보드 화면에서 숨긴 작업입니다.' }
  },
  locked: {
    review: '검토 에이전트가 카드를 맡으면 작업 배정기가 검토 상태로 변경합니다.',
    running: '작업 실행기가 카드를 맡으면 작업 배정기가 실행 중 상태로 변경합니다.',
    scheduled: '예약 상태에는 실행 시각이 필요합니다. 에이전트가 설정하는 상태로, 카드를 끌어서 옮길 수 없습니다.'
  },
  arcRunning: '에이전트가 현재 이 작업을 처리 중입니다.',
  arcStale: '작업을 맡았지만 2분 넘게 실행기의 응답이 없습니다. 작업 배정기가 회수할 예정입니다.',
  title: '칸반',
  orchestrationSettings: '작업 조율 설정',
  newTask: '새 작업',
  filterCards: '카드 필터링…',
  noMatch: '필터에 맞는 작업이 없습니다',
  noTasks: '이 보드에 작업이 없습니다',
  open: '열기',
  select: modifier => `선택 (${modifier}+클릭)`,
  deselect: '선택 해제',
  moveTo: label => `${label}(으)로 이동`,
  delete: '삭제',
  reviewChecking: '검토 에이전트가 완료된 작업을 확인 중입니다.',
  attachedTip: name => `${name} 연결됨 — 작업 배정기가 다음 확인 주기(1분 이내)에 전달합니다.`,
  orchestratorTip: name => `${name}(작업 조율 에이전트)이 다음 확인 주기에 작업을 맡아 명세를 작성합니다.`,
  autoAssignTip: name => `다음 배정 주기에 “${name}”(kanban.default_assignee)에게 자동 배정됩니다.`,
  wontRun: '실행되지 않음',
  wontRunTip:
    '준비 상태의 카드는 프로필을 지정해야 실행됩니다. 카드를 열어 담당자를 지정하거나 작업 조율 설정에서 기본 담당자를 설정하세요.',
  noHeartbeat: '응답 신호 없음',
  expand: label => `${label} 펼치기`,
  collapse: label => `${label} 접기`,
  newTaskIn: label => `${label}에 새 작업`,
  empty: '비어 있음',
  unassigned: '미지정',
  filters: '필터',
  allProfiles: '모든 프로필',
  allTenants: '모든 테넌트',
  showArchived: '보관된 작업 표시',
  groupRunning: '실행 중인 작업을 프로필별로 묶기',
  nSelected: n => `${n}개 선택됨`,
  moveToShort: '이동',
  assign: '담당자 지정',
  unassignAction: '담당자 해제',
  archive: '보관',
  clearSelection: '선택 해제 (Esc)',
  refused: '거부됨',
  bulkFailed: (failed, total, err) => `${total}개 중 ${failed}개 실패 — ${err}. 실패한 카드는 선택된 상태로 남습니다.`,
  titlePlaceholderTriage: '아이디어를 적어 주세요. 명세 작성 에이전트가 구체화합니다',
  titlePlaceholder: '제목',
  descPlaceholder: '설명 (선택 사항)',
  priority: '우선순위',
  workspace: '작업 공간',
  boardDefaultSuffix: ' · 보드 기본값',
  workspaceOverride: '작업 공간 경로 (기본값 대신 사용할 경우 입력)',
  model: '모델',
  modelInherit: '프로필 기본값',
  modelClear: '모델 재정의 해제',
  modelHint: '이 작업에 사용할 모델과 추론 수준을 지정합니다. 지정하지 않으면 담당 프로필의 설정을 사용합니다.',
  workspaceInherit: '보드의 프로젝트 디렉터리를 사용합니다',
  workspaceInheritDir: dir => `비워 두면 ${dir}을(를) 사용합니다`,
  workspaceInheritGeneric: '비워 두면 보드의 프로젝트 디렉터리를 사용합니다.',
  assignee: '담당자',
  defaultOption: name => `${name} (기본값)`,
  parkedOption: '미지정 (대기 — 실행되지 않음)',
  skills: '스킬 (쉼표로 구분)',
  skillsPlaceholder: 'translation, github',
  parent: '상위 작업 (완료될 때까지 대기)',
  noParent: '— 상위 작업 없음 —',
  goalMode: '목표 모드 (검증 에이전트가 완료를 확인할 때까지 반복)',
  creating: '생성 중…',
  createTask: '작업 만들기',
  cancel: '취소',
  save: '저장',
  estimate: '예상 사용량',
  estimateEffort: '작업량 추정',
  estimating: '추정 중…',
  reEstimate: '다시 추정',
  makesModelCall: '모델을 호출합니다',
  estimateTip: '보조 모델로 토큰 사용량과 복잡도를 대략 추정합니다. 모델 호출이 발생합니다.',
  estimateTipLong:
    '보조 모델을 짧게 호출해 토큰 사용량과 복잡도를 추정합니다. 실제 청구 금액이 아닌 참고용 추정치입니다.',
  roughEstimate: '대략적인 추정치',
  tokUnit: '토큰',
  couldNotEstimate: '추정하지 못했습니다',
  complexity: { S: '낮음', M: '보통', L: '높음' },
  introBody:
    '카드는 에이전트가 실행합니다. 담당자를 지정하고 준비 상태로 옮기면 1분 이내에 에이전트가 작업을 맡습니다. 담당자가 없으면 실행되지 않습니다. 분류: 에이전트가 아이디어를 구체적인 작업으로 정리합니다. 할 일: 다른 카드의 완료를 기다립니다. 예약됨: 예약 시각을 기다립니다. 실행 중과 검토: 에이전트가 관리하는 단계입니다. 막힘: 사용자 입력을 기다립니다. 결과는 카드에서 확인할 수 있습니다.',
  introGotIt: '확인',
  evtCreated: (where, assignee) => `${where ? `${where}에 ` : ''}생성됨${assignee ? ` · 담당자: ${assignee}` : ''}`,
  evtMovedTo: col => `${col}(으)로 이동됨`,
  evtParentReopened: parent => `상위 작업 ${parent} 다시 열림`,
  evtAssignedTo: assignee => `담당자 지정: ${assignee}`,
  evtUnassigned: '담당자 해제됨',
  evtCommentBy: author => `${author}의 댓글`,
  evtClaimedReview: '검토 에이전트가 작업을 맡음',
  evtClaimedWorker: '작업 실행기가 작업을 맡음',
  evtWorkerStarted: '작업 실행기 시작됨',
  evtCompleted: '완료됨',
  evtBlocked: '막힘 — 사용자 입력 필요',
  evtUnblocked: col => `막힘 해제 → ${col || '준비'}`,
  evtReclaimed: '회수됨 — 대기열로 돌아감',
  evtSpecified: '분류 에이전트가 명세를 작성함',
  evtPromoted: '선행 작업 완료 — 준비 상태로 변경됨',
  evtScheduled: '나중에 실행하도록 예약됨',
  evtArchived: '보관됨',
  evtReprioritized: priority => `우선순위 변경: ${priority}`,
  someone: '누군가',
  metaPriority: '우선순위',
  metaTenant: '테넌트',
  metaCreatedBy: '생성자',
  metaCreated: '생성 시각',
  metaWorkerPid: '작업 실행기 PID',
  readyUnassignedTitle: '준비 상태이지만 담당자가 없어 실행되지 않습니다.',
  readyUnassignedBody:
    '작업 배정기는 담당자가 지정된 준비 상태의 카드만 실행합니다. 위의 담당자 항목에서 프로필을 선택하거나 작업 조율 설정에서 기본 담당자를 지정하면 1분 이내에 실행됩니다.',
  diagnosticsN: n => `진단 · ${n}개`,
  commandCopied: '명령 복사됨',
  description: '설명',
  editDescription: '설명 편집',
  cancelEdit: '편집 취소',
  noDescription: '아직 설명이 없습니다.',
  result: '결과',
  latestSummary: '최근 요약',
  dependencies: '의존 관계',
  blockedBy: '먼저 완료되어야 하는 작업',
  blocks: '이 작업의 완료를 기다리는 작업',
  comments: n => `댓글 · ${n}개`,
  commentsHelpRunning:
    '이 작업은 실행 중입니다. 메모는 몇 초 안에 작업 실행기의 현재 대화 턴에 반영되므로 작업을 멈췄다 재개할 필요가 없습니다. “메모와 함께 다시 대기열에 넣기”는 메모를 참고하여 작업을 처음부터 다시 시작합니다.',
  commentsHelp: '댓글은 작업 스레드에 추가됩니다. 작업 실행기가 작업을 맡을 때 참고 내용으로 읽습니다.',
  send: '보내기',
  comment: '댓글',
  messageWorker: '실행 중인 작업 실행기에 메시지 보내기…',
  addComment: '댓글 추가…',
  deliveredLive: '실행 중인 작업 실행기에 몇 초 안에 전달됩니다.',
  requeueWithNote: '메모와 함께 다시 대기열에 넣기',
  notePosted: '메모 게시됨 — 작업을 다시 대기열에 넣었습니다',
  activity: n => `활동 · ${n}개`,
  runs: n => `실행 · ${n}회`,
  workerLog: '작업 실행기 로그',
  workerLogTail: '작업 실행기 로그 · 최근 부분',
  attachments: n => `첨부 파일 · ${n}개`,
  noAttachments: '아직 첨부 파일이 없습니다.',
  uploadAttachment: '첨부 파일 업로드',
  taskActions: '작업 메뉴',
  copyTaskId: '작업 ID 복사',
  copyTitle: '제목 복사',
  copiedId: id => `${id} 복사됨`,
  copiedTitle: '제목 복사됨',
  close: '닫기',
  working: '작업 중',
  board: '보드',
  newBoard: '새 보드',
  newBoardDots: '새 보드…',
  exportDots: '내보내기…',
  importDots: '가져오기…',
  renameDots: '이름 변경…',
  settingsDots: '설정…',
  exportBoardTitle: '보드 내보내기…',
  importBoardTitle: '보드 가져오기…',
  boardExported: path => `보드를 ${path}(으)로 내보냈습니다`,
  boardImported: name => `${name} 보드를 가져왔습니다`,
  boardImportedAs: slug => `이미 사용 중인 이름이어서 ${slug}(으)로 가져왔습니다`,
  renameBoardTitle: '보드 이름 변경',
  deleteBoardTitle: name => `“${name}” 보드를 삭제할까요?`,
  deleteBoardConfirm: '보드를 지우지 않고 보관합니다. 작업과 첨부 파일은 디스크에 남으며 복원할 수 있습니다.',
  boardArchived: path => `보드를 ${path}에 보관했습니다`,
  boardSettingsFor: name => `보드 설정 — ${name}`,
  name: '이름',
  boardNamePlaceholder: '보드 이름',
  slug: slug => `슬러그: ${slug}`,
  project: '프로젝트',
  noProject: '프로젝트 없음 (임시 샌드박스)',
  projectHintPre:
    '새 작업은 프로젝트 저장소에서 실행되며 작업마다 worktree를 사용합니다. 작업을 만들 때 별도의 작업 공간을 지정할 수도 있습니다. 프로젝트 관리 명령: ',
  projectHintCmd: 'hermes project',
  createBoard: '보드 만들기',
  orchestratorProfile: '작업 조율 프로필',
  defaultAssignee: '기본 담당자',
  defaultParen: '(기본값)',
  autoDecompose: '분류 단계의 작업 자동 분할',
  profileDescriptions: '프로필 설명',
  profileDescriptionsHint:
    '작업 분할 에이전트가 프로필 설명을 참고해 작업을 배정합니다. 보조 모델로 자동 생성하거나 직접 작성할 수 있습니다.',
  profileGoodAt: '이 프로필은 어떤 작업에 능숙한가요?',
  auto: '자동',
  notify: {
    completedTitle: '작업 완료',
    blockedTitle: '작업이 막힘 — 입력이 필요합니다',
    blockLoopTitle: '분류 단계로 이동됨 — 결정이 필요합니다',
    gaveUpTitle: '작업 포기',
    gaveUpBody: '여러 번 시도했지만 완료하지 못했습니다. 작업 내용을 확인한 뒤 다시 실행하세요.',
    crashedTitle: '작업 실행기 비정상 종료 — 다시 시도합니다',
    timedOutTitle: '작업 시간 초과 — 다시 시도합니다',
    openKanban: '칸반 열기',
    artifacts: n => `결과물 ${n}개`
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

const zh: KanbanMessages = {
  nav: '看板',
  openBoard: '看板：打开面板',
  newTaskCommand: '看板：新建任务',
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
  select: modifier => `选择（${modifier}点击）`,
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
  model: '模型',
  modelInherit: '配置文件默认',
  modelClear: '清除模型覆盖',
  modelHint: '让该任务使用指定的模型与思考深度。未设置时使用所指派配置文件自身的设置。',
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
  evtCreated: (where, assignee) => `已创建${where ? `（${where}）` : ''}${assignee ? `・分配给 ${assignee}` : ''}`,
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
  blockKindTip: kind =>
    kind === 'needs_input'
      ? '工作单元正在等待人工输入。'
      : kind === 'capability'
        ? '工作单元遇到了无法逾越的能力限制。'
        : kind === 'transient'
          ? '工作单元因临时性错误失败。'
          : '工作单元报告了需要等待的依赖。',
  blockReason: '阻塞类型',
  blockRecurrences: '阻塞次数',
  blockRecurrencesTip: '人工解除阻塞后，该任务因同一原因再次阻塞的次数。',
  consecutiveFailures: '连续失败',
  lastFailureError: '最近失败',
  unblockedMessage: id => `已解除 ${id} 的阻塞。任务将在下一轮调度中就绪。`,
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
  close: '关闭',
  working: '进行中',
  board: '面板',
  newBoard: '新建面板',
  switchBoard: '切换面板',
  newBoardDots: '新建面板…',
  exportDots: '导出…',
  importDots: '导入…',
  renameDots: '重命名…',
  settingsDots: '设置…',
  exportBoardTitle: '导出面板…',
  importBoardTitle: '导入面板…',
  boardExported: path => `面板已导出至 ${path}`,
  boardImported: name => `已导入 ${name}`,
  boardImportedAs: slug => `该名称已被占用，已导入为 ${slug}`,
  renameBoardTitle: '重命名面板',
  deleteBoardTitle: name => `确定删除“${name}”？`,
  deleteBoardConfirm: '面板会被归档而非清除，其任务和附件仍保留在磁盘上，可以恢复。',
  boardArchived: path => `面板已归档至 ${path}`,
  boardSettingsFor: name => `面板设置 — ${name}`,
  name: '名称',
  boardNamePlaceholder: '面板名称',
  slug: slug => `slug: ${slug}`,
  project: '项目',
  noProject: '无项目（临时沙箱）',
  projectHintPre:
    '新任务将在项目的仓库中运行（每个任务一个 worktree）；每个任务在创建时仍可覆盖其工作区。管理项目请使用 ',
  projectHintCmd: 'hermes project',
  createBoard: '创建面板',
  orchestratorProfile: '编排者配置档',
  defaultAssignee: '默认负责人',
  defaultParen: '（默认）',
  autoDecompose: '自动分解分诊任务',
  profileDescriptions: '配置档说明',
  profileDescriptionsHint: '说明用于引导分解器的路由。可用辅助模型自动生成，或自行填写。',
  profileGoodAt: '这个配置档擅长什么？',
  auto: '自动',
  notify: {
    completedTitle: '任务已完成',
    blockedTitle: '任务受阻 — 需要你的输入',
    blockLoopTitle: '任务已转入分类 — 需要人工决定',
    gaveUpTitle: '任务已停止',
    gaveUpBody: 'Hermes 无法完成这个任务。打开看板查看原因并重新分配。',
    crashedTitle: '任务遇到问题 — Hermes 将自动重试',
    timedOutTitle: '任务耗时过长 — Hermes 将自动重试',
    openKanban: '打开看板',
    artifacts: (n: number) => `${n} 个产物`
  }
}

const zhHant: KanbanMessages = {
  nav: '看板',
  openBoard: '看板：開啟面板',
  newTaskCommand: '看板：新增任務',
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
  select: modifier => `選取（${modifier}點擊）`,
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
  model: '模型',
  modelInherit: '設定檔預設',
  modelClear: '清除模型覆寫',
  modelHint: '讓此任務使用指定的模型與思考深度。未設定時使用所指派設定檔本身的設定。',
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
  evtCreated: (where, assignee) => `已建立${where ? `（${where}）` : ''}${assignee ? `・指派給 ${assignee}` : ''}`,
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
  blockKindTip: kind =>
    kind === 'needs_input'
      ? '工作單元正在等待人工輸入。'
      : kind === 'capability'
        ? '工作單元遇到了無法跨越的能力限制。'
        : kind === 'transient'
          ? '工作單元因暫時性錯誤失敗。'
          : '工作單元回報了需要等待的相依關係。',
  blockReason: '封鎖類型',
  blockRecurrences: '封鎖次數',
  blockRecurrencesTip: '人工解除封鎖後，該任務因同一原因再次封鎖的次數。',
  consecutiveFailures: '連續失敗',
  lastFailureError: '最近失敗',
  unblockedMessage: id => `已解除 ${id} 的封鎖。任務將在下一輪排程中就緒。`,
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
  close: '關閉',
  working: '進行中',
  board: '面板',
  newBoard: '新增面板',
  switchBoard: '切換面板',
  newBoardDots: '新增面板…',
  exportDots: '匯出…',
  importDots: '匯入…',
  renameDots: '重新命名…',
  settingsDots: '設定…',
  exportBoardTitle: '匯出面板…',
  importBoardTitle: '匯入面板…',
  boardExported: path => `面板已匯出至 ${path}`,
  boardImported: name => `已匯入 ${name}`,
  boardImportedAs: slug => `該名稱已被使用，已匯入為 ${slug}`,
  renameBoardTitle: '重新命名面板',
  deleteBoardTitle: name => `確定刪除「${name}」？`,
  deleteBoardConfirm: '面板會被封存而非清除，其任務和附件仍保留在磁碟上，可以還原。',
  boardArchived: path => `面板已封存至 ${path}`,
  boardSettingsFor: name => `面板設定 — ${name}`,
  name: '名稱',
  boardNamePlaceholder: '面板名稱',
  slug: slug => `slug: ${slug}`,
  project: '專案',
  noProject: '無專案（暫存沙箱）',
  projectHintPre:
    '新任務將在專案的儲存庫中執行（每個任務一個 worktree）；每個任務在建立時仍可覆寫其工作區。管理專案請使用 ',
  projectHintCmd: 'hermes project',
  createBoard: '建立面板',
  orchestratorProfile: '編排者設定檔',
  defaultAssignee: '預設負責人',
  defaultParen: '（預設）',
  autoDecompose: '自動分解分類任務',
  profileDescriptions: '設定檔說明',
  profileDescriptionsHint: '說明用於引導分解器的路由。可用輔助模型自動產生，或自行填寫。',
  profileGoodAt: '這個設定檔擅長什麼？',
  auto: '自動',
  notify: {
    completedTitle: '任務已完成',
    blockedTitle: '任務受阻 — 需要你的輸入',
    blockLoopTitle: '任務已轉入分類 — 需要人工決定',
    gaveUpTitle: '任務已停止',
    gaveUpBody: 'Hermes 無法完成這個任務。開啟看板查看原因並重新指派。',
    crashedTitle: '任務遇到問題 — Hermes 將自動重試',
    timedOutTitle: '任務耗時過長 — Hermes 將自動重試',
    openKanban: '開啟看板',
    artifacts: (n: number) => `${n} 個產物`
  }
}

/** Registered via `ctx.i18n.register` at plugin load (disposer tracked). */
export const KANBAN_LOCALES: PluginLocaleBundles = { en, ko, ja, zh, 'zh-hant': zhHant }

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
