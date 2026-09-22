import { mergeTranslations, type TranslationOverride } from '@hermes/shared/i18n'

import { managedRolloutsEn } from './en'
import type { ManagedRolloutMessages } from '../managed-rollouts-types'

const overrides: TranslationOverride<ManagedRolloutMessages> = {
  title: '托管发布',
  noActive: '没有活动的发布快照。',
  history: '发布历史',
  unresolved: count => `未解决的安全栅栏：${count}`,
  archived: '已归档',
  active: '活动',
  sections: {
    fleet: '托管发布目标列表',
    preparation: '托管发布准备',
    configuration: '托管发布配置',
    preflight: '托管发布预检审查',
    wavePreview: '托管发布波次预览',
    active: '正在进行的托管发布',
    controls: '托管发布控制项',
    recovery: '托管发布恢复',
    history: '托管发布历史',
    summary: '托管发布摘要'
  },
  actions: {
    select: '选择',
    selected: '已选择',
    prepare: '准备所选目标',
    preparing: '准备中…',
    recheckEligibility: '重新检查资格',
    continueToPreflight: '继续进行预检',
    start: '开始发布',
    starting: '开始中…',
    pause: '暂停',
    pausing: '暂停中…',
    stop: '停止',
    stopping: '停止中…',
    resume: '恢复运行',
    verifyBeforePromotion: '晋级前验证',
    verifyingBeforePromotion: '正在晋级前验证',
    recheckOutcome: '重新检查结果',
    recoverConnections: '恢复连接',
    retry: '重试',
    exclude: '排除',
    stopAndPlanRetry: '停止并规划重试',
    archiveStoppedRollout: '归档已停止的发布'
  },
  status: {
    activePhase: phase => `当前阶段：${phase}`,
    queued: '已排队',
    preparing: '准备中',
    ready: '已就绪',
    running: '运行中',
    awaitingPromotion: '等待晋级',
    attentionRequired: '需要处理',
    paused: '已暂停',
    stopped: '已停止',
    completed: '已完成',
    completedWithExclusions: '已完成，但有排除项',
    failed: '失败',
    refused: '已拒绝',
    unknown: '结果未知',
    unverified: '尚未验证',
    recoveryRequired: '需要恢复',
    fenced: '已保留恢复安全栅栏',
    alreadyCurrent: '已经是当前版本',
    pending: '等待证据'
  },
  policy: {
    manual: '手动审批',
    automatic: '获得金丝雀审批且保持健康后自动推进',
    mode: mode => `模式：${mode}`,
    canaryGate: gate => `金丝雀门槛：${gate}`
  },
  labels: {
    progressionMode: '推进模式',
    concurrency: '并发数',
    targetDetails: (label, alias, machineId, installId) =>
      `${label} · ${alias ? `${alias} · ` : ''}${machineId} · ${installId}`,
    wave: (number, targets) => `第 ${number} 波：${targets || '无目标'}`,
    emptyWave: number => `第 ${number} 波：没有目标`,
    progress: (completed, total) => `进度：已完成 ${total} 个目标中的 ${completed} 个。`,
    receipt: (outcome, correlationId) => `回执：${outcome}（${correlationId}）`,
    readiness: value => `就绪状态：${value}`,
    reason: value => `原因：${value}`,
    attempt: (installId, phase) => `${installId} · ${phase}`,
    historyEntry: (id, phase, updatedAt, unresolved, archived) =>
      `${id} · ${phase} · ${updatedAt} · 未解决栅栏 ${unresolved} · ${archived ? '已归档' : '活动'}`,
    archive: archived => `归档状态：${archived ? '已归档' : '活动'}`
  },
  warnings: {
    sharedMachine: '共享机器；准备前请检查目标所有权。',
    unsupportedTarget: '目标不受支持。',
    preparation: '准备操作遵循已配置的分支尖端；它不是固定目标的发布。',
    preparationMayDisconnect: '准备操作可能会暂时断开会话。',
    preparationInvalidatesReview: '准备操作会使之前的审查失效；目标、别名、来源或范围变化后必须重新确认资格。',
    changedPlan: '配置已变化；请更新审查令牌后再继续。',
    projectionUnavailable: '波次预测不可用；预检无法继续。',
    incompatible: '当前能力与此计划不兼容；开始操作不可用。',
    commitmentBoundary: '停止会阻止新的授权；已经提交的更新仍可能继续排空会话、应用更新并恢复范围。',
    unknownOutcome: '远程结果未知；没有回执不能被当作成功。',
    recoveryFence: '恢复安全栅栏仍然保留；新的不安全操作继续被阻止。',
    unknownAndFenced: '结果未知，并且恢复安全栅栏仍然保留；新的不安全操作已被阻止。',
    estimateUnavailable: '无法提供估算。',
    unavailable: '托管发布数据不可用；没有开始新的工作。',
    stale: '此发布快照可能已过期；执行操作前请重新检查。',
    reconnecting: '正在重新连接托管发布状态…'
  },
  descriptions: {
    serialCapability: '当前目标契约要求串行能力。在草稿发生变化前，已选择的金丝雀目标会保持不变。',
    canonicalPlan: '提交的草稿与预览中的规范波次计划完全一致。更新或变化的行需要重新确认。',
    manualPolicy: '每个波次完成后都需要手动审批。',
    automaticPolicy: '只有在明确批准金丝雀、证据经过重新检查且发布保持健康时，才允许自动推进。',
    commitmentBoundary: '你停止发布后，已经提交的更新仍可能完成。停止操作会阻止新的授权，但不能取消已经交给远程更新器的工作；远程更新器可能仍在排空会话、应用版本并恢复配置范围。',
    lastSettledLocalFenceNotLive: '这是上次已结算状态和本地安全栅栏的记录，并非实时状态。此桌面未观察到的远程变化不在此证据之内。',
    archiveRetainsFence: '归档只改变展示和历史记录；它不会删除证据，也不会释放尚未解决的安全栅栏。',
    noAutomaticResume: '桌面重启后必须先重新协调，再明确选择恢复或晋级；已保存的策略不会自动成为新的授权。',
    historicalEstimate: '历史估算仅供参考，不会授予开始新工作的权限。'
  },
  summary: {
    outcome: phase => `结果：${phase}`,
    excludedTargets: count => `排除的目标：${count}`,
    unresolvedFences: count => `未解决的安全栅栏：${count}`,
    archive: archived => `归档状态：${archived ? '已归档' : '活动'}`,
    reason: reason => `原因：${reason}`
  },
  a11y: {
    confirmPreparation: '我确认这些目标正是要单独准备的目标集合。',
    confirmPreflight: '我确认这次预检审查。',
    selectTarget: label => `选择目标 ${label}`,
    selectedTarget: label => `取消选择目标 ${label}`,
    warning: message => `警告：${message}`,
    status: message => `状态：${message}`,
    action: message => `操作：${message}`,
    historyEntry: id => `打开发布 ${id} 的详细信息`,
    attemptToggle: (installId, expanded) => `${expanded ? '收起' : '展开'}目标 ${installId} 的尝试详情`
  }
}

export const managedRolloutsZh = mergeTranslations<ManagedRolloutMessages>(managedRolloutsEn, overrides)
export const managedRollouts = managedRolloutsZh
export default managedRolloutsZh
