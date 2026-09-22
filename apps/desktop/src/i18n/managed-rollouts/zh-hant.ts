import { mergeTranslations, type TranslationOverride } from '@hermes/shared/i18n'

import { managedRolloutsEn } from './en'
import type { ManagedRolloutMessages } from '../managed-rollouts-types'

const overrides: TranslationOverride<ManagedRolloutMessages> = {
  title: '受控發布',
  noActive: '沒有進行中的發布快照。',
  history: '發布記錄',
  unresolved: count => `未解決的安全圍欄：${count}`,
  archived: '已封存',
  active: '進行中',
  sections: {
    fleet: '受控發布目標清單',
    preparation: '受控發布準備',
    configuration: '受控發布設定',
    preflight: '受控發布預檢審查',
    wavePreview: '受控發布波次預覽',
    active: '進行中的受控發布',
    controls: '受控發布控制項',
    recovery: '受控發布復原',
    history: '受控發布記錄',
    summary: '受控發布摘要'
  },
  actions: {
    select: '選取',
    selected: '已選取',
    prepare: '準備選取的目標',
    preparing: '準備中…',
    recheckEligibility: '重新檢查資格',
    continueToPreflight: '繼續進行預檢',
    start: '開始發布',
    starting: '開始中…',
    pause: '暫停',
    pausing: '暫停中…',
    stop: '停止',
    stopping: '停止中…',
    resume: '繼續',
    verifyBeforePromotion: '晉級前驗證',
    verifyingBeforePromotion: '正在晉級前驗證',
    recheckOutcome: '重新檢查結果',
    recoverConnections: '復原連線',
    retry: '重試',
    exclude: '排除',
    stopAndPlanRetry: '停止並規劃重試',
    archiveStoppedRollout: '封存已停止的發布'
  },
  status: {
    activePhase: phase => `目前階段：${phase}`,
    queued: '已排入佇列',
    preparing: '準備中',
    ready: '已就緒',
    running: '執行中',
    awaitingPromotion: '等待晉級',
    attentionRequired: '需要處理',
    paused: '已暫停',
    stopped: '已停止',
    completed: '已完成',
    completedWithExclusions: '已完成，但有排除項目',
    failed: '失敗',
    refused: '已拒絕',
    unknown: '結果未知',
    unverified: '尚未驗證',
    recoveryRequired: '需要復原',
    fenced: '已保留復原安全圍欄',
    alreadyCurrent: '已是目前版本',
    pending: '等待證據'
  },
  policy: {
    manual: '手動核准',
    automatic: '取得金絲雀核准且保持健康後自動推進',
    mode: mode => `模式：${mode}`,
    canaryGate: gate => `金絲雀閘門：${gate}`
  },
  labels: {
    progressionMode: '推進模式',
    concurrency: '並行數',
    targetDetails: (label, alias, machineId, installId) =>
      `${label} · ${alias ? `${alias} · ` : ''}${machineId} · ${installId}`,
    wave: (number, targets) => `第 ${number} 波：${targets || '沒有目標'}`,
    emptyWave: number => `第 ${number} 波：沒有目標`,
    progress: (completed, total) => `進度：${total} 個目標中已完成 ${completed} 個。`,
    receipt: (outcome, correlationId) => `回執：${outcome}（${correlationId}）`,
    readiness: value => `就緒狀態：${value}`,
    reason: value => `原因：${value}`,
    attempt: (installId, phase) => `${installId} · ${phase}`,
    historyEntry: (id, phase, updatedAt, unresolved, archived) =>
      `${id} · ${phase} · ${updatedAt} · 未解決圍欄 ${unresolved} · ${archived ? '已封存' : '進行中'}`,
    archive: archived => `封存狀態：${archived ? '已封存' : '進行中'}`
  },
  warnings: {
    sharedMachine: '共用機器；準備前請檢查目標所有權。',
    unsupportedTarget: '不支援的目標。',
    preparation: '準備操作會遵循已設定的分支尖端；它不是固定目標的發布。',
    preparationMayDisconnect: '準備操作可能會暫時中斷工作階段。',
    preparationInvalidatesReview: '準備操作會使先前的審查失效；目標、別名、來源或範圍變更後必須重新確認資格。',
    changedPlan: '設定已變更；請更新審查權杖後再繼續。',
    projectionUnavailable: '波次預測無法使用；預檢無法繼續。',
    incompatible: '目前能力與此計畫不相容；無法開始。',
    commitmentBoundary: '停止會阻止新的授權；已提交的更新仍可能繼續排空工作階段、套用更新並復原範圍。',
    unknownOutcome: '遠端結果未知；沒有回執不能被視為成功。',
    recoveryFence: '復原安全圍欄仍然保留；新的不安全操作會繼續被阻止。',
    unknownAndFenced: '結果未知，而且復原安全圍欄仍然保留；新的不安全操作已被阻止。',
    estimateUnavailable: '無法提供估算。',
    unavailable: '受控發布資料無法使用；沒有開始新的工作。',
    stale: '此發布快照可能已過期；執行操作前請重新檢查。',
    reconnecting: '正在重新連線受控發布狀態…'
  },
  descriptions: {
    serialCapability: '目前目標契約要求序列能力。在草稿變更前，已選取的金絲雀目標會保持不變。',
    canonicalPlan: '提交的草稿與預覽中的規範波次計畫完全一致。更新或變更的列需要重新確認。',
    manualPolicy: '每個已結算的波次後都需要手動核准。',
    automaticPolicy: '只有在明確核准金絲雀、證據重新檢查且發布保持健康時，才允許自動推進。',
    commitmentBoundary: '你停止發布後，已提交的更新仍可能完成。停止會阻止新的授權，但不能取消已交給遠端更新器的工作；遠端更新器可能仍在排空工作階段、套用版本並復原設定範圍。',
    lastSettledLocalFenceNotLive: '這是上次已結算狀態加上本機安全圍欄的記錄，不是即時狀態。此桌面未觀察到的遠端變化不在這項證據之內。',
    archiveRetainsFence: '封存只會改變顯示與記錄；不會刪除證據，也不會釋放尚未解決的安全圍欄。',
    noAutomaticResume: '桌面重新啟動後必須先重新協調，再明確選擇繼續或晉級；已儲存的策略不會自動成為新的授權。',
    historicalEstimate: '歷史估算僅供參考，不會授予開始新工作的權限。'
  },
  summary: {
    outcome: phase => `結果：${phase}`,
    excludedTargets: count => `排除的目標：${count}`,
    unresolvedFences: count => `未解決的安全圍欄：${count}`,
    archive: archived => `封存狀態：${archived ? '已封存' : '進行中'}`,
    reason: reason => `原因：${reason}`
  },
  a11y: {
    confirmPreparation: '我確認這些目標正是要分開準備的目標集合。',
    confirmPreflight: '我確認這次預檢審查。',
    selectTarget: label => `選取目標 ${label}`,
    selectedTarget: label => `取消選取目標 ${label}`,
    warning: message => `警告：${message}`,
    status: message => `狀態：${message}`,
    action: message => `操作：${message}`,
    historyEntry: id => `開啟發布 ${id} 的詳細資料`,
    attemptToggle: (installId, expanded) => `${expanded ? '收合' : '展開'}目標 ${installId} 的嘗試詳細資料`
  }
}

export const managedRolloutsZhHant = mergeTranslations<ManagedRolloutMessages>(managedRolloutsEn, overrides)
export const managedRollouts = managedRolloutsZhHant
export default managedRolloutsZhHant
