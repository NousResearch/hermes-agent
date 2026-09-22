import { mergeTranslations, type TranslationOverride } from '@hermes/shared/i18n'

import { managedRolloutsEn } from './en'
import type { ManagedRolloutMessages } from '../managed-rollouts-types'

const overrides: TranslationOverride<ManagedRolloutMessages> = {
  title: '管理ロールアウト',
  noActive: 'アクティブなロールアウトのスナップショットはありません。',
  history: 'ロールアウト履歴',
  unresolved: count => `未解決のフェンス：${count}`,
  archived: 'アーカイブ済み',
  active: 'アクティブ',
  sections: {
    fleet: '管理ロールアウトの対象一覧',
    preparation: '管理ロールアウトの準備',
    configuration: '管理ロールアウトの設定',
    preflight: '管理ロールアウトの事前確認',
    wavePreview: '管理ロールアウトのウェーブプレビュー',
    active: 'アクティブな管理ロールアウト',
    controls: '管理ロールアウトの操作',
    recovery: '管理ロールアウトの復旧',
    history: '管理ロールアウトの履歴',
    summary: '管理ロールアウトの概要'
  },
  actions: {
    select: '選択',
    selected: '選択済み',
    prepare: '選択した対象を準備',
    preparing: '準備中…',
    recheckEligibility: '資格を再確認',
    continueToPreflight: '事前確認へ進む',
    start: 'ロールアウトを開始',
    starting: '開始中…',
    pause: '一時停止',
    pausing: '一時停止中…',
    stop: '停止',
    stopping: '停止中…',
    resume: '再開',
    verifyBeforePromotion: '昇格前に検証',
    verifyingBeforePromotion: '昇格前に検証中',
    recheckOutcome: '結果を再確認',
    recoverConnections: '接続を復旧',
    retry: '再試行',
    exclude: '除外',
    stopAndPlanRetry: '停止して再試行を計画',
    archiveStoppedRollout: '停止したロールアウトをアーカイブ'
  },
  status: {
    activePhase: phase => `現在のフェーズ：${phase}`,
    queued: 'キュー待ち',
    preparing: '準備中',
    ready: '準備完了',
    running: '実行中',
    awaitingPromotion: '昇格待ち',
    attentionRequired: '対応が必要',
    paused: '一時停止中',
    stopped: '停止済み',
    completed: '完了',
    completedWithExclusions: '除外を含めて完了',
    failed: '失敗',
    refused: '拒否',
    unknown: '結果不明',
    unverified: '未検証',
    recoveryRequired: '復旧が必要',
    fenced: '復旧フェンスを保持中',
    alreadyCurrent: 'すでに最新',
    pending: '証拠待ち'
  },
  policy: {
    manual: '手動承認',
    automatic: 'カナリア承認後、健全な間は自動で進める',
    mode: mode => `モード：${mode}`,
    canaryGate: gate => `カナリアゲート：${gate}`
  },
  labels: {
    progressionMode: '進行モード',
    concurrency: '同時実行数',
    targetDetails: (label, alias, machineId, installId) =>
      `${label} · ${alias ? `${alias} · ` : ''}${machineId} · ${installId}`,
    wave: (number, targets) => `ウェーブ ${number}：${targets || '対象なし'}`,
    emptyWave: number => `ウェーブ ${number}：対象なし`,
    progress: (completed, total) => `進捗：${total} 件中 ${completed} 件の対象が完了しました。`,
    receipt: (outcome, correlationId) => `レシート：${outcome}（${correlationId}）`,
    readiness: value => `準備状態：${value}`,
    reason: value => `理由：${value}`,
    attempt: (installId, phase) => `${installId} · ${phase}`,
    historyEntry: (id, phase, updatedAt, unresolved, archived) =>
      `${id} · ${phase} · ${updatedAt} · 未解決フェンス ${unresolved} · ${archived ? 'アーカイブ済み' : 'アクティブ'}`,
    archive: archived => `アーカイブ：${archived ? 'アーカイブ済み' : 'アクティブ'}`
  },
  warnings: {
    sharedMachine: '共有マシンです。準備前に所有権を確認してください。',
    unsupportedTarget: 'サポートされていない対象です。',
    preparation: '準備は設定済みのブランチ先端に従います。固定されたロールアウトではありません。',
    preparationMayDisconnect: '準備中にセッションが一時的に切断される場合があります。',
    preparationInvalidatesReview: '準備を行うと以前の確認は無効になります。対象、別名、ソース、またはスコープを変更した後は再確認が必要です。',
    changedPlan: '設定が変更されました。続行する前に確認トークンを更新してください。',
    projectionUnavailable: 'ウェーブ予測を利用できないため、事前確認を進められません。',
    incompatible: '現在の機能はこの計画と互換性がないため、開始できません。',
    commitmentBoundary: '停止すると新しい承認は止まりますが、コミット済みの更新はセッションの排出、適用、スコープ復元を続ける場合があります。',
    unknownOutcome: 'リモートの結果は不明です。レシートがないことを成功の証拠として扱わないでください。',
    recoveryFence: '復旧フェンスが保持されているため、新しい安全でない処理はブロックされます。',
    unknownAndFenced: '結果は不明で復旧フェンスも保持されています。新しい安全でない処理はブロックされます。',
    estimateUnavailable: '見積もりは利用できません。',
    unavailable: '管理ロールアウトのデータを利用できません。新しい処理は開始されていません。',
    stale: 'このロールアウトのスナップショットは古い可能性があります。操作前に再確認してください。',
    reconnecting: '管理ロールアウトの状態に再接続中…'
  },
  descriptions: {
    serialCapability: '現在の対象契約では直列実行が必要です。草稿を変更するまで、選択したカナリアは固定されます。',
    canonicalPlan: '送信された草稿はプレビューした正規のウェーブ計画と完全に一致します。更新または変更された行は再確認が必要です。',
    manualPolicy: '確定した各ウェーブの後に手動承認が必要です。',
    automaticPolicy: '明示的なカナリア承認と新しい証拠の確認があり、ロールアウトが健全な場合だけ自動で進められます。',
    commitmentBoundary: 'ロールアウトを停止した後も、コミット済みの更新は完了する場合があります。停止は新しい承認を防ぎますが、リモート更新処理へ渡された作業を取り消すことはできず、セッションの排出、バージョンの適用、設定スコープの復元が続く場合があります。',
    lastSettledLocalFenceNotLive: 'これは最後に確定した状態とローカルフェンスの記録であり、ライブ状態ではありません。この Desktop が観測していないリモートの変化は確認できません。',
    archiveRetainsFence: 'アーカイブは表示と履歴だけを変更します。証拠を削除したり、未解決のフェンスを解除したりすることはありません。',
    noAutomaticResume: 'Desktop の再起動後は、まず再調整してから明示的に再開または昇格してください。保存されたポリシーは新しい許可にはなりません。',
    historicalEstimate: '過去の見積もりは情報提供のみで、新しい処理を許可するものではありません。'
  },
  summary: {
    outcome: phase => `結果：${phase}`,
    excludedTargets: count => `除外した対象：${count}`,
    unresolvedFences: count => `未解決フェンス：${count}`,
    archive: archived => `アーカイブ：${archived ? 'アーカイブ済み' : 'アクティブ'}`,
    reason: reason => `理由：${reason}`
  },
  a11y: {
    confirmPreparation: 'これらの対象が個別に準備する対象セットであることを確認します。',
    confirmPreflight: 'この事前確認を確認します。',
    selectTarget: label => `対象 ${label} を選択`,
    selectedTarget: label => `対象 ${label} の選択を解除`,
    warning: message => `警告：${message}`,
    status: message => `状態：${message}`,
    action: message => `操作：${message}`,
    historyEntry: id => `ロールアウト ${id} の詳細を開く`,
    attemptToggle: (installId, expanded) => `${expanded ? '閉じる' : '開く'}対象 ${installId} の試行詳細`
  }
}

export const managedRolloutsJa = mergeTranslations<ManagedRolloutMessages>(managedRolloutsEn, overrides)
export const managedRollouts = managedRolloutsJa
export default managedRolloutsJa
