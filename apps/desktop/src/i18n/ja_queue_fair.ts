import type { QueueFairCopy } from './types_queue_fair'

export const jaQueueFair: QueueFairCopy = {
  holdUnknown: reason => `待機中: ${reason}`,
  hold: {
    session_busy: '現在のターンの完了を待っています',
    slot: '空きセッション枠を待っています',
    editing: '編集中のため、まだ開始しません',
    restarting: 'ゲートウェイの再起動を待っています',
    paused: '失われたターンが解決されるまで一時停止中',
    provider_limit: 'プロバイダーの上限リセットを待っています',
    memory: '空きメモリを待っています'
  },
  moveUp: '上へ移動',
  moveDown: '下へ移動',
  holdFailed: '編集中にこのキュー内のメッセージを保留できませんでした。保存前に開始される可能性があります'
}
