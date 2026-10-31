import type { QueueFairCopy } from './types_queue_fair'

export const zhHantQueueFair: QueueFairCopy = {
  holdUnknown: reason => `等待中：${reason}`,
  hold: {
    session_busy: '正在等待目前輪次結束',
    slot: '正在等待空閒工作階段名額',
    editing: '正在編輯，暫不開始',
    restarting: '正在等待閘道重新啟動',
    paused: '已暫停，直到遺失的輪次被處理',
    provider_limit: '正在等待服務商限額重設',
    memory: '正在等待可用記憶體'
  },
  moveUp: '上移',
  moveDown: '下移',
  holdFailed: '編輯期間無法暫停這則排隊訊息；它可能在儲存前開始'
}
