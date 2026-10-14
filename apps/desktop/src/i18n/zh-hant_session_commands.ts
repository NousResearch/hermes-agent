// Typed /undo /retry /steer /queue lines (en_session_commands.ts is the source).
export const zhHantSessionCommands = {
  undoUsage: '用法：/undo [N]',
  undoNothing: '沒有可復原的內容',
  undidOne: (count: string) => `已復原 ${count} 則訊息 — 被移除的訊息已放回輸入框`,
  undidOther: (count: string) => `已復原 ${count} 則訊息 — 被移除的訊息已放回輸入框`,
  retryNothing: '沒有可重試的內容',
  retryUnsupported: '/retry：無法重試包含媒體或未知內容的訊息',
  steerUsage: '用法：/steer <提示>',
  steerQueued: (preview: string) => `引導已排入佇列 — 將在下一次工具呼叫後送達：「${preview}」`,
  steerNoActiveTurn: (preview: string) => `目前沒有進行中的回合 — 已排入下一回合：「${preview}」`,
  steerRejected: '引導遭拒 — 沒有進行中的回合，已排入下一回合',
  queued: (preview: string) => `已排入佇列：「${preview}」`,
  queueFailed: '無法將訊息排入佇列',
  countOne: (count: string) => `${count} 則排隊訊息`,
  countOther: (count: string) => `${count} 則排隊訊息`
}
