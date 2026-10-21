// Typed /undo /retry /steer /queue lines (en_session_commands.ts is the source).
export const zhHantSessionCommands = {
  undoUsage: '用法：/undo [N]',
  undoNothing: '沒有可復原的內容',
  undidOne: (count: string) => `已復原 ${count} 則訊息 — 被移除的訊息已放回輸入框`,
  undidOther: (count: string) => `已復原 ${count} 則訊息 — 被移除的訊息已放回輸入框`,
  retryNothing: '沒有可重試的內容',
  retryUnsupported: '/retry：無法重試包含媒體或未知內容的訊息',
  busy: (command: string) =>
    `會話忙碌中 — Hermes 仍在回覆。請先停止目前回覆（Stop 按鈕，或在終端機中按 Ctrl+C），然後再執行 /${command}。`,
  busyCompressing: (command: string) =>
    `會話忙碌中 — 正在執行 /compress；完成後再執行 /${command}。`,
  steerUsage: '用法：/steer <提示>',
  steerQueued: (preview: string) => `引導已排入佇列 — 將在下一次工具呼叫後送達：「${preview}」`,
  steerNoActiveTurn: (preview: string) => `目前沒有進行中的回合 — 已排入下一回合：「${preview}」`,
  steerRejected: '引導遭拒 — 沒有進行中的回合，已排入下一回合',
  queued: (preview: string) => `已排入佇列：「${preview}」`,
  queueFailed: '無法將訊息排入佇列',
  countOne: (count: string) => `${count} 則排隊訊息`,
  countOther: (count: string) => `${count} 則排隊訊息`,
  queueEditKeptTitle: '排隊訊息未變更',
  queueEditKept: '在儲存編輯之前它已開始或被拒絕。編輯後的文字仍在輸入框中。',
  queueEditUnconfirmedTitle: '編輯未確認',
  queueEditUnconfirmed: 'Hermes 無法確認你的編輯。再次儲存將重試同一編輯，不會重複排隊。'
}
