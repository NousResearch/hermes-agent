// Typed /undo /retry /steer /queue lines (en_session_commands.ts is the source).
export const zhSessionCommands = {
  undoUsage: '用法：/undo [N]',
  undoNothing: '没有可撤销的内容',
  undidOne: (count: string) => `已撤销 ${count} 条消息 — 被移除的消息已放回输入框`,
  undidOther: (count: string) => `已撤销 ${count} 条消息 — 被移除的消息已放回输入框`,
  retryNothing: '没有可重试的内容',
  retryUnsupported: '/retry：无法重试包含媒体或未知内容的消息',
  steerUsage: '用法：/steer <提示>',
  steerQueued: (preview: string) => `引导已排队 — 将在下一次工具调用后送达：“${preview}”`,
  steerNoActiveTurn: (preview: string) => `当前没有进行中的回合 — 已排队到下一回合：“${preview}”`,
  steerRejected: '引导被拒绝 — 没有进行中的回合，已排队到下一回合',
  queued: (preview: string) => `已排队：“${preview}”`,
  queueFailed: '无法将消息加入队列',
  countOne: (count: string) => `${count} 条排队消息`,
  countOther: (count: string) => `${count} 条排队消息`
}
