import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's per-code title and body (lib/error-surface-copy.ts), keyed by error_surface.code.
export const zhErrorCodes: Partial<ThreadCopy['errorCodes']> = {
  cron_session_read_only: {
    title: '此聊天属于一个定时任务',
    body: '它只运行该任务，因此你的消息未被执行。请新建聊天后再发送。'
  },
  session_read_only: {
    title: '此聊天为只读',
    body: '它是 Hermes 只能显示、无法继续的已保存历史记录，因此你的消息未被执行。请新建聊天后再发送。'
  },
  workspace_unavailable: {
    title: '此聊天的文件夹已不存在',
    body: '此聊天使用的文件夹已被移动或删除，因此你的消息未被执行。请重新创建该文件夹，或使用“移动到项目”为聊天指定其他文件夹，然后重试。'
  },
  launch_credentials_unavailable: {
    title: '此聊天的 API 密钥已更改',
    body: '此聊天开始时使用的 API 密钥后来已被更改或删除，因此你的消息未发送。打开的聊天不会自行切换到其他密钥。请使用“开始新会话”以使用当前密钥。'
  }
}
