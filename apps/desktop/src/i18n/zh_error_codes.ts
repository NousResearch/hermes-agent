import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's per-code title and body (lib/error-surface-copy.ts), keyed by error_surface.code.
export const zhErrorCodes: Partial<ThreadCopy['errorCodes']> = {
  auth: {
    title: provider => `${provider} 拒绝了你的 API 密钥`,
    body: () => '已保存的密钥无效或已被撤销。请更新密钥，然后重新发送消息。'
  },
  auth_permanent: {
    title: provider => `${provider} 拒绝了你的 API 密钥`,
    body: () => '已保存的密钥无效或已被撤销。请更新密钥，然后重新发送消息。'
  },
  billing: {
    title: provider => `${provider} 额度已用完`,
    body: provider => `请充值额度，以继续在 ${provider} 上使用付费模型。`
  },
  no_reply: {
    title: '未收到回复',
    body: 'Hermes 已完成本轮，但回复未送达此窗口。请重试以重新发送。'
  },
  upstream_blocked: {
    title: provider => `${provider} 拦截了此请求`,
    body: provider => `${provider} 前端的防火墙在请求到达模型前将其拒绝。请切换服务商，或为其设置自定义 User-Agent。`
  },
  context_overflow: {
    title: '此聊天过长',
    body: (_provider, model) => `对话已超出${model ? ` ${model} ` : '模型'}的上下文窗口。请压缩对话以继续。`
  },
  payload_too_large: {
    title: '此消息过大',
    body: provider => `请求（含附件）超出了 ${provider} 的大小限制。请压缩对话或移除附件。`
  },
  model_not_found: {
    title: (_provider, model) => `${model || '此模型'} 不可用`,
    body: (provider, model) => `${provider} 未在你的账户上提供 ${model || '此模型'}。请选择其他模型。`
  },
  truncated: {
    title: '回复被截断',
    body: (_provider, model) => `${model || '模型'} 的回复在完成前中断了。点击“继续”从中断处接着生成。`
  },
  loop_error: {
    title: 'Hermes 无法完成此步骤',
    body: '处理回复时出现问题。请重试；若再次发生，请从错误详情中发送诊断信息。'
  },
  agent_init_failed: {
    title: '无法开始此聊天',
    body: provider => `Hermes 无法为此聊天连接到 ${provider}。请重试；若持续失败，请在设置中检查该服务商。`
  },
  SESSION_NOT_OWNED: {
    title: '聊天已在别处打开',
    body: '此聊天已在另一个 Hermes 窗口或终端中打开。请在这里新建聊天，或先在那边关闭它。'
  },
  disk_full: {
    title: '磁盘已满',
    body: 'Hermes 无法保存此对话。请释放空间后重试。'
  },
  // Nous free tier. The body is normally the backend's own sentence; these stand in for an older backend.
  free_tier_disabled: {
    title: '登录以继续聊天',
    body: '使用 Nous 账户登录即可继续聊天，完全免费。'
  },
  free_tier_rate_limited: {
    title: '免费额度已用完',
    body: '额度很快就会刷新。使用 Nous 账户登录可获得更多额度，完全免费。'
  },
  free_tier_at_capacity: {
    title: '免费服务繁忙',
    body: '免费登录即可跳过排队，或稍后再试。'
  },
  free_tier_model_not_free: {
    title: '此模型需要登录',
    body: 'Hermes 暂时使用免费模型。使用 Nous 账户登录可使用更多模型，完全免费。'
  },
  free_tier_route: {
    title: '无法连接免费模型',
    body: '使用 Nous 账户免费登录，或检查 NOUS_INFERENCE_BASE_URL 设置。'
  },
  free_tier_outage: {
    title: '免费模型出现问题',
    body: '请过一分钟后再发送消息。'
  },
  free_tier_refused: {
    title: '登录以继续聊天',
    body: '使用 Nous 账户登录是免费的。'
  },
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
  },
  local_runtime_unavailable: {
    title: '无法连接本地模型服务器',
    body: '本地模型服务器尚未运行：它可能仍在启动。请稍后重试，或切换到其他模型。',
    bodyNoRetry:
      '本地模型服务器已关闭。请在 Hermes 桌面应用或终端中的 `hermes model` 启动它，或切换到其他模型。'
  }
}
