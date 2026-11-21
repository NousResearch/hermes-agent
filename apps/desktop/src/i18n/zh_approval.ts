import type { ApprovalCopy } from './types_approval'

export const zhApproval: Partial<ApprovalCopy> = {
  gatewayDisconnected: 'Hermes 网关未连接',
  sendFailed: '无法发送审批响应',
  run: '运行',
  command: '命令',
  moreOptions: '更多审批选项',
  allowSession: '允许本会话',
  alwaysAllowMenu: '始终允许…',
  jumpToApproval: '需要审批',
  reject: '拒绝',
  alwaysTitle: '始终允许此命令？',
  alwaysDescription: pattern =>
    `这会将“${pattern}”模式加入永久允许列表 (~/.hermes/config.yaml)。Hermes 对类似命令将不再询问，包括当前会话和未来会话。`,
  alwaysAllow: '始终允许',
  deny: '拒绝',
  allow: '允许',
  allowHint: '直到关闭沙盒',
  denyReason: '拒绝原因',
  denyReasonPlaceholder: '告诉 Hermes 原因（可选）',
  access: {
    title: '文件夹访问',
    readOnly: '只读',
    readWrite: '读写',
    alwaysTitle: '始终允许此文件夹？',
    alwaysDescription: '该文件夹将加入此档案的沙盒。Hermes 在任何对话中都不会再询问。'
  },
  network: {
    title: '网站访问',
    alwaysTitle: '始终允许此网站？',
    alwaysDescription: '该网站将加入此档案的允许网站。Hermes 在任何对话中都不会再询问。'
  },
  schedule: {
    title: '安排此任务？',
    allowSession: '在此对话中允许',
    alwaysAllowMenu: '始终允许新任务…',
    alwaysTitle: '始终允许新的计划任务？',
    alwaysDescription:
      'Hermes 将不经询问直接创建计划任务。从 config.yaml 的 command_allowlist 中移除 cron_create 即可重新开启询问。',
    timedOutSystemLine:
      '审批已超时 — 任务未创建。请让 Hermes 重试，或在 设置 → 安全 → 审批超时 中提高上限。',
    editTitle: '修改此任务？',
    editTimedOutSystemLine:
      '审批已超时 — 任务未修改。请让 Hermes 重试，或在 设置 → 安全 → 审批超时 中提高上限。'
  }
}
