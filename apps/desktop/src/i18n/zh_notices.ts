import type { TranslationOverrides } from './define-locale'
import { zhDesktopCommands } from './zh_desktop_commands'
import { zhSessionCommands } from './zh_session_commands'

// Shell notices (remote-display toast, butterbar, keep-awake offer), spread into zh.ts.
export const zhNotices = {
  remoteDisplayBanner: {
    message: reason => `软件渲染已启用 — 检测到远程显示（${reason}）。为防止画面闪烁，已禁用 GPU 加速。`
  },
  butterbar: {
    goTo: (index, total) => `显示第 ${index} 条通知，共 ${total} 条`,
    legal: {
      before: '使用 Hermes Agent 即表示受我们的',
      terms: '服务条款',
      between: '和',
      privacy: '隐私政策',
      after: '约束。'
    }
  },
  promptNotices: {
    legacySendUnconfirmed: '此服务器无法确认这条消息之前的发送，它可能已经运行过。再次发送前请先查看对话。',
    restoredDraftAttachmentsDropped: '未发送消息中的附件没有恢复。发送前请重新添加。',
    mcpReloadDeferred: '此对话正忙：MCP 更改将在当前轮次结束并再次重新加载 MCP 后生效，或在重新打开对话时生效',
    mcpReloadUnavailable: servers => `未能连接：${servers}`,
    compactedTurnUnavailable: '这条消息在压缩对话时已被概括，无法再恢复或编辑。',
    staleTurnUnavailable: '这条消息已不在此对话的历史记录中。',
    sessionCommands: zhSessionCommands,
    desktopCommands: zhDesktopCommands
  },
  keepAwakeOffer: {
    title: '电脑在 Hermes 工作时进入了睡眠',
    message: 'Hermes 可以在任务运行时保持电脑唤醒。屏幕仍会照常关闭，使用电池时电脑仍可能睡眠。',
    action: '运行期间保持唤醒'
  },
  billingBlock: {
    titleNous: 'Nous 额度已用尽',
    titleProvider: provider => `额度已用尽 — ${provider}`,
    fallbackMessage: '您的账户额度已用尽。请充值以继续使用。',
    openBilling: '打开账单',
    addCredits: '添加额度',
    dismiss: '忽略'
  }
} satisfies Pick<
  TranslationOverrides,
  'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer' | 'billingBlock'
>
