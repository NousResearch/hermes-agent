import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into zh.ts.
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
    legacySendUnconfirmed:
      '此服务器无法确认这条消息之前的发送，它可能已经运行过。再次发送前请先查看对话。'
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices'>
