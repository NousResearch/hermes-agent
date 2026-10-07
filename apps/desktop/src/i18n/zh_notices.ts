import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into zh.ts.
export const zhNotices = {
  remoteDisplayBanner: {
    message: reason => `软件渲染已启用 — 检测到远程显示（${reason}）。为防止画面闪烁，已禁用 GPU 加速。`
  },
  butterbar: {
    goTo: (index, total) => `显示第 ${index} 条通知，共 ${total} 条`
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar'>
