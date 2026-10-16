import type { TranslationOverrides } from './define-locale'

// Temporary (incognito) chat copy, composed by zh.ts.
export const zhTemporaryChat = {
  temporaryChat: {
    title: '临时聊天',
    description: '临时聊天不会出现在你的历史记录中，不会使用或更新记忆，离开聊天后即被删除。',
    placeholder: '在临时聊天中提问',
    use: '使用临时聊天',
    exit: '退出临时聊天',
    badge: '临时',
    badgeHint: '不会保存。关闭时结束，或在你离开期间闲置 10 分钟后结束。',
    newChat: '新建临时聊天',
    unsupportedBackend: '此 Hermes 后端尚不支持临时聊天。请更新它，或关闭临时模式以正常聊天。',
    endChat: '结束临时聊天',
    endTitle: '结束此临时聊天？',
    endDescription: '它将被永久删除，不会保存任何内容。',
    endConfirm: '结束并删除',
    returnTo: '返回临时聊天',
    running: '仍在回复…'
  }
} satisfies TranslationOverrides
