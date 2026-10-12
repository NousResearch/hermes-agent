import type { TranslationOverrides } from './define-locale'

// Simple-mode surfaces (interface-mode picker, chat starters), spread into zh.ts.
export const zhSimpleMode = {
  interfaceMode: {
    title: '界面模式',
    hint: '只改变显示的内容，不改变 Hermes 的能力。',
    sessionNote: '由简洁模式设定。此处的更改仅在本次会话内生效；切换到高级模式即可保留为你的设置。',
    switchToAdvanced: '切换到高级模式',
    simple: {
      label: '简洁',
      description: '用于与 Hermes 对话。只有侧边栏和聊天；没有终端、文件或差异面板。'
    },
    advanced: {
      label: '高级',
      description: '面向开发者。终端、文件、差异、状态栏和布局，按你的设置显示。'
    }
  },
  chatStarters: {
    label: '可以先试试这些',
    intro: '问我任何事，或者从下面这些开始。',
    email: {
      title: '帮我写一封邮件',
      prompt: '帮我写一封邮件。收件人和我想说的内容是： '
    },
    dinners: {
      title: '安排这周的晚餐',
      prompt: '帮我安排这周的五顿简单晚餐，并把所需食材整理成一份购物清单。'
    },
    summarize: {
      title: '总结一份文档',
      prompt: '我会附上一个文件——请告诉我其中的要点。'
    },
    lookup: {
      title: '帮我查点东西',
      prompt: '帮我查一下这个，并告诉我你查到了什么： '
    }
  }
} satisfies Pick<TranslationOverrides, 'interfaceMode' | 'chatStarters'>
