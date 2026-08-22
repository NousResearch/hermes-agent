import type { Locale } from './types'

export interface ThinkingFontCopy {
  ariaLabel: string
  description: string
  title: string
}

export const THINKING_FONT_COPY: Record<Locale, ThinkingFontCopy> = {
  en: {
    ariaLabel: 'Thinking and action text size',
    description: 'Adjust the size of Thinking text and agent action/tool notices without changing normal reply text.',
    title: 'Thinking & action text size'
  },
  zh: {
    ariaLabel: '思考与操作文字大小',
    description: '调整思考文字以及代理操作/工具提示的大小，不影响普通回复文字。',
    title: '思考与操作文字大小'
  },
  'zh-hant': {
    ariaLabel: '思考與操作文字大小',
    description: '調整思考文字以及代理操作/工具提示的大小，不影響一般回覆文字。',
    title: '思考與操作文字大小'
  },
  ja: {
    ariaLabel: '思考とアクションの文字サイズ',
    description: '通常の返信テキストを変えずに、思考テキストとエージェントのアクション/ツール通知のサイズを調整します。',
    title: '思考とアクションの文字サイズ'
  },
  ar: {
    ariaLabel: 'حجم نص التفكير والإجراءات',
    description: 'اضبط حجم نص التفكير وإشعارات إجراءات/أدوات الوكيل دون تغيير نص الرد العادي.',
    title: 'حجم نص التفكير والإجراءات'
  }
}
