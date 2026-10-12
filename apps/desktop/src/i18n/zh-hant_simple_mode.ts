import type { TranslationOverrides } from './define-locale'

// Simple-mode surfaces (interface-mode picker, chat starters), spread into zh-hant.ts.
export const zhHantSimpleMode = {
  interfaceMode: {
    title: '介面模式',
    hint: '只改變顯示的內容，不改變 Hermes 的能力。',
    sessionNote: '由簡潔模式設定。此處的變更僅在本次工作階段內生效；切換到進階模式即可保留為你的設定。',
    switchToAdvanced: '切換到進階',
    simple: {
      label: '簡潔',
      description: '用於與 Hermes 對話。只有側邊欄和聊天；沒有終端機、檔案或差異面板。'
    },
    advanced: {
      label: '進階',
      description: '面向開發者。終端機、檔案、差異、狀態列和版面配置，按你的設定顯示。'
    }
  },
  chatStarters: {
    label: '可以從這些開始',
    intro: '問我任何事，或從下面這些開始。',
    email: {
      title: '幫我寫一封電子郵件',
      prompt: '幫我寫一封電子郵件。收件人和我想說的內容是： '
    },
    dinners: {
      title: '規劃這週的晚餐',
      prompt: '幫我規劃這週的五頓簡單晚餐，並把所有食材整理成一份購物清單。'
    },
    summarize: {
      title: '摘要一份文件',
      prompt: '我會附上一個檔案——請告訴我重點。'
    },
    lookup: {
      title: '幫我查點東西',
      prompt: '幫我查一下這個，並告訴我你找到了什麼： '
    }
  }
} satisfies Pick<TranslationOverrides, 'interfaceMode' | 'chatStarters'>
