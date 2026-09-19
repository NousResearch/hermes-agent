import type { TranslationOverrides } from './define-locale'

export const zhHantCommon = {
  common: {
    apply: '套用',
    back: '返回',
    save: '儲存',
    saving: '儲存中…',
    cancel: '取消',
    change: '變更',
    choose: '選擇',
    clear: '清除',
    close: '關閉',
    collapse: '收合',
    confirm: '確認',
    connect: '連線',
    connecting: '連線中',
    continue: '繼續',
    copied: '已複製',
    copy: '複製',
    copyFailed: '複製失敗',
    delete: '刪除',
    docs: '文件',
    done: '完成',
    error: '錯誤',
    expand: '展開',
    failed: '失敗',
    formatJson: '格式化 JSON',
    free: '免費',
    loading: '載入中…',
    notSet: '未設定',
    refresh: '重新整理',
    remove: '移除',
    replace: '取代',
    retry: '重試',
    run: '執行',
    send: '傳送',
    set: '設定',
    skip: '略過',
    update: '更新',
    tryHint: term => `試試「${term}」`,
    on: '開啟',
    off: '關閉'
  },

  billingBlock: {
    titleNous: 'Nous 額度已用盡',
    titleProvider: provider => `額度已用盡 — ${provider}`,
    fallbackMessage: '您的帳戶額度已用盡。請儲值以繼續使用。',
    openBilling: '開啟帳單',
    addCredits: '新增額度',
    dismiss: '忽略'
  },

  ui: {
    search: {
      clear: '清除搜尋'
    },
    pagination: {
      label: '分頁',
      previous: '上一頁',
      previousAria: '前往上一頁',
      next: '下一頁',
      nextAria: '前往下一頁'
    },
    sidebar: {
      title: '側邊欄',
      description: '顯示行動裝置側邊欄。',
      toggle: open => `${open ? '顯示' : '隱藏'}側邊欄`
    }
  }
} satisfies Pick<TranslationOverrides, 'common' | 'billingBlock' | 'ui'>
