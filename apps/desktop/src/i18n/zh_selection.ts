import type { TranslationOverrides } from './define-locale'

export const zhSelectionTranslate = {
  title: '翻译',
  providerNote: '使用你配置的 Hermes 模型。所选文本可能通过该提供商离开本机。',
  target: '首选语言',
  preferredHint: '将保存用于后续翻译。如果文本已匹配非英语目标语言，Hermes 会改为翻译成英语。',
  searchLanguages: '搜索语言…',
  noLanguages: '未找到语言。',
  useLanguageTag: (name, tag) => `使用 ${name}（${tag}）`,
  languageTagHint: '也可以输入语言标签，例如 pt-BR 或 zh-Hant。',
  source: '所选文本',
  translation: '译文',
  translating: '翻译中…',
  failed: '翻译失败',
  emptyResult: '提供商返回了空译文。',
  tooLong: '请选择不超过 4,000 个字符的文本进行翻译。',
  retry: '重试',
  copy: '复制',
  copied: '译文已复制',
  copyFailed: '无法复制译文'
} satisfies TranslationOverrides['selectionTranslate']

export const zhSelectionActions = { readAloud: '朗读', lookUp: '查询', translate: '翻译…', stop: '停止' }

export const zhContextMenu = {
  link: {
    openInApp: '在应用内浏览器中打开',
    openExternal: '在外部浏览器中打开',
    copyUrl: '复制 URL',
    copyResolvedUrl: '复制解析后的 URL'
  },
  image: {
    copyImage: '复制图片',
    copyImageAddress: '复制图片地址',
    saveImageAs: '图片另存为…'
  },
  edit: {
    cut: '剪切',
    paste: '粘贴',
    selectAll: '全选',
    addToDictionary: '添加到词典'
  },
  page: {
    copyPageUrl: '复制页面 URL',
    inspectElement: '检查元素'
  }
} satisfies TranslationOverrides['contextMenu']
