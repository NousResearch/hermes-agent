// Labels for the native right-click context menu, keyed by the renderer's UI
// language (display.language). Electron only localizes role labels (Cut/Copy/
// Paste/Select All) from the SYSTEM menus on macOS — on Windows/Linux they are
// hardcoded English — and the app's language setting can differ from the OS
// locale anyway, so the renderer reports its locale over IPC and the menu
// builds from this table.

export interface ContextMenuLabels {
  openImage: string
  copyImage: string
  copyImageAddress: string
  saveImageAs: string
  openLink: string
  copyLink: string
  addToDictionary: string
  cut: string
  copy: string
  paste: string
  selectAll: string
}

const EN: ContextMenuLabels = {
  openImage: 'Open Image',
  copyImage: 'Copy Image',
  copyImageAddress: 'Copy Image Address',
  saveImageAs: 'Save Image As...',
  openLink: 'Open Link',
  copyLink: 'Copy Link',
  addToDictionary: 'Add to dictionary',
  cut: 'Cut',
  copy: 'Copy',
  paste: 'Paste',
  selectAll: 'Select All'
}

const LABELS: Record<string, ContextMenuLabels> = {
  en: EN,
  zh: {
    openImage: '打开图片',
    copyImage: '复制图片',
    copyImageAddress: '复制图片地址',
    saveImageAs: '图片另存为…',
    openLink: '打开链接',
    copyLink: '复制链接',
    addToDictionary: '添加到词典',
    cut: '剪切',
    copy: '复制',
    paste: '粘贴',
    selectAll: '全选'
  },
  'zh-hant': {
    openImage: '開啟圖片',
    copyImage: '複製圖片',
    copyImageAddress: '複製圖片位址',
    saveImageAs: '另存圖片…',
    openLink: '開啟連結',
    copyLink: '複製連結',
    addToDictionary: '加入字典',
    cut: '剪下',
    copy: '複製',
    paste: '貼上',
    selectAll: '全選'
  },
  ja: {
    openImage: '画像を開く',
    copyImage: '画像をコピー',
    copyImageAddress: '画像アドレスをコピー',
    saveImageAs: '名前を付けて画像を保存…',
    openLink: 'リンクを開く',
    copyLink: 'リンクをコピー',
    addToDictionary: '辞書に追加',
    cut: '切り取り',
    copy: 'コピー',
    paste: '貼り付け',
    selectAll: 'すべて選択'
  }
}

/** Resolve a renderer UI locale to its label set; unknown values fall back to English. */
export function getContextMenuLabels(locale: unknown): ContextMenuLabels {
  return (typeof locale === 'string' && LABELS[locale.trim().toLowerCase()]) || EN
}
