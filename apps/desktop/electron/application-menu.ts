import type { MenuItemConstructorOptions } from 'electron'

export const APPLICATION_MENU_LOCALES = ['en', 'zh', 'zh-hant', 'ja', 'ar', 'ru'] as const

export type ApplicationMenuLocale = (typeof APPLICATION_MENU_LOCALES)[number]

interface ApplicationMenuLabels {
  about: (appName: string) => string
  actualSize: string
  bringAllToFront: string
  checkForUpdates: string
  close: string
  copy: string
  cut: string
  delete: string
  edit: string
  file: string
  forceReload: string
  help: string
  hide: (appName: string) => string
  hideOthers: string
  minimize: string
  newWindow: string
  openFolder: string
  paste: string
  pasteAndMatchStyle: string
  quit: (appName: string) => string
  redo: string
  reload: string
  selectAll: string
  services: string
  showAll: string
  toggleDevTools: string
  toggleFullScreen: string
  undo: string
  view: string
  window: string
  zoom: string
  zoomIn: string
  zoomOut: string
}

const APPLICATION_MENU_LABELS: Record<ApplicationMenuLocale, ApplicationMenuLabels> = {
  en: {
    about: appName => `About ${appName}`,
    actualSize: 'Actual Size',
    bringAllToFront: 'Bring All to Front',
    checkForUpdates: 'Check for Updates…',
    close: 'Close',
    copy: 'Copy',
    cut: 'Cut',
    delete: 'Delete',
    edit: 'Edit',
    file: 'File',
    forceReload: 'Force Reload',
    help: 'Help',
    hide: appName => `Hide ${appName}`,
    hideOthers: 'Hide Others',
    minimize: 'Minimize',
    newWindow: 'New Window',
    openFolder: 'Open Folder…',
    paste: 'Paste',
    pasteAndMatchStyle: 'Paste and Match Style',
    quit: appName => `Quit ${appName}`,
    redo: 'Redo',
    reload: 'Reload',
    selectAll: 'Select All',
    services: 'Services',
    showAll: 'Show All',
    toggleDevTools: 'Toggle Developer Tools',
    toggleFullScreen: 'Toggle Full Screen',
    undo: 'Undo',
    view: 'View',
    window: 'Window',
    zoom: 'Zoom',
    zoomIn: 'Zoom In',
    zoomOut: 'Zoom Out'
  },
  zh: {
    about: appName => `关于 ${appName}`,
    actualSize: '实际大小',
    bringAllToFront: '全部移到前面',
    checkForUpdates: '检查更新…',
    close: '关闭',
    copy: '复制',
    cut: '剪切',
    delete: '删除',
    edit: '编辑',
    file: '文件',
    forceReload: '强制重新加载',
    help: '帮助',
    hide: appName => `隐藏 ${appName}`,
    hideOthers: '隐藏其他',
    minimize: '最小化',
    newWindow: '新建窗口',
    openFolder: '打开文件夹…',
    paste: '粘贴',
    pasteAndMatchStyle: '粘贴并匹配样式',
    quit: appName => `退出 ${appName}`,
    redo: '重做',
    reload: '重新加载',
    selectAll: '全选',
    services: '服务',
    showAll: '全部显示',
    toggleDevTools: '切换开发者工具',
    toggleFullScreen: '切换全屏幕',
    undo: '撤销',
    view: '显示',
    window: '窗口',
    zoom: '缩放',
    zoomIn: '放大',
    zoomOut: '缩小'
  },
  'zh-hant': {
    about: appName => `關於 ${appName}`,
    actualSize: '實際大小',
    bringAllToFront: '全部移到最前面',
    checkForUpdates: '檢查更新…',
    close: '關閉',
    copy: '拷貝',
    cut: '剪下',
    delete: '刪除',
    edit: '編輯',
    file: '檔案',
    forceReload: '強制重新載入',
    help: '說明',
    hide: appName => `隱藏 ${appName}`,
    hideOthers: '隱藏其他',
    minimize: '最小化',
    newWindow: '新增視窗',
    openFolder: '開啟資料夾…',
    paste: '貼上',
    pasteAndMatchStyle: '貼上並符合樣式',
    quit: appName => `結束 ${appName}`,
    redo: '重做',
    reload: '重新載入',
    selectAll: '全選',
    services: '服務',
    showAll: '全部顯示',
    toggleDevTools: '切換開發者工具',
    toggleFullScreen: '切換全螢幕',
    undo: '還原',
    view: '顯示方式',
    window: '視窗',
    zoom: '縮放',
    zoomIn: '放大',
    zoomOut: '縮小'
  },
  ja: {
    about: appName => `${appName} について`,
    actualSize: '実際のサイズ',
    bringAllToFront: 'すべてを手前に移動',
    checkForUpdates: 'アップデートを確認…',
    close: '閉じる',
    copy: 'コピー',
    cut: '切り取り',
    delete: '削除',
    edit: '編集',
    file: 'ファイル',
    forceReload: '強制再読み込み',
    help: 'ヘルプ',
    hide: appName => `${appName} を隠す`,
    hideOthers: 'ほかを隠す',
    minimize: 'しまう',
    newWindow: '新規ウインドウ',
    openFolder: 'フォルダを開く…',
    paste: 'ペースト',
    pasteAndMatchStyle: 'ペーストしてスタイルを合わせる',
    quit: appName => `${appName} を終了`,
    redo: 'やり直す',
    reload: '再読み込み',
    selectAll: 'すべてを選択',
    services: 'サービス',
    showAll: 'すべてを表示',
    toggleDevTools: 'デベロッパーツールを切り替える',
    toggleFullScreen: 'フルスクリーンにする',
    undo: '取り消す',
    view: '表示',
    window: 'ウインドウ',
    zoom: '拡大／縮小',
    zoomIn: '拡大',
    zoomOut: '縮小'
  },
  ar: {
    about: appName => `حول ${appName}`,
    actualSize: 'الحجم الفعلي',
    bringAllToFront: 'إحضار الكل إلى المقدمة',
    checkForUpdates: 'التحقق من وجود تحديثات…',
    close: 'إغلاق',
    copy: 'نسخ',
    cut: 'قص',
    delete: 'حذف',
    edit: 'تحرير',
    file: 'ملف',
    forceReload: 'فرض إعادة التحميل',
    help: 'مساعدة',
    hide: appName => `إخفاء ${appName}`,
    hideOthers: 'إخفاء الآخرين',
    minimize: 'تصغير',
    newWindow: 'نافذة جديدة',
    openFolder: 'فتح مجلد…',
    paste: 'لصق',
    pasteAndMatchStyle: 'لصق ومطابقة النمط',
    quit: appName => `إنهاء ${appName}`,
    redo: 'إعادة',
    reload: 'إعادة التحميل',
    selectAll: 'تحديد الكل',
    services: 'الخدمات',
    showAll: 'إظهار الكل',
    toggleDevTools: 'تبديل أدوات المطور',
    toggleFullScreen: 'تبديل ملء الشاشة',
    undo: 'تراجع',
    view: 'عرض',
    window: 'نافذة',
    zoom: 'تكبير/تصغير',
    zoomIn: 'تكبير',
    zoomOut: 'تصغير'
  },
  ru: {
    about: appName => `О программе ${appName}`,
    actualSize: 'Фактический размер',
    bringAllToFront: 'Все окна — на передний план',
    checkForUpdates: 'Проверить обновления…',
    close: 'Закрыть',
    copy: 'Копировать',
    cut: 'Вырезать',
    delete: 'Удалить',
    edit: 'Правка',
    file: 'Файл',
    forceReload: 'Принудительно перезагрузить',
    help: 'Справка',
    hide: appName => `Скрыть ${appName}`,
    hideOthers: 'Скрыть остальные',
    minimize: 'Свернуть',
    newWindow: 'Новое окно',
    openFolder: 'Открыть папку…',
    paste: 'Вставить',
    pasteAndMatchStyle: 'Вставить в текущем стиле',
    quit: appName => `Завершить ${appName}`,
    redo: 'Повторить',
    reload: 'Перезагрузить',
    selectAll: 'Выбрать все',
    services: 'Службы',
    showAll: 'Показать все',
    toggleDevTools: 'Переключить инструменты разработчика',
    toggleFullScreen: 'Переключить полноэкранный режим',
    undo: 'Отменить',
    view: 'Вид',
    window: 'Окно',
    zoom: 'Масштаб',
    zoomIn: 'Увеличить',
    zoomOut: 'Уменьшить'
  }
}

export interface ApplicationMenuActions {
  checkForUpdates: () => void
  close: () => void
  createWindow: () => void
  openFolder: () => void
  reloadPreview: () => void
  resetZoom: () => void
  showAbout: () => void
  toggleDevTools: (window: Electron.BaseWindow | undefined) => void
  zoomIn: () => void
  zoomOut: () => void
}

interface BuildApplicationMenuOptions {
  actions: ApplicationMenuActions
  appName: string
  isMac: boolean
  locale: ApplicationMenuLocale
}

export function isApplicationMenuLocale(value: unknown): value is ApplicationMenuLocale {
  return typeof value === 'string' && (APPLICATION_MENU_LOCALES as readonly string[]).includes(value)
}

export function buildApplicationMenuTemplate({
  actions,
  appName,
  isMac,
  locale
}: BuildApplicationMenuOptions): MenuItemConstructorOptions[] {
  const labels = APPLICATION_MENU_LABELS[locale]
  const template: MenuItemConstructorOptions[] = []

  const checkForUpdatesItem: MenuItemConstructorOptions = {
    label: labels.checkForUpdates,
    click: actions.checkForUpdates
  }

  if (isMac) {
    template.push({
      label: appName,
      submenu: [
        { label: labels.about(appName), click: actions.showAbout },
        checkForUpdatesItem,
        { type: 'separator' },
        { label: labels.services, role: 'services' },
        { type: 'separator' },
        { label: labels.hide(appName), role: 'hide' },
        { label: labels.hideOthers, role: 'hideOthers' },
        { label: labels.showAll, role: 'unhide' },
        { type: 'separator' },
        { label: labels.quit(appName), role: 'quit' }
      ]
    })
  }

  template.push({
    label: labels.file,
    submenu: [
      { click: actions.createWindow, label: labels.newWindow },
      { click: actions.openFolder, label: labels.openFolder },
      { type: 'separator' },
      isMac ? { click: actions.close, label: labels.close } : { role: 'quit' }
    ]
  })
  template.push({
    label: labels.edit,
    submenu: [
      { label: labels.undo, role: 'undo' },
      { label: labels.redo, role: 'redo' },
      { type: 'separator' },
      { label: labels.cut, role: 'cut' },
      { label: labels.copy, role: 'copy' },
      { label: labels.paste, role: 'paste' },
      { label: labels.pasteAndMatchStyle, role: 'pasteAndMatchStyle' },
      { label: labels.delete, role: 'delete' },
      { label: labels.selectAll, role: 'selectAll' }
    ]
  })
  template.push({
    label: labels.view,
    submenu: [
      { click: actions.reloadPreview, label: labels.reload },
      { label: labels.forceReload, role: 'forceReload' },
      {
        label: labels.toggleDevTools,
        accelerator: isMac ? 'Alt+Cmd+I' : 'Ctrl+Shift+I',
        click: (_menuItem, browserWindow) => actions.toggleDevTools(browserWindow)
      },
      { type: 'separator' },
      {
        label: labels.actualSize,
        accelerator: 'CommandOrControl+0',
        click: actions.resetZoom
      },
      {
        label: labels.zoomIn,
        accelerator: 'CommandOrControl+Plus',
        click: actions.zoomIn
      },
      {
        label: labels.zoomOut,
        accelerator: 'CommandOrControl+-',
        click: actions.zoomOut
      },
      { type: 'separator' },
      { label: labels.toggleFullScreen, role: 'togglefullscreen' }
    ]
  })
  template.push({
    label: labels.window,
    submenu: isMac
      ? [
          { label: labels.minimize, role: 'minimize' },
          { label: labels.zoom, role: 'zoom' },
          { label: labels.bringAllToFront, role: 'front' }
        ]
      : [{ label: labels.minimize, role: 'minimize' }, { role: 'close' }]
  })
  template.push({
    label: labels.help,
    role: 'help',
    submenu: [checkForUpdatesItem]
  })

  return template
}
