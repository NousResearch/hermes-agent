import type { MenuItemConstructorOptions } from 'electron'

import { windowMenuTemplate } from './window-menu'

export const APPLICATION_MENU_LOCALES = ['en', 'zh', 'zh-hant', 'ja', 'ar', 'ru', 'fr', 'de', 'es'] as const

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
  showSubstitutions: string
  substitutions: string
  textReplacement: string
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
    showSubstitutions: 'Show Substitutions',
    substitutions: 'Substitutions',
    textReplacement: 'Text Replacement',
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
    showSubstitutions: '显示替换',
    substitutions: '替换',
    textReplacement: '文本替换',
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
    showSubstitutions: '顯示替代視窗',
    substitutions: '替代',
    textReplacement: '替代文字',
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
    about: appName => `${appName}について`,
    actualSize: '実際のサイズ',
    bringAllToFront: 'すべてを手前に移動',
    checkForUpdates: 'アップデートを確認…',
    close: '閉じる',
    copy: 'コピー',
    cut: 'カット',
    delete: '削除',
    edit: '編集',
    file: 'ファイル',
    forceReload: '強制再読み込み',
    help: 'ヘルプ',
    hide: appName => `${appName}を非表示`,
    hideOthers: 'ほかを非表示',
    minimize: 'しまう',
    newWindow: '新規ウインドウ',
    openFolder: 'フォルダを開く…',
    paste: 'ペースト',
    pasteAndMatchStyle: 'ペーストしてスタイルを合わせる',
    quit: appName => `${appName}を終了`,
    redo: 'やり直す',
    reload: '再読み込み',
    selectAll: 'すべてを選択',
    services: 'サービス',
    showAll: 'すべてを表示',
    showSubstitutions: '自動置換を表示',
    substitutions: '自動置換',
    textReplacement: 'テキストの置換',
    toggleDevTools: 'デベロッパーツールを切り替える',
    toggleFullScreen: 'フルスクリーンにする',
    undo: '取り消す',
    view: '表示',
    window: 'ウインドウ',
    zoom: '拡大/縮小',
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
    showSubstitutions: 'إظهار البدائل',
    substitutions: 'بدائل',
    textReplacement: 'استبدال النص',
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
    showSubstitutions: 'Показать замены',
    substitutions: 'Замены',
    textReplacement: 'Замена текста',
    toggleDevTools: 'Переключить инструменты разработчика',
    toggleFullScreen: 'Переключить полноэкранный режим',
    undo: 'Отменить',
    view: 'Вид',
    window: 'Окно',
    zoom: 'Масштаб',
    zoomIn: 'Увеличить',
    zoomOut: 'Уменьшить'
  },
  fr: {
    about: appName => `À propos de ${appName}`,
    actualSize: 'Taille réelle',
    bringAllToFront: 'Tout ramener au premier plan',
    checkForUpdates: 'Rechercher des mises à jour…',
    close: 'Fermer',
    copy: 'Copier',
    cut: 'Couper',
    delete: 'Supprimer',
    edit: 'Édition',
    file: 'Fichier',
    forceReload: 'Forcer le rechargement',
    help: 'Aide',
    hide: appName => `Masquer ${appName}`,
    hideOthers: 'Masquer les autres',
    minimize: 'Minimiser',
    newWindow: 'Nouvelle fenêtre',
    openFolder: 'Ouvrir un dossier…',
    paste: 'Coller',
    pasteAndMatchStyle: 'Coller et adapter le style',
    quit: appName => `Quitter ${appName}`,
    redo: 'Rétablir',
    reload: 'Recharger',
    selectAll: 'Tout sélectionner',
    services: 'Services',
    showAll: 'Tout afficher',
    showSubstitutions: 'Afficher les substitutions',
    substitutions: 'Substitutions',
    textReplacement: 'Remplacement de texte',
    toggleDevTools: 'Afficher/masquer les outils de développement',
    toggleFullScreen: 'Activer le mode plein écran',
    undo: 'Annuler',
    view: 'Présentation',
    window: 'Fenêtre',
    zoom: 'Réduire/agrandir',
    zoomIn: 'Zoom avant',
    zoomOut: 'Zoom arrière'
  },
  de: {
    about: appName => `Über ${appName}`,
    actualSize: 'Originalgröße',
    bringAllToFront: 'Alle nach vorne bringen',
    checkForUpdates: 'Nach Updates suchen …',
    close: 'Schließen',
    copy: 'Kopieren',
    cut: 'Ausschneiden',
    delete: 'Löschen',
    edit: 'Bearbeiten',
    file: 'Ablage',
    forceReload: 'Neu laden erzwingen',
    help: 'Hilfe',
    hide: appName => `${appName} ausblenden`,
    hideOthers: 'Andere ausblenden',
    minimize: 'Im Dock ablegen',
    newWindow: 'Neues Fenster',
    openFolder: 'Ordner öffnen …',
    paste: 'Einsetzen',
    pasteAndMatchStyle: 'Einsetzen und Stil anpassen',
    quit: appName => `${appName} beenden`,
    redo: 'Wiederholen',
    reload: 'Neu laden',
    selectAll: 'Alles auswählen',
    services: 'Dienste',
    showAll: 'Alle einblenden',
    showSubstitutions: 'Ersetzungen einblenden',
    substitutions: 'Ersetzungen',
    textReplacement: 'Text ersetzen',
    toggleDevTools: 'Entwicklerwerkzeuge ein-/ausblenden',
    toggleFullScreen: 'Vollbildmodus',
    undo: 'Widerrufen',
    view: 'Darstellung',
    window: 'Fenster',
    zoom: 'Zoomen',
    zoomIn: 'Vergrößern',
    zoomOut: 'Verkleinern'
  },
  es: {
    about: appName => `Acerca de ${appName}`,
    actualSize: 'Tamaño real',
    bringAllToFront: 'Traer todo al frente',
    checkForUpdates: 'Buscar actualizaciones…',
    close: 'Cerrar',
    copy: 'Copiar',
    cut: 'Cortar',
    delete: 'Eliminar',
    edit: 'Edición',
    file: 'Archivo',
    forceReload: 'Forzar recarga',
    help: 'Ayuda',
    hide: appName => `Ocultar ${appName}`,
    hideOthers: 'Ocultar otros',
    minimize: 'Minimizar',
    newWindow: 'Nueva ventana',
    openFolder: 'Abrir carpeta…',
    paste: 'Pegar',
    pasteAndMatchStyle: 'Pegar con el mismo estilo',
    quit: appName => `Salir de ${appName}`,
    redo: 'Rehacer',
    reload: 'Recargar',
    selectAll: 'Seleccionar todo',
    services: 'Servicios',
    showAll: 'Mostrar todo',
    showSubstitutions: 'Mostrar sustituciones',
    substitutions: 'Sustituciones',
    textReplacement: 'Reemplazar texto',
    toggleDevTools: 'Mostrar/ocultar herramientas de desarrollo',
    toggleFullScreen: 'Usar pantalla completa',
    undo: 'Deshacer',
    view: 'Visualización',
    window: 'Ventana',
    zoom: 'Zoom',
    zoomIn: 'Aumentar zoom',
    zoomOut: 'Reducir zoom'
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
      { label: labels.selectAll, role: 'selectAll' },
      ...(isMac
        ? ([
            { type: 'separator' },
            {
              label: labels.substitutions,
              submenu: [
                { label: labels.showSubstitutions, role: 'showSubstitutions' },
                { type: 'separator' },
                { label: labels.textReplacement, role: 'toggleTextReplacement' }
              ]
            }
          ] satisfies MenuItemConstructorOptions[])
        : [])
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
  template.push(windowMenuTemplate(isMac, labels))
  template.push({
    label: labels.help,
    role: 'help',
    submenu: [checkForUpdatesItem]
  })

  return template
}
