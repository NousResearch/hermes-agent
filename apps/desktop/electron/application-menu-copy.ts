export type ApplicationMenuLocale = 'ar' | 'en'

export interface ApplicationMenuCopy {
  // These three carry the app name, so `HERMES_DESKTOP_APP_NAME` rebrands keep
  // working and the English wording matches what Electron's own roles produce.
  about: (appName: string) => string
  hide: (appName: string) => string
  quit: (appName: string) => string
  actualSize: string
  checkForUpdates: string
  close: string
  copy: string
  cut: string
  delete: string
  edit: string
  file: string
  forceReload: string
  front: string
  fullscreen: string
  help: string
  hideOthers: string
  minimize: string
  paste: string
  redo: string
  reload: string
  selectAll: string
  services: string
  toggleDevTools: string
  undo: string
  unhide: string
  view: string
  window: string
  zoom: string
  zoomIn: string
  zoomOut: string
}

const ENGLISH_COPY: ApplicationMenuCopy = {
  about: appName => `About ${appName}`,
  actualSize: 'Actual Size',
  checkForUpdates: 'Check for Updates…',
  close: 'Close',
  copy: 'Copy',
  cut: 'Cut',
  delete: 'Delete',
  edit: 'Edit',
  file: 'File',
  forceReload: 'Force Reload',
  front: 'Bring All to Front',
  fullscreen: 'Toggle Full Screen',
  help: 'Help',
  hide: appName => `Hide ${appName}`,
  hideOthers: 'Hide Others',
  minimize: 'Minimize',
  paste: 'Paste',
  quit: appName => `Quit ${appName}`,
  redo: 'Redo',
  reload: 'Reload',
  selectAll: 'Select All',
  services: 'Services',
  toggleDevTools: 'Toggle Developer Tools',
  undo: 'Undo',
  unhide: 'Show All',
  view: 'View',
  window: 'Window',
  zoom: 'Zoom',
  zoomIn: 'Zoom In',
  zoomOut: 'Zoom Out'
}

const ARABIC_COPY: ApplicationMenuCopy = {
  about: appName => `حول ${appName}`,
  actualSize: 'الحجم الفعلي',
  checkForUpdates: 'التحقق من التحديثات…',
  close: 'إغلاق',
  copy: 'نسخ',
  cut: 'قص',
  delete: 'حذف',
  edit: 'تحرير',
  file: 'ملف',
  forceReload: 'فرض إعادة التحميل',
  front: 'إحضار الكل إلى المقدمة',
  fullscreen: 'تبديل ملء الشاشة',
  help: 'مساعدة',
  hide: appName => `إخفاء ${appName}`,
  hideOthers: 'إخفاء التطبيقات الأخرى',
  minimize: 'تصغير',
  paste: 'لصق',
  quit: appName => `إنهاء ${appName}`,
  redo: 'إعادة',
  reload: 'إعادة التحميل',
  selectAll: 'تحديد الكل',
  services: 'الخدمات',
  toggleDevTools: 'تبديل أدوات المطور',
  undo: 'تراجع',
  unhide: 'إظهار الكل',
  view: 'عرض',
  window: 'نافذة',
  zoom: 'تكبير',
  zoomIn: 'تكبير العرض',
  zoomOut: 'تصغير العرض'
}

export function normalizeApplicationMenuLocale(value: unknown): ApplicationMenuLocale {
  return typeof value === 'string' && value.toLowerCase().startsWith('ar') ? 'ar' : 'en'
}

export function applicationMenuCopy(locale: ApplicationMenuLocale): ApplicationMenuCopy {
  return locale === 'ar' ? ARABIC_COPY : ENGLISH_COPY
}
