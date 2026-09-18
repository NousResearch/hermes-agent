/** Files copy retained from #104199, including all six supported locales. */
import { useI18n } from '@hermes/plugin-sdk'

export const CANONICAL_FILES_LOCALES = {
  en: {
    attachmentDownloadFailed: 'This attachment could not be downloaded.',
    sharedFiles: 'Files',
    searchSharedFiles: 'Search files',
    sharedFilesLoading: 'Loading files',
    sharedFilesError: 'Files could not be loaded.',
    sharedFilesExpired: 'Refresh the file list to continue.',
    sharedFilesOffline: 'Files are temporarily unavailable.',
    sharedFilesUnavailable: "File browsing isn't available for this Group Chat yet.",
    sharedFilesEmpty: 'No files shared yet.',
    sharedFilesPageEmpty: 'No files on this page',
    sharedFilesNoResults: 'No matching files.',
    sharedFilesRetry: 'Retry',
    olderFiles: 'Older',
    newerFiles: 'Newer',
    returnToLatest: 'Show latest',
    showLatest: 'Show latest',
    filesReconnected: 'Reconnected',
    filesClearSearch: 'Clear search',
    filesRefresh: 'Refresh list',
    fileGone: 'This file is no longer available.',
    fileVerificationFailed: "This file couldn't be verified. Nothing was downloaded.",
    fileTimeout: 'The download timed out.',
    filesAccessUnavailable: 'Files are unavailable for this Group Chat.'
  },
  ja: {
    attachmentDownloadFailed: 'この添付ファイルをダウンロードできませんでした。',
    sharedFiles: 'ファイル',
    searchSharedFiles: 'ファイルを検索',
    sharedFilesLoading: 'ファイルを読み込み中',
    sharedFilesError: 'ファイルを読み込めませんでした。',
    sharedFilesExpired: 'このファイル一覧の有効期限が切れました。',
    sharedFilesOffline: 'ファイルを一時的に利用できません。',
    sharedFilesUnavailable: 'このグループチャットではまだファイルを一覧表示できません。',
    sharedFilesEmpty: '共有されたファイルはまだありません。',
    sharedFilesPageEmpty: 'このページにファイルはありません',
    sharedFilesNoResults: '一致するファイルはありません',
    sharedFilesRetry: '再試行',
    olderFiles: '古いファイル',
    newerFiles: '新しいファイル',
    returnToLatest: '最新に戻る',
    showLatest: '最新を表示',
    filesReconnected: '再接続しました',
    filesClearSearch: '検索をクリア',
    filesRefresh: '一覧を更新',
    fileGone: 'このファイルは利用できなくなりました。',
    fileVerificationFailed: 'このファイルを検証できませんでした。何もダウンロードされていません。',
    fileTimeout: 'ダウンロードがタイムアウトしました。',
    filesAccessUnavailable: 'このグループチャットのファイルを利用できません。'
  },
  zh: {
    attachmentDownloadFailed: '无法下载此附件。',
    sharedFiles: '文件',
    searchSharedFiles: '搜索文件',
    sharedFilesLoading: '正在加载文件',
    sharedFilesError: '无法加载文件。',
    sharedFilesExpired: '此文件列表已过期。',
    sharedFilesOffline: '文件暂时不可用。',
    sharedFilesUnavailable: '此群聊尚不支持文件浏览。',
    sharedFilesEmpty: '尚未共享任何文件。',
    sharedFilesPageEmpty: '此页没有文件',
    sharedFilesNoResults: '没有匹配的文件',
    sharedFilesRetry: '重试',
    olderFiles: '较早的文件',
    newerFiles: '较新的文件',
    returnToLatest: '返回最新内容',
    showLatest: '显示最新内容',
    filesReconnected: '已重新连接',
    filesClearSearch: '清除搜索',
    filesRefresh: '刷新列表',
    fileGone: '此文件已不可用。',
    fileVerificationFailed: '无法验证此文件。未下载任何内容。',
    fileTimeout: '下载超时。',
    filesAccessUnavailable: '此群聊的文件不可用。'
  },
  'zh-hant': {
    attachmentDownloadFailed: '無法下載此附件。',
    sharedFiles: '檔案',
    searchSharedFiles: '搜尋檔案',
    sharedFilesLoading: '正在載入檔案',
    sharedFilesError: '無法載入檔案。',
    sharedFilesExpired: '此檔案清單已過期。',
    sharedFilesOffline: '檔案暫時無法使用。',
    sharedFilesUnavailable: '此群組聊天尚未支援檔案瀏覽。',
    sharedFilesEmpty: '尚未共享任何檔案。',
    sharedFilesPageEmpty: '此頁沒有檔案',
    sharedFilesNoResults: '找不到相符的檔案',
    sharedFilesRetry: '重試',
    olderFiles: '較舊的檔案',
    newerFiles: '較新的檔案',
    returnToLatest: '返回最新內容',
    showLatest: '顯示最新內容',
    filesReconnected: '已重新連線',
    filesClearSearch: '清除搜尋',
    filesRefresh: '重新整理清單',
    fileGone: '此檔案已無法使用。',
    fileVerificationFailed: '無法驗證此檔案。未下載任何內容。',
    fileTimeout: '下載逾時。',
    filesAccessUnavailable: '此群組聊天的檔案無法使用。'
  },
  ar: {
    attachmentDownloadFailed: 'تعذر تنزيل هذا المرفق.',
    sharedFiles: 'الملفات',
    searchSharedFiles: 'البحث في الملفات',
    sharedFilesLoading: 'جارٍ تحميل الملفات',
    sharedFilesError: 'تعذر تحميل الملفات.',
    sharedFilesExpired: 'انتهت صلاحية قائمة الملفات هذه.',
    sharedFilesOffline: 'الملفات غير متاحة مؤقتاً.',
    sharedFilesUnavailable: 'تصفح الملفات غير متاح لهذه المحادثة الجماعية بعد.',
    sharedFilesEmpty: 'لم تتم مشاركة أي ملفات بعد.',
    sharedFilesPageEmpty: 'لا توجد ملفات في هذه الصفحة',
    sharedFilesNoResults: 'لا توجد ملفات مطابقة',
    sharedFilesRetry: 'إعادة المحاولة',
    olderFiles: 'ملفات أقدم',
    newerFiles: 'ملفات أحدث',
    returnToLatest: 'العودة إلى الأحدث',
    showLatest: 'عرض الأحدث',
    filesReconnected: 'تمت إعادة الاتصال',
    filesClearSearch: 'مسح البحث',
    filesRefresh: 'تحديث القائمة',
    fileGone: 'لم يعد هذا الملف متاحاً.',
    fileVerificationFailed: 'تعذر التحقق من هذا الملف. لم يتم تنزيل أي شيء.',
    fileTimeout: 'انتهت مهلة التنزيل.',
    filesAccessUnavailable: 'الملفات غير متاحة لهذه المحادثة الجماعية.'
  },
  ru: {
    attachmentDownloadFailed: 'Не удалось скачать это вложение.',
    sharedFiles: 'Файлы',
    searchSharedFiles: 'Поиск файлов',
    sharedFilesLoading: 'Загрузка файлов',
    sharedFilesError: 'Не удалось загрузить файлы.',
    sharedFilesExpired: 'Срок действия этого списка файлов истёк.',
    sharedFilesOffline: 'Файлы временно недоступны.',
    sharedFilesUnavailable: 'Просмотр файлов пока недоступен для этого группового чата.',
    sharedFilesEmpty: 'Файлами ещё не делились.',
    sharedFilesPageEmpty: 'На этой странице нет файлов',
    sharedFilesNoResults: 'Подходящие файлы не найдены',
    sharedFilesRetry: 'Повторить',
    olderFiles: 'Более старые файлы',
    newerFiles: 'Более новые файлы',
    returnToLatest: 'Вернуться к последним',
    showLatest: 'Показать последние',
    filesReconnected: 'Соединение восстановлено',
    filesClearSearch: 'Очистить поиск',
    filesRefresh: 'Обновить список',
    fileGone: 'Этот файл больше недоступен.',
    fileVerificationFailed: 'Не удалось проверить этот файл. Ничего не скачано.',
    fileTimeout: 'Время ожидания скачивания истекло.',
    filesAccessUnavailable: 'Файлы недоступны для этого группового чата.'
  }
} satisfies Record<
  string,
  Record<
    | 'attachmentDownloadFailed'
    | 'sharedFiles'
    | 'searchSharedFiles'
    | 'sharedFilesLoading'
    | 'sharedFilesError'
    | 'sharedFilesExpired'
    | 'sharedFilesOffline'
    | 'sharedFilesUnavailable'
    | 'sharedFilesEmpty'
    | 'sharedFilesPageEmpty'
    | 'sharedFilesNoResults'
    | 'sharedFilesRetry'
    | 'olderFiles'
    | 'newerFiles'
    | 'returnToLatest'
    | 'showLatest'
    | 'filesReconnected'
    | 'filesClearSearch'
    | 'filesRefresh'
    | 'fileGone'
    | 'fileVerificationFailed'
    | 'fileTimeout'
    | 'filesAccessUnavailable',
    string
  >
>

export function useCanonicalFilesLabels() {
  const { locale, t } = useI18n()

  return {
    ...(CANONICAL_FILES_LOCALES[locale as keyof typeof CANONICAL_FILES_LOCALES] ?? CANONICAL_FILES_LOCALES.en),
    download: t.fileMenu.download,
    locale
  }
}
