import type { BundledLocale } from './types'

export interface ConversationSearchCopy {
  open: string
  placeholder: string
  searching: string
  empty: string
  close: string
  retry: string
  searchFailed: string
  jumpFailed: string
  unavailable: string
  matches: (start: number, end: number) => string
}

export const conversationSearchCopy: Record<BundledLocale, ConversationSearchCopy> = {
  en: {
    open: 'Search in chat',
    placeholder: 'Search the full transcript…',
    searching: 'Searching stored messages…',
    empty: 'No matches in the stored transcript',
    close: 'Close chat search',
    retry: 'Try again',
    searchFailed: 'Could not search stored messages.',
    jumpFailed: 'Could not show this message. Try again.',
    unavailable: 'Update the connected Hermes runtime to search stored messages.',
    matches: (start, end) => `Matches ${start}–${end}`
  },
  zh: {
    open: '在聊天中搜索',
    placeholder: '搜索完整聊天记录…',
    searching: '正在搜索已保存的消息…',
    empty: '已保存的聊天记录中没有匹配项',
    close: '关闭聊天搜索',
    retry: '重试',
    searchFailed: '无法搜索已保存的消息。',
    jumpFailed: '无法显示此消息，请重试。',
    unavailable: '请更新已连接的 Hermes 运行环境以搜索保存的消息。',
    matches: (start, end) => `匹配项 ${start}–${end}`
  },
  'zh-hant': {
    open: '在聊天中搜尋',
    placeholder: '搜尋完整聊天記錄…',
    searching: '正在搜尋已儲存的訊息…',
    empty: '已儲存的聊天記錄中沒有符合項目',
    close: '關閉聊天搜尋',
    retry: '重試',
    searchFailed: '無法搜尋已儲存的訊息。',
    jumpFailed: '無法顯示此訊息，請重試。',
    unavailable: '請更新已連線的 Hermes 執行環境以搜尋儲存的訊息。',
    matches: (start, end) => `符合項目 ${start}–${end}`
  },
  ja: {
    open: 'チャット内を検索',
    placeholder: '会話全体を検索…',
    searching: '保存済みメッセージを検索中…',
    empty: '保存済みの会話に一致するメッセージはありません',
    close: 'チャット検索を閉じる',
    retry: '再試行',
    searchFailed: '保存済みメッセージを検索できませんでした。',
    jumpFailed: 'このメッセージを表示できませんでした。再試行してください。',
    unavailable: '保存済みメッセージを検索するには、接続先の Hermes ランタイムを更新してください。',
    matches: (start, end) => `一致 ${start}–${end}`
  },
  ar: {
    open: 'البحث في المحادثة',
    placeholder: 'البحث في سجل المحادثة بالكامل…',
    searching: 'جارٍ البحث في الرسائل المحفوظة…',
    empty: 'لا توجد نتائج في سجل المحادثة المحفوظ',
    close: 'إغلاق البحث في المحادثة',
    retry: 'إعادة المحاولة',
    searchFailed: 'تعذر البحث في الرسائل المحفوظة.',
    jumpFailed: 'تعذر عرض هذه الرسالة. أعد المحاولة.',
    unavailable: 'حدّث بيئة Hermes المتصلة للبحث في الرسائل المحفوظة.',
    matches: (start, end) => `النتائج ${start}–${end}`
  },
  ru: {
    open: 'Поиск в чате',
    placeholder: 'Поиск по всей переписке…',
    searching: 'Поиск в сохранённых сообщениях…',
    empty: 'В сохранённой переписке нет совпадений',
    close: 'Закрыть поиск в чате',
    retry: 'Повторить',
    searchFailed: 'Не удалось найти сохранённые сообщения.',
    jumpFailed: 'Не удалось показать сообщение. Повторите попытку.',
    unavailable: 'Обновите подключённую среду Hermes для поиска сохранённых сообщений.',
    matches: (start, end) => `Совпадения ${start}–${end}`
  },
  fr: {
    open: 'Rechercher dans la conversation',
    placeholder: 'Rechercher dans toute la conversation…',
    searching: 'Recherche dans les messages enregistrés…',
    empty: 'Aucun résultat dans la conversation enregistrée',
    close: 'Fermer la recherche',
    retry: 'Réessayer',
    searchFailed: 'Impossible de rechercher dans les messages enregistrés.',
    jumpFailed: 'Impossible d’afficher ce message. Réessayez.',
    unavailable: 'Mettez à jour le runtime Hermes connecté pour rechercher les messages enregistrés.',
    matches: (start, end) => `Résultats ${start}–${end}`
  },
  de: {
    open: 'Im Chat suchen',
    placeholder: 'Gesamten Verlauf durchsuchen…',
    searching: 'Gespeicherte Nachrichten werden durchsucht…',
    empty: 'Keine Treffer im gespeicherten Verlauf',
    close: 'Chatsuche schließen',
    retry: 'Erneut versuchen',
    searchFailed: 'Gespeicherte Nachrichten konnten nicht durchsucht werden.',
    jumpFailed: 'Diese Nachricht konnte nicht angezeigt werden. Versuchen Sie es erneut.',
    unavailable: 'Aktualisieren Sie die verbundene Hermes-Laufzeit, um gespeicherte Nachrichten zu durchsuchen.',
    matches: (start, end) => `Treffer ${start}–${end}`
  },
  es: {
    open: 'Buscar en el chat',
    placeholder: 'Buscar en toda la conversación…',
    searching: 'Buscando en los mensajes guardados…',
    empty: 'No hay coincidencias en la conversación guardada',
    close: 'Cerrar búsqueda del chat',
    retry: 'Reintentar',
    searchFailed: 'No se pudo buscar en los mensajes guardados.',
    jumpFailed: 'No se pudo mostrar este mensaje. Inténtalo de nuevo.',
    unavailable: 'Actualiza el entorno Hermes conectado para buscar mensajes guardados.',
    matches: (start, end) => `Coincidencias ${start}–${end}`
  }
}

export interface FindInPageCopy {
  next: string
  previous: string
}

export const findInPageCopy = {
  en: {
    next: 'Next match',
    previous: 'Previous match'
  },
  de: {
    next: 'Nächster Treffer',
    previous: 'Vorheriger Treffer'
  },
  es: {
    next: 'Siguiente coincidencia',
    previous: 'Coincidencia anterior'
  },
  fr: {
    next: 'Correspondance suivante',
    previous: 'Correspondance précédente'
  },
  ru: {
    next: 'Следующее вхождение',
    previous: 'Предыдущее вхождение'
  },
  zh: {
    next: '下一个匹配',
    previous: '上一个匹配'
  }
} satisfies Partial<Record<BundledLocale, FindInPageCopy>>
