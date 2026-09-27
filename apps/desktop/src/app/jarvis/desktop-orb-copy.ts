import type { Locale } from '@/i18n'

export interface DesktopOrbCopy {
  show: string
  hide: string
  drag: string
  open: string
  start: string
  stop: string
  idle: string
  listening: string
  speaking: string
  working: string
  offline: string
  error: string
}

export const desktopOrbCopy: Record<Locale, DesktopOrbCopy> = {
  pl: {
    show: 'Zostaw kulę na pulpicie',
    hide: 'Schowaj kulę',
    drag: 'Przeciągnij kulę',
    open: 'Otwórz Cześka',
    start: 'Rozpocznij rozmowę',
    stop: 'Zakończ rozmowę',
    idle: 'Jestem obok',
    listening: 'Słucham Cię',
    speaking: 'Mówię do Ciebie',
    working: 'Agenci pracują',
    offline: 'Łączę z Cześkiem…',
    error: 'Sprawdź rozmowę w aplikacji'
  },
  en: {
    show: 'Leave the orb on desktop',
    hide: 'Hide orb',
    drag: 'Drag the orb',
    open: 'Open Czesiek',
    start: 'Start conversation',
    stop: 'End conversation',
    idle: 'Here with you',
    listening: 'Listening',
    speaking: 'Speaking',
    working: 'Agents are working',
    offline: 'Connecting to Czesiek…',
    error: 'Check the conversation in the app'
  },
  ja: {
    show: 'デスクトップのオーブ',
    hide: 'オーブを隠す',
    drag: 'オーブをドラッグ',
    open: 'Czesiekを開く',
    start: '会話を開始',
    stop: '会話を終了',
    idle: 'そばにいます',
    listening: '聞いています',
    speaking: '話しています',
    working: 'エージェントが作業中',
    offline: '接続中…',
    error: 'アプリで会話を確認してください'
  },
  zh: {
    show: '桌面光球',
    hide: '隐藏光球',
    drag: '拖动光球',
    open: '打开Czesiek',
    start: '开始对话',
    stop: '结束对话',
    idle: '陪在你身边',
    listening: '正在聆听',
    speaking: '正在说话',
    working: '智能体正在工作',
    offline: '正在连接…',
    error: '请在应用中查看对话'
  },
  'zh-hant': {
    show: '桌面光球',
    hide: '隱藏光球',
    drag: '拖動光球',
    open: '開啟Czesiek',
    start: '開始對話',
    stop: '結束對話',
    idle: '陪在你身邊',
    listening: '正在聆聽',
    speaking: '正在說話',
    working: '智慧體正在工作',
    offline: '正在連線…',
    error: '請在應用程式中查看對話'
  },
  ru: {
    show: 'Сфера на рабочем столе',
    hide: 'Скрыть сферу',
    drag: 'Перетащить сферу',
    open: 'Открыть Czesiek',
    start: 'Начать разговор',
    stop: 'Завершить разговор',
    idle: 'Я рядом',
    listening: 'Слушаю',
    speaking: 'Говорю',
    working: 'Агенты работают',
    offline: 'Подключение…',
    error: 'Проверьте разговор в приложении'
  },
  ar: {
    show: 'الكرة على سطح المكتب',
    hide: 'إخفاء الكرة',
    drag: 'اسحب الكرة',
    open: 'افتح Czesiek',
    start: 'بدء المحادثة',
    stop: 'إنهاء المحادثة',
    idle: 'أنا بجانبك',
    listening: 'أستمع إليك',
    speaking: 'أتحدث إليك',
    working: 'الوكلاء يعملون',
    offline: 'جارٍ الاتصال…',
    error: 'تحقق من المحادثة في التطبيق'
  }
}
