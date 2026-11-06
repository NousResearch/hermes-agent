import { getPlayerService, type PlayerSource } from '../src/lib/player-services'

export interface NativePlayerCopy {
  player: string
  dock: string
  togglePin: string
  fullscreen: string
  exitFullscreen: string
  close: string
  signIn: string
  allowSession: string
  deny: string
  locationTitle: string
  locationMessage: string
  protectedTitle: string
  protectedMessage: string
}

const COPY: Record<'en' | 'ja' | 'zh' | 'zh-hant' | 'ar' | 'ru', NativePlayerCopy> = {
  en: {
    player: 'Player',
    dock: 'Dock in Hermes',
    togglePin: 'Toggle pin preference',
    fullscreen: 'Fullscreen',
    exitFullscreen: 'Exit fullscreen',
    close: 'Close player',
    signIn: 'Sign in to YouTube',
    allowSession: 'Allow for this session',
    deny: 'Deny',
    locationTitle: 'YouTube TV location',
    locationMessage: 'YouTube TV requests your location to verify service availability.',
    protectedTitle: 'Protected media access',
    protectedMessage: 'YouTube requests protected media access for playback.'
  },
  ja: {
    player: 'プレーヤー',
    dock: 'Hermes にドック',
    togglePin: 'ピン留め設定を切り替え',
    fullscreen: '全画面表示',
    exitFullscreen: '全画面表示を終了',
    close: 'プレーヤーを閉じる',
    signIn: 'YouTube にログイン',
    allowSession: 'このセッションで許可',
    deny: '拒否',
    locationTitle: 'YouTube TV の位置情報',
    locationMessage: 'YouTube TV はサービスの利用可否を確認するために位置情報を求めています。',
    protectedTitle: '保護されたメディアへのアクセス',
    protectedMessage: 'YouTube は再生のために保護されたメディアへのアクセスを求めています。'
  },
  zh: {
    player: '播放器',
    dock: '停靠到 Hermes',
    togglePin: '切换固定偏好',
    fullscreen: '全屏',
    exitFullscreen: '退出全屏',
    close: '关闭播放器',
    signIn: '登录 YouTube',
    allowSession: '仅在此会话中允许',
    deny: '拒绝',
    locationTitle: 'YouTube TV 位置信息',
    locationMessage: 'YouTube TV 请求访问你的位置，以确认服务是否可用。',
    protectedTitle: '受保护的媒体访问',
    protectedMessage: 'YouTube 请求访问受保护的媒体以进行播放。'
  },
  'zh-hant': {
    player: '播放器',
    dock: '停靠至 Hermes',
    togglePin: '切換固定偏好設定',
    fullscreen: '全螢幕',
    exitFullscreen: '結束全螢幕',
    close: '關閉播放器',
    signIn: '登入 YouTube',
    allowSession: '僅在此工作階段允許',
    deny: '拒絕',
    locationTitle: 'YouTube TV 位置資訊',
    locationMessage: 'YouTube TV 要求存取你的位置，以確認服務是否可用。',
    protectedTitle: '受保護的媒體存取',
    protectedMessage: 'YouTube 要求存取受保護的媒體以進行播放。'
  },
  ar: {
    player: 'المشغل',
    dock: 'إرساء في Hermes',
    togglePin: 'تبديل تفضيل التثبيت',
    fullscreen: 'ملء الشاشة',
    exitFullscreen: 'إنهاء ملء الشاشة',
    close: 'إغلاق المشغل',
    signIn: 'تسجيل الدخول إلى YouTube',
    allowSession: 'السماح لهذه الجلسة',
    deny: 'رفض',
    locationTitle: 'موقع YouTube TV',
    locationMessage: 'يطلب YouTube TV موقعك للتحقق من توفر الخدمة.',
    protectedTitle: 'الوصول إلى الوسائط المحمية',
    protectedMessage: 'يطلب YouTube الوصول إلى الوسائط المحمية لتشغيلها.'
  },
  ru: {
    player: 'Плеер',
    dock: 'Закрепить в Hermes',
    togglePin: 'Переключить закрепление',
    fullscreen: 'На весь экран',
    exitFullscreen: 'Выйти из полноэкранного режима',
    close: 'Закрыть плеер',
    signIn: 'Войти в YouTube',
    allowSession: 'Разрешить на этот сеанс',
    deny: 'Запретить',
    locationTitle: 'Местоположение YouTube TV',
    locationMessage: 'YouTube TV запрашивает ваше местоположение, чтобы проверить доступность сервиса.',
    protectedTitle: 'Доступ к защищённым медиафайлам',
    protectedMessage: 'YouTube запрашивает доступ к защищённым медиафайлам для воспроизведения.'
  }
}

function normalizeLocale(locale: string): keyof typeof COPY {
  const normalized = locale.trim().toLowerCase().replaceAll('_', '-')

  if (normalized.startsWith('zh')) {
    return /(?:^|-)hant(?:-|$)|(?:^|-)tw(?:-|$)|(?:^|-)hk(?:-|$)|(?:^|-)mo(?:-|$)/.test(normalized) ? 'zh-hant' : 'zh'
  }

  if (normalized.startsWith('ja')) {
    return 'ja'
  }

  if (normalized.startsWith('ar')) {
    return 'ar'
  }

  if (normalized.startsWith('ru')) {
    return 'ru'
  }

  return 'en'
}

/** Translate small native player surfaces using Electron's system locale. */
export function getNativePlayerCopy(locale: string, source: PlayerSource = 'youtube'): NativePlayerCopy {
  const copy = COPY[normalizeLocale(locale)]

  if (source === 'youtube' || source === 'tv') {
    return copy
  }

  const label = getPlayerService(source).label

  return {
    ...copy,
    signIn: copy.signIn.replace('YouTube', label),
    locationTitle: copy.locationTitle.replace('YouTube TV', label),
    locationMessage: copy.locationMessage.replace('YouTube TV', label),
    protectedMessage: copy.protectedMessage.replace('YouTube', label)
  }
}
