import type { BundledLocale, Locale } from '@/i18n/types'

interface WebappInstanceCopy {
  checking: string
  redirecting: string
  error: string
  missing: string
  inactive: string
  unauthorized: string
  organizationRequired: string
  signInAgain: string
  openUnitalk: string
  signOut: string
}

const copy: Record<BundledLocale, WebappInstanceCopy> = {
  en: {
    checking: 'Finding your instance…',
    redirecting: 'Opening your instance…',
    error: 'Could not open your instance',
    missing: 'Create an instance for this organization to continue.',
    inactive: 'Activate your instance in Unitalk to continue.',
    unauthorized: 'Your Unitalk session could not be authenticated. Sign in again to continue.',
    organizationRequired: 'Select an organization to open your instance.',
    signInAgain: 'Sign in again',
    openUnitalk: 'Open Unitalk',
    signOut: 'Sign out'
  },
  ar: {
    checking: 'جارٍ البحث عن مثيلك…',
    redirecting: 'جارٍ فتح مثيلك…',
    error: 'تعذر فتح مثيلك',
    missing: 'أنشئ مثيلًا لهذه المؤسسة للمتابعة.',
    inactive: 'فعّل مثيلك في Unitalk للمتابعة.',
    unauthorized: 'تعذر التحقق من جلسة Unitalk. سجّل الدخول مجددًا للمتابعة.',
    organizationRequired: 'اختر مؤسسة لفتح مثيلك.',
    signInAgain: 'تسجيل الدخول مجددًا',
    openUnitalk: 'فتح Unitalk',
    signOut: 'تسجيل الخروج'
  },
  de: {
    checking: 'Deine Instanz wird gesucht…',
    redirecting: 'Deine Instanz wird geöffnet…',
    error: 'Deine Instanz konnte nicht geöffnet werden',
    missing: 'Erstelle eine Instanz für diese Organisation, um fortzufahren.',
    inactive: 'Aktiviere deine Instanz in Unitalk, um fortzufahren.',
    unauthorized: 'Deine Unitalk-Sitzung konnte nicht authentifiziert werden. Melde dich erneut an.',
    organizationRequired: 'Wähle eine Organisation, um deine Instanz zu öffnen.',
    signInAgain: 'Erneut anmelden',
    openUnitalk: 'Unitalk öffnen',
    signOut: 'Abmelden'
  },
  es: {
    checking: 'Buscando tu instancia…',
    redirecting: 'Abriendo tu instancia…',
    error: 'No se pudo abrir tu instancia',
    missing: 'Crea una instancia para esta organización para continuar.',
    inactive: 'Activa tu instancia en Unitalk para continuar.',
    unauthorized: 'No se pudo autenticar tu sesión de Unitalk. Inicia sesión de nuevo para continuar.',
    organizationRequired: 'Selecciona una organización para abrir tu instancia.',
    signInAgain: 'Iniciar sesión de nuevo',
    openUnitalk: 'Abrir Unitalk',
    signOut: 'Cerrar sesión'
  },
  fr: {
    checking: 'Recherche de votre instance…',
    redirecting: 'Ouverture de votre instance…',
    error: 'Impossible d’ouvrir votre instance',
    missing: 'Créez une instance pour cette organisation pour continuer.',
    inactive: 'Activez votre instance dans Unitalk pour continuer.',
    unauthorized: 'Votre session Unitalk n’a pas pu être authentifiée. Reconnectez-vous pour continuer.',
    organizationRequired: 'Sélectionnez une organisation pour ouvrir votre instance.',
    signInAgain: 'Se reconnecter',
    openUnitalk: 'Ouvrir Unitalk',
    signOut: 'Se déconnecter'
  },
  ja: {
    checking: 'インスタンスを検索中…',
    redirecting: 'インスタンスを開いています…',
    error: 'インスタンスを開けませんでした',
    missing: '続行するには、この組織のインスタンスを作成してください。',
    inactive: '続行するには、Unitalk でインスタンスを有効にしてください。',
    unauthorized: 'Unitalk セッションを認証できませんでした。続行するには再度サインインしてください。',
    organizationRequired: 'インスタンスを開く組織を選択してください。',
    signInAgain: '再度サインイン',
    openUnitalk: 'Unitalk を開く',
    signOut: 'サインアウト'
  },
  ru: {
    checking: 'Поиск вашего экземпляра…',
    redirecting: 'Открытие вашего экземпляра…',
    error: 'Не удалось открыть ваш экземпляр',
    missing: 'Создайте экземпляр для этой организации, чтобы продолжить.',
    inactive: 'Активируйте свой экземпляр в Unitalk, чтобы продолжить.',
    unauthorized: 'Не удалось подтвердить сеанс Unitalk. Войдите снова, чтобы продолжить.',
    organizationRequired: 'Выберите организацию, чтобы открыть свой экземпляр.',
    signInAgain: 'Войти снова',
    openUnitalk: 'Открыть Unitalk',
    signOut: 'Выйти'
  },
  zh: {
    checking: '正在查找你的实例…',
    redirecting: '正在打开你的实例…',
    error: '无法打开你的实例',
    missing: '请为此组织创建实例以继续。',
    inactive: '请在 Unitalk 中激活你的实例以继续。',
    unauthorized: '无法验证你的 Unitalk 会话。请重新登录以继续。',
    organizationRequired: '请选择组织以打开你的实例。',
    signInAgain: '重新登录',
    openUnitalk: '打开 Unitalk',
    signOut: '退出登录'
  },
  'zh-hant': {
    checking: '正在尋找你的執行個體…',
    redirecting: '正在開啟你的執行個體…',
    error: '無法開啟你的執行個體',
    missing: '請為此組織建立執行個體以繼續。',
    inactive: '請在 Unitalk 中啟用你的執行個體以繼續。',
    unauthorized: '無法驗證你的 Unitalk 工作階段。請重新登入以繼續。',
    organizationRequired: '請選擇組織以開啟你的執行個體。',
    signInAgain: '重新登入',
    openUnitalk: '開啟 Unitalk',
    signOut: '登出'
  }
}

export function webappInstanceCopy(locale: Locale): WebappInstanceCopy {
  return copy[locale as BundledLocale] ?? copy.en
}
