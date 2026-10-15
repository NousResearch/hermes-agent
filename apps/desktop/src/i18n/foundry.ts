import type { BundledLocale } from './types'

export interface FoundryTranslations {
  deployment: string
  source: string
  defaultChain: string
  azureCli: string
  cliHint: string
  defaultHint: string
  advanced: string
  tenant: string
  subscription: string
  cliAccount: string
  cliDefault: string
  scope: string
  saveAndUse: string
  saved: string
  loadFailed: string
  profileHint: string
}

export const foundryTranslations: Record<BundledLocale, FoundryTranslations> = {
  en: {
    deployment: 'Deployment name',
    source: 'Entra credential source',
    defaultChain: 'Default credential chain',
    azureCli: 'Azure CLI login',
    cliHint:
      'Uses the Azure CLI account on the machine running this profile. Run az login there first. Tokens refresh automatically; other credential sources are not tried.',
    defaultHint:
      'Uses the existing Azure credential chain. Desktop and gateway profiles need their own Azure identity settings; ambient credentials remain blocked under multiplexing.',
    advanced: 'Advanced settings',
    tenant: 'Tenant ID',
    subscription: 'Subscription name or ID',
    cliAccount: 'Azure CLI account',
    cliDefault: 'Azure CLI default',
    scope: 'Token scope',
    saveAndUse: 'Save and use',
    saved: 'Foundry saved for this profile.',
    loadFailed: 'Could not load Foundry settings.',
    profileHint: 'Sets the provider and deployment for new chats in the selected profile.'
  },
  de: {
    deployment: 'Bereitstellungsname',
    source: 'Entra-Anmeldequelle',
    defaultChain: 'Standard-Anmeldekette',
    azureCli: 'Azure-CLI-Anmeldung',
    cliHint:
      'Verwendet das Azure-CLI-Konto auf dem Rechner dieses Profils. Führe dort zuerst az login aus. Tokens werden automatisch erneuert; andere Quellen werden nicht versucht.',
    defaultHint:
      'Verwendet die bestehende Azure-Anmeldekette. Desktop- und Gateway-Profile benötigen eigene Azure-Identitätseinstellungen; bei Multiplexing bleiben Umgebungsanmeldedaten gesperrt.',
    advanced: 'Erweiterte Einstellungen',
    tenant: 'Mandanten-ID',
    subscription: 'Abonnementname oder -ID',
    cliAccount: 'Azure-CLI-Konto',
    cliDefault: 'Azure-CLI-Standard',
    scope: 'Token-Gültigkeitsbereich',
    saveAndUse: 'Speichern und verwenden',
    saved: 'Foundry für dieses Profil gespeichert.',
    loadFailed: 'Foundry-Einstellungen konnten nicht geladen werden.',
    profileHint: 'Legt Anbieter und Bereitstellung für neue Chats im ausgewählten Profil fest.'
  },
  fr: {
    deployment: 'Nom du déploiement',
    source: 'Source des identifiants Entra',
    defaultChain: 'Chaîne d’identifiants par défaut',
    azureCli: 'Connexion Azure CLI',
    cliHint:
      'Utilise le compte Azure CLI de la machine exécutant ce profil. Lancez d’abord az login sur cette machine. Les jetons sont renouvelés automatiquement ; aucune autre source n’est essayée.',
    defaultHint:
      'Utilise la chaîne d’identifiants Azure existante. Les profils Desktop et passerelle nécessitent leurs propres paramètres d’identité Azure ; les identifiants ambiants restent bloqués en multiplexage.',
    advanced: 'Paramètres avancés',
    tenant: 'ID du locataire',
    subscription: 'Nom ou ID de l’abonnement',
    cliAccount: 'Compte Azure CLI',
    cliDefault: 'Valeur par défaut d’Azure CLI',
    scope: 'Étendue du jeton',
    saveAndUse: 'Enregistrer et utiliser',
    saved: 'Foundry enregistré pour ce profil.',
    loadFailed: 'Impossible de charger les paramètres Foundry.',
    profileHint: 'Définit le fournisseur et le déploiement des nouvelles conversations du profil sélectionné.'
  },
  es: {
    deployment: 'Nombre de la implementación',
    source: 'Origen de credenciales de Entra',
    defaultChain: 'Cadena de credenciales predeterminada',
    azureCli: 'Inicio de sesión de Azure CLI',
    cliHint:
      'Usa la cuenta de Azure CLI del equipo que ejecuta este perfil. Ejecuta az login allí primero. Los tokens se renuevan automáticamente; no se prueban otros orígenes.',
    defaultHint:
      'Usa la cadena de credenciales de Azure existente. Los perfiles de escritorio y puerta de enlace necesitan su propia configuración de identidad de Azure; las credenciales ambientales siguen bloqueadas en multiplexación.',
    advanced: 'Configuración avanzada',
    tenant: 'ID del inquilino',
    subscription: 'Nombre o ID de la suscripción',
    cliAccount: 'Cuenta de Azure CLI',
    cliDefault: 'Predeterminado de Azure CLI',
    scope: 'Ámbito del token',
    saveAndUse: 'Guardar y usar',
    saved: 'Foundry guardado para este perfil.',
    loadFailed: 'No se pudo cargar la configuración de Foundry.',
    profileHint: 'Establece el proveedor y la implementación para los nuevos chats del perfil seleccionado.'
  },
  zh: {
    deployment: '部署名称',
    source: 'Entra 凭据来源',
    defaultChain: '默认凭据链',
    azureCli: 'Azure CLI 登录',
    cliHint:
      '使用运行此配置文件的计算机上的 Azure CLI 账户。请先在该计算机运行 az login。令牌会自动刷新，不会尝试其他凭据来源。',
    defaultHint: '使用现有的 Azure 凭据链。桌面和网关配置文件需要各自的 Azure 身份设置；多路复用时仍禁止使用环境凭据。',
    advanced: '高级设置',
    tenant: '租户 ID',
    subscription: '订阅名称或 ID',
    cliAccount: 'Azure CLI 账户',
    cliDefault: 'Azure CLI 默认值',
    scope: '令牌范围',
    saveAndUse: '保存并使用',
    saved: '已为此配置文件保存 Foundry。',
    loadFailed: '无法加载 Foundry 设置。',
    profileHint: '为所选配置文件的新聊天设置提供商和部署。'
  },
  'zh-hant': {
    deployment: '部署名稱',
    source: 'Entra 憑證來源',
    defaultChain: '預設憑證鏈',
    azureCli: 'Azure CLI 登入',
    cliHint:
      '使用執行此設定檔的電腦上的 Azure CLI 帳戶。請先在該電腦執行 az login。權杖會自動更新，不會嘗試其他憑證來源。',
    defaultHint: '使用現有的 Azure 憑證鏈。桌面和閘道設定檔需要各自的 Azure 身分設定；多工時仍禁止使用環境憑證。',
    advanced: '進階設定',
    tenant: '租用戶 ID',
    subscription: '訂閱名稱或 ID',
    cliAccount: 'Azure CLI 帳戶',
    cliDefault: 'Azure CLI 預設值',
    scope: '權杖範圍',
    saveAndUse: '儲存並使用',
    saved: '已為此設定檔儲存 Foundry。',
    loadFailed: '無法載入 Foundry 設定。',
    profileHint: '為所選設定檔的新聊天設定提供者和部署。'
  },
  ja: {
    deployment: 'デプロイ名',
    source: 'Entra 資格情報ソース',
    defaultChain: '既定の資格情報チェーン',
    azureCli: 'Azure CLI ログイン',
    cliHint:
      'このプロファイルを実行するマシンの Azure CLI アカウントを使用します。事前にそのマシンで az login を実行してください。トークンは自動更新され、他の資格情報ソースは試行されません。',
    defaultHint:
      '既存の Azure 資格情報チェーンを使用します。デスクトップとゲートウェイのプロファイルには個別の Azure ID 設定が必要です。多重化時は環境の資格情報を引き続き拒否します。',
    advanced: '詳細設定',
    tenant: 'テナント ID',
    subscription: 'サブスクリプション名または ID',
    cliAccount: 'Azure CLI アカウント',
    cliDefault: 'Azure CLI の既定値',
    scope: 'トークンスコープ',
    saveAndUse: '保存して使用',
    saved: 'このプロファイルの Foundry を保存しました。',
    loadFailed: 'Foundry の設定を読み込めませんでした。',
    profileHint: '選択したプロファイルの新しいチャットで使用するプロバイダーとデプロイを設定します。'
  },
  ru: {
    deployment: 'Имя развёртывания',
    source: 'Источник учётных данных Entra',
    defaultChain: 'Стандартная цепочка учётных данных',
    azureCli: 'Вход через Azure CLI',
    cliHint:
      'Использует учётную запись Azure CLI на компьютере, где работает профиль. Сначала выполните там az login. Токены обновляются автоматически; другие источники не проверяются.',
    defaultHint:
      'Использует существующую цепочку учётных данных Azure. Профилям приложения и шлюза нужны собственные настройки удостоверения Azure; при мультиплексировании учётные данные окружения остаются заблокированными.',
    advanced: 'Дополнительные настройки',
    tenant: 'ID арендатора',
    subscription: 'Имя или ID подписки',
    cliAccount: 'Учётная запись Azure CLI',
    cliDefault: 'По умолчанию в Azure CLI',
    scope: 'Область токена',
    saveAndUse: 'Сохранить и использовать',
    saved: 'Foundry сохранён для этого профиля.',
    loadFailed: 'Не удалось загрузить настройки Foundry.',
    profileHint: 'Задаёт поставщика и развёртывание для новых чатов выбранного профиля.'
  },
  ar: {
    deployment: 'اسم النشر',
    source: 'مصدر بيانات اعتماد Entra',
    defaultChain: 'سلسلة بيانات الاعتماد الافتراضية',
    azureCli: 'تسجيل الدخول عبر Azure CLI',
    cliHint:
      'يستخدم حساب Azure CLI على الجهاز الذي يشغّل هذا الملف التعريفي. شغّل az login هناك أولاً. تتجدد الرموز تلقائياً ولا تُجرّب مصادر اعتماد أخرى.',
    defaultHint:
      'يستخدم سلسلة بيانات اعتماد Azure الحالية. تحتاج ملفات سطح المكتب والبوابة إلى إعدادات هوية Azure خاصة بها؛ وتبقى بيانات الاعتماد المحيطة محظورة عند تعدد الملفات.',
    advanced: 'إعدادات متقدمة',
    tenant: 'معرّف المستأجر',
    subscription: 'اسم الاشتراك أو معرّفه',
    cliAccount: 'حساب Azure CLI',
    cliDefault: 'إعداد Azure CLI الافتراضي',
    scope: 'نطاق الرمز',
    saveAndUse: 'حفظ واستخدام',
    saved: 'تم حفظ Foundry لهذا الملف التعريفي.',
    loadFailed: 'تعذر تحميل إعدادات Foundry.',
    profileHint: 'يحدد الموفر والنشر للمحادثات الجديدة في الملف التعريفي المحدد.'
  }
}
