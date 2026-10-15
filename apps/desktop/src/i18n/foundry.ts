import type { BundledLocale } from './types'

export interface FoundryTranslations {
  findDeployments: string
  deploymentHint: string
  deploymentsUnavailable: string
  deployment: string
  source: string
  automaticCredentials: string
  azureCli: string
  cliHint: string
  defaultHint: string
  advanced: string
  tenant: string
  subscription: string
  accountHint: string
  searchDeployments: string
  noDeployments: string
  scope: string
  saveAndUse: string
  saved: string
  loadFailed: string
  profileHint: string
}

export const foundryTranslations: Record<BundledLocale, FoundryTranslations> = {
  en: {
    deployment: 'Deployment name',
    source: 'Get credentials from',
    automaticCredentials: 'Automatic Azure credentials',
    findDeployments: 'Refresh deployments',
    deploymentHint: 'Choose a deployment or enter its exact name.',
    deploymentsUnavailable:
      'No deployments found or listing unavailable. Enter a deployment name and test it directly.',
    azureCli: 'Azure CLI login',
    cliHint:
      'Uses the Azure CLI account on the machine running this profile. Run az login there first. Tokens refresh automatically; other credential sources are not tried.',
    defaultHint:
      'Uses Azure credentials configured for this profile or its hosting environment. Multiplexed bots need their own identity settings.',
    advanced: 'Advanced settings',
    tenant: 'Tenant ID',
    subscription: 'Subscription name or ID',
    accountHint: 'Use a tenant or a subscription, not both. Leave both blank to use the current Azure CLI account.',
    searchDeployments: 'Search deployments…',
    noDeployments: 'No matching deployments.',
    scope: 'Token scope',
    saveAndUse: 'Save and use',
    saved: 'Foundry saved for this profile.',
    loadFailed: 'Could not load Foundry settings.',
    profileHint: 'Sets the provider and deployment for new chats in the selected profile.'
  },
  de: {
    deployment: 'Bereitstellungsname',
    source: 'Anmeldedaten beziehen von',
    automaticCredentials: 'Automatische Azure-Anmeldedaten',
    findDeployments: 'Bereitstellungen aktualisieren',
    deploymentHint: 'Wähle eine Bereitstellung oder gib ihren genauen Namen ein.',
    deploymentsUnavailable:
      'Keine Bereitstellungen gefunden oder Auflistung nicht verfügbar. Gib einen Bereitstellungsnamen ein und teste ihn direkt.',
    azureCli: 'Azure-CLI-Anmeldung',
    cliHint:
      'Verwendet das Azure-CLI-Konto auf dem Rechner dieses Profils. Führe dort zuerst az login aus. Tokens werden automatisch erneuert; andere Quellen werden nicht versucht.',
    defaultHint:
      'Verwendet die Azure-Anmeldedaten dieses Profils oder seiner Hostumgebung. Bots im Multiplexbetrieb benötigen eigene Identitätseinstellungen.',
    advanced: 'Erweiterte Einstellungen',
    tenant: 'Mandanten-ID',
    subscription: 'Abonnementname oder -ID',
    accountHint:
      'Verwende entweder einen Mandanten oder ein Abonnement. Lasse beide Felder leer, um das aktuelle Azure-CLI-Konto zu verwenden.',
    searchDeployments: 'Bereitstellungen suchen…',
    noDeployments: 'Keine passenden Bereitstellungen.',
    scope: 'Token-Gültigkeitsbereich',
    saveAndUse: 'Speichern und verwenden',
    saved: 'Foundry für dieses Profil gespeichert.',
    loadFailed: 'Foundry-Einstellungen konnten nicht geladen werden.',
    profileHint: 'Legt Anbieter und Bereitstellung für neue Chats im ausgewählten Profil fest.'
  },
  fr: {
    deployment: 'Nom du déploiement',
    source: 'Obtenir les identifiants depuis',
    automaticCredentials: 'Identifiants Azure automatiques',
    findDeployments: 'Actualiser les déploiements',
    deploymentHint: 'Choisissez un déploiement ou saisissez son nom exact.',
    deploymentsUnavailable:
      'Aucun déploiement trouvé ou liste indisponible. Saisissez un nom de déploiement et testez-le directement.',
    azureCli: 'Connexion Azure CLI',
    cliHint:
      'Utilise le compte Azure CLI de la machine exécutant ce profil. Lancez d’abord az login sur cette machine. Les jetons sont renouvelés automatiquement ; aucune autre source n’est essayée.',
    defaultHint:
      'Utilise les identifiants Azure de ce profil ou de son environnement hôte. Les bots multiplexés nécessitent leur propre identité.',
    advanced: 'Paramètres avancés',
    tenant: 'ID du locataire',
    subscription: 'Nom ou ID de l’abonnement',
    accountHint:
      'Utilisez un locataire ou un abonnement, pas les deux. Laissez les deux champs vides pour utiliser le compte Azure CLI actuel.',
    searchDeployments: 'Rechercher des déploiements…',
    noDeployments: 'Aucun déploiement correspondant.',
    scope: 'Étendue du jeton',
    saveAndUse: 'Enregistrer et utiliser',
    saved: 'Foundry enregistré pour ce profil.',
    loadFailed: 'Impossible de charger les paramètres Foundry.',
    profileHint: 'Définit le fournisseur et le déploiement des nouvelles conversations du profil sélectionné.'
  },
  es: {
    deployment: 'Nombre de la implementación',
    source: 'Obtener credenciales de',
    automaticCredentials: 'Credenciales automáticas de Azure',
    findDeployments: 'Actualizar implementaciones',
    deploymentHint: 'Elige una implementación o introduce su nombre exacto.',
    deploymentsUnavailable:
      'No se encontraron implementaciones o la lista no está disponible. Introduce un nombre y pruébalo directamente.',
    azureCli: 'Inicio de sesión de Azure CLI',
    cliHint:
      'Usa la cuenta de Azure CLI del equipo que ejecuta este perfil. Ejecuta az login allí primero. Los tokens se renuevan automáticamente; no se prueban otros orígenes.',
    defaultHint:
      'Usa las credenciales de Azure configuradas para este perfil o su entorno. Los bots multiplexados necesitan su propia identidad.',
    advanced: 'Configuración avanzada',
    tenant: 'ID del inquilino',
    subscription: 'Nombre o ID de la suscripción',
    accountHint:
      'Usa un inquilino o una suscripción, no ambos. Deja ambos campos vacíos para usar la cuenta actual de Azure CLI.',
    searchDeployments: 'Buscar implementaciones…',
    noDeployments: 'No hay implementaciones coincidentes.',
    scope: 'Ámbito del token',
    saveAndUse: 'Guardar y usar',
    saved: 'Foundry guardado para este perfil.',
    loadFailed: 'No se pudo cargar la configuración de Foundry.',
    profileHint: 'Establece el proveedor y la implementación para los nuevos chats del perfil seleccionado.'
  },
  zh: {
    deployment: '部署名称',
    source: '凭据来源',
    automaticCredentials: '自动 Azure 凭据',
    findDeployments: '刷新部署',
    deploymentHint: '选择部署或输入其确切名称。',
    deploymentsUnavailable: '未找到部署或无法列出部署。请输入部署名称并直接测试。',
    azureCli: 'Azure CLI 登录',
    cliHint:
      '使用运行此配置文件的计算机上的 Azure CLI 账户。请先在该计算机运行 az login。令牌会自动刷新，不会尝试其他凭据来源。',
    defaultHint: '使用此配置文件或其托管环境中配置的 Azure 凭据。多路复用的机器人需要各自的身份设置。',
    advanced: '高级设置',
    tenant: '租户 ID',
    subscription: '订阅名称或 ID',
    accountHint: '租户和订阅只能填写一项。两项均留空时使用当前 Azure CLI 账户。',
    searchDeployments: '搜索部署…',
    noDeployments: '没有匹配的部署。',
    scope: '令牌范围',
    saveAndUse: '保存并使用',
    saved: '已为此配置文件保存 Foundry。',
    loadFailed: '无法加载 Foundry 设置。',
    profileHint: '为所选配置文件的新聊天设置提供商和部署。'
  },
  'zh-hant': {
    deployment: '部署名稱',
    source: '憑證來源',
    automaticCredentials: '自動 Azure 憑證',
    findDeployments: '重新整理部署',
    deploymentHint: '選擇部署或輸入其確切名稱。',
    deploymentsUnavailable: '找不到部署或無法列出部署。請輸入部署名稱並直接測試。',
    azureCli: 'Azure CLI 登入',
    cliHint:
      '使用執行此設定檔的電腦上的 Azure CLI 帳戶。請先在該電腦執行 az login。權杖會自動更新，不會嘗試其他憑證來源。',
    defaultHint: '使用此設定檔或其主控環境中設定的 Azure 憑證。多工機器人需要各自的身分設定。',
    advanced: '進階設定',
    tenant: '租用戶 ID',
    subscription: '訂閱名稱或 ID',
    accountHint: '租用戶與訂閱只能填寫一項。兩項皆留空時使用目前的 Azure CLI 帳戶。',
    searchDeployments: '搜尋部署…',
    noDeployments: '沒有相符的部署。',
    scope: '權杖範圍',
    saveAndUse: '儲存並使用',
    saved: '已為此設定檔儲存 Foundry。',
    loadFailed: '無法載入 Foundry 設定。',
    profileHint: '為所選設定檔的新聊天設定提供者和部署。'
  },
  ja: {
    deployment: 'デプロイ名',
    source: '資格情報の取得元',
    automaticCredentials: 'Azure 資格情報の自動選択',
    findDeployments: 'デプロイを更新',
    deploymentHint: 'デプロイを選ぶか正確な名前を入力してください。',
    deploymentsUnavailable: 'デプロイが見つからないか一覧を取得できません。名前を入力して直接テストしてください。',
    azureCli: 'Azure CLI ログイン',
    cliHint:
      'このプロファイルを実行するマシンの Azure CLI アカウントを使用します。事前にそのマシンで az login を実行してください。トークンは自動更新され、他の資格情報ソースは試行されません。',
    defaultHint:
      'このプロファイルまたはホスト環境に設定された Azure 資格情報を使います。多重化されたボットには個別の ID 設定が必要です。',
    advanced: '詳細設定',
    tenant: 'テナント ID',
    subscription: 'サブスクリプション名または ID',
    accountHint:
      'テナントまたはサブスクリプションの一方を指定してください。両方を空欄にすると現在の Azure CLI アカウントを使います。',
    searchDeployments: 'デプロイを検索…',
    noDeployments: '一致するデプロイはありません。',
    scope: 'トークンスコープ',
    saveAndUse: '保存して使用',
    saved: 'このプロファイルの Foundry を保存しました。',
    loadFailed: 'Foundry の設定を読み込めませんでした。',
    profileHint: '選択したプロファイルの新しいチャットで使用するプロバイダーとデプロイを設定します。'
  },
  ru: {
    deployment: 'Имя развёртывания',
    source: 'Источник учётных данных',
    automaticCredentials: 'Автоматические учётные данные Azure',
    findDeployments: 'Обновить развёртывания',
    deploymentHint: 'Выберите развёртывание или введите его точное имя.',
    deploymentsUnavailable: 'Развёртывания не найдены или список недоступен. Введите имя и проверьте его напрямую.',
    azureCli: 'Вход через Azure CLI',
    cliHint:
      'Использует учётную запись Azure CLI на компьютере, где работает профиль. Сначала выполните там az login. Токены обновляются автоматически; другие источники не проверяются.',
    defaultHint:
      'Использует учётные данные Azure этого профиля или его среды. Мультиплексированным ботам нужны собственные настройки удостоверения.',
    advanced: 'Дополнительные настройки',
    tenant: 'ID арендатора',
    subscription: 'Имя или ID подписки',
    accountHint:
      'Укажите арендатора или подписку, но не оба. Оставьте оба поля пустыми для текущей учётной записи Azure CLI.',
    searchDeployments: 'Поиск развёртываний…',
    noDeployments: 'Подходящих развёртываний нет.',
    scope: 'Область токена',
    saveAndUse: 'Сохранить и использовать',
    saved: 'Foundry сохранён для этого профиля.',
    loadFailed: 'Не удалось загрузить настройки Foundry.',
    profileHint: 'Задаёт поставщика и развёртывание для новых чатов выбранного профиля.'
  },
  ar: {
    deployment: 'اسم النشر',
    source: 'الحصول على بيانات الاعتماد من',
    automaticCredentials: 'بيانات اعتماد Azure التلقائية',
    findDeployments: 'تحديث عمليات النشر',
    deploymentHint: 'اختر عملية نشر أو أدخل اسمها الدقيق.',
    deploymentsUnavailable: 'لم يتم العثور على عمليات نشر أو أن القائمة غير متاحة. أدخل اسم النشر واختبره مباشرة.',
    azureCli: 'تسجيل الدخول عبر Azure CLI',
    cliHint:
      'يستخدم حساب Azure CLI على الجهاز الذي يشغّل هذا الملف التعريفي. شغّل az login هناك أولاً. تتجدد الرموز تلقائياً ولا تُجرّب مصادر اعتماد أخرى.',
    defaultHint:
      'يستخدم بيانات اعتماد Azure المعدّة لهذا الملف التعريفي أو بيئة استضافته. تحتاج الروبوتات متعددة الملفات إلى إعدادات هوية خاصة بها.',
    advanced: 'إعدادات متقدمة',
    tenant: 'معرّف المستأجر',
    subscription: 'اسم الاشتراك أو معرّفه',
    accountHint: 'استخدم مستأجراً أو اشتراكاً، وليس كليهما. اترك الحقلين فارغين لاستخدام حساب Azure CLI الحالي.',
    searchDeployments: 'البحث عن عمليات النشر…',
    noDeployments: 'لا توجد عمليات نشر مطابقة.',
    scope: 'نطاق الرمز',
    saveAndUse: 'حفظ واستخدام',
    saved: 'تم حفظ Foundry لهذا الملف التعريفي.',
    loadFailed: 'تعذر تحميل إعدادات Foundry.',
    profileHint: 'يحدد الموفر والنشر للمحادثات الجديدة في الملف التعريفي المحدد.'
  }
}
