import type { BundledLocale } from './types'

export interface FoundryTranslations {
  findModels: string
  deploymentHint: string
  modelsUnavailable: string
  deployment: string
  source: string
  automaticCredentials: string
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
    source: 'Get credentials from',
    automaticCredentials: 'Automatic Azure credentials',
    findModels: 'Refresh models',
    deploymentHint: 'Choose a listed model or enter your Azure deployment name.',
    modelsUnavailable: 'Could not list models. Enter your deployment name and test it directly.',
    azureCli: 'Azure CLI login',
    cliHint:
      'Uses the Azure CLI account on the machine running this profile. Run az login there first. Tokens refresh automatically; other credential sources are not tried.',
    defaultHint:
      'Uses Azure credentials configured for this profile or its hosting environment. Multiplexed bots need their own identity settings.',
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
    source: 'Anmeldedaten beziehen von',
    automaticCredentials: 'Automatische Azure-Anmeldedaten',
    findModels: 'Modelle aktualisieren',
    deploymentHint: 'Wähle ein aufgeführtes Modell oder gib deinen Azure-Bereitstellungsnamen ein.',
    modelsUnavailable:
      'Modelle konnten nicht aufgelistet werden. Gib den Bereitstellungsnamen ein und teste ihn direkt.',
    azureCli: 'Azure-CLI-Anmeldung',
    cliHint:
      'Verwendet das Azure-CLI-Konto auf dem Rechner dieses Profils. Führe dort zuerst az login aus. Tokens werden automatisch erneuert; andere Quellen werden nicht versucht.',
    defaultHint:
      'Verwendet die Azure-Anmeldedaten dieses Profils oder seiner Hostumgebung. Bots im Multiplexbetrieb benötigen eigene Identitätseinstellungen.',
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
    source: 'Obtenir les identifiants depuis',
    automaticCredentials: 'Identifiants Azure automatiques',
    findModels: 'Actualiser les modèles',
    deploymentHint: 'Choisissez un modèle proposé ou saisissez le nom de votre déploiement Azure.',
    modelsUnavailable: 'Impossible de lister les modèles. Saisissez le nom du déploiement et testez-le directement.',
    azureCli: 'Connexion Azure CLI',
    cliHint:
      'Utilise le compte Azure CLI de la machine exécutant ce profil. Lancez d’abord az login sur cette machine. Les jetons sont renouvelés automatiquement ; aucune autre source n’est essayée.',
    defaultHint:
      'Utilise les identifiants Azure de ce profil ou de son environnement hôte. Les bots multiplexés nécessitent leur propre identité.',
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
    source: 'Obtener credenciales de',
    automaticCredentials: 'Credenciales automáticas de Azure',
    findModels: 'Actualizar modelos',
    deploymentHint: 'Elige un modelo de la lista o introduce el nombre de tu implementación de Azure.',
    modelsUnavailable:
      'No se pudieron listar los modelos. Introduce el nombre de la implementación y pruébala directamente.',
    azureCli: 'Inicio de sesión de Azure CLI',
    cliHint:
      'Usa la cuenta de Azure CLI del equipo que ejecuta este perfil. Ejecuta az login allí primero. Los tokens se renuevan automáticamente; no se prueban otros orígenes.',
    defaultHint:
      'Usa las credenciales de Azure configuradas para este perfil o su entorno. Los bots multiplexados necesitan su propia identidad.',
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
    source: '凭据来源',
    automaticCredentials: '自动 Azure 凭据',
    findModels: '刷新模型',
    deploymentHint: '选择列出的模型，或输入 Azure 部署名称。',
    modelsUnavailable: '无法列出模型。请输入部署名称并直接测试。',
    azureCli: 'Azure CLI 登录',
    cliHint:
      '使用运行此配置文件的计算机上的 Azure CLI 账户。请先在该计算机运行 az login。令牌会自动刷新，不会尝试其他凭据来源。',
    defaultHint: '使用此配置文件或其托管环境中配置的 Azure 凭据。多路复用的机器人需要各自的身份设置。',
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
    source: '憑證來源',
    automaticCredentials: '自動 Azure 憑證',
    findModels: '重新整理模型',
    deploymentHint: '選擇列出的模型，或輸入 Azure 部署名稱。',
    modelsUnavailable: '無法列出模型。請輸入部署名稱並直接測試。',
    azureCli: 'Azure CLI 登入',
    cliHint:
      '使用執行此設定檔的電腦上的 Azure CLI 帳戶。請先在該電腦執行 az login。權杖會自動更新，不會嘗試其他憑證來源。',
    defaultHint: '使用此設定檔或其主控環境中設定的 Azure 憑證。多工機器人需要各自的身分設定。',
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
    source: '資格情報の取得元',
    automaticCredentials: 'Azure 資格情報の自動選択',
    findModels: 'モデルを更新',
    deploymentHint: '一覧のモデルを選ぶか、Azure のデプロイ名を入力してください。',
    modelsUnavailable: 'モデルを取得できませんでした。デプロイ名を入力して直接テストしてください。',
    azureCli: 'Azure CLI ログイン',
    cliHint:
      'このプロファイルを実行するマシンの Azure CLI アカウントを使用します。事前にそのマシンで az login を実行してください。トークンは自動更新され、他の資格情報ソースは試行されません。',
    defaultHint:
      'このプロファイルまたはホスト環境に設定された Azure 資格情報を使います。多重化されたボットには個別の ID 設定が必要です。',
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
    source: 'Источник учётных данных',
    automaticCredentials: 'Автоматические учётные данные Azure',
    findModels: 'Обновить модели',
    deploymentHint: 'Выберите модель из списка или введите имя развёртывания Azure.',
    modelsUnavailable: 'Не удалось получить список моделей. Введите имя развёртывания и проверьте его напрямую.',
    azureCli: 'Вход через Azure CLI',
    cliHint:
      'Использует учётную запись Azure CLI на компьютере, где работает профиль. Сначала выполните там az login. Токены обновляются автоматически; другие источники не проверяются.',
    defaultHint:
      'Использует учётные данные Azure этого профиля или его среды. Мультиплексированным ботам нужны собственные настройки удостоверения.',
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
    source: 'الحصول على بيانات الاعتماد من',
    automaticCredentials: 'بيانات اعتماد Azure التلقائية',
    findModels: 'تحديث النماذج',
    deploymentHint: 'اختر نموذجاً من القائمة أو أدخل اسم نشر Azure.',
    modelsUnavailable: 'تعذر عرض النماذج. أدخل اسم النشر واختبره مباشرة.',
    azureCli: 'تسجيل الدخول عبر Azure CLI',
    cliHint:
      'يستخدم حساب Azure CLI على الجهاز الذي يشغّل هذا الملف التعريفي. شغّل az login هناك أولاً. تتجدد الرموز تلقائياً ولا تُجرّب مصادر اعتماد أخرى.',
    defaultHint:
      'يستخدم بيانات اعتماد Azure المعدّة لهذا الملف التعريفي أو بيئة استضافته. تحتاج الروبوتات متعددة الملفات إلى إعدادات هوية خاصة بها.',
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
