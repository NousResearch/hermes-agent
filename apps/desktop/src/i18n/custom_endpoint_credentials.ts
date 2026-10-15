import type { BundledLocale } from './types'
import type { CustomEndpointsTranslations } from './types_custom_endpoints'

export const customEndpointCredentials: Record<
  Exclude<BundledLocale, 'en'>,
  CustomEndpointsTranslations['credentials']
> = {
  de: {
    authentication: 'Authentifizierung',
    azureCli: 'Azure-CLI-Anmeldung',
    command: 'Anmeldebefehl',
    commandLabel: 'Token-Befehl',
    azureCliHint:
      'Verwendet das Azure-CLI-Konto auf dem Rechner dieses Profils. Installiere dort Azure CLI und führe az login aus. Tokens werden automatisch erneuert. Für Foundry ist die Responses API ausgewählt.',
    commandHint:
      'Läuft auf dem Rechner dieses Profils. Gibt nur ein Token oder JSON mit access_token und Ablaufzeit aus. Der Befehl wird im Profil gespeichert, Tokens nicht.',
    commandPlaceholder: 'Befehl zur Ausgabe eines Zugriffstokens',
    savedCommandPlaceholder: 'Leer lassen, um den gespeicherten Befehl zu behalten'
  },
  fr: {
    authentication: 'Authentification',
    azureCli: 'Connexion Azure CLI',
    command: 'Commande d’authentification',
    commandLabel: 'Commande de jeton',
    azureCliHint:
      'Utilise le compte Azure CLI de la machine exécutant ce profil. Installez Azure CLI et lancez az login sur cette machine. Les jetons sont renouvelés automatiquement. Responses API est sélectionné pour Foundry.',
    commandHint:
      'S’exécute sur la machine hébergeant ce profil. Affiche uniquement un jeton ou du JSON avec access_token et une expiration. Le profil conserve la commande, jamais les jetons.',
    commandPlaceholder: 'Commande affichant un jeton d’accès',
    savedCommandPlaceholder: 'Laisser vide pour conserver la commande enregistrée'
  },
  es: {
    authentication: 'Autenticación',
    azureCli: 'Inicio de sesión de Azure CLI',
    command: 'Comando de credenciales',
    commandLabel: 'Comando del token',
    azureCliHint:
      'Usa la cuenta de Azure CLI del equipo que ejecuta este perfil. Instala Azure CLI y ejecuta az login allí. Los tokens se renuevan automáticamente. Se selecciona Responses API para Foundry.',
    commandHint:
      'Se ejecuta en el equipo que aloja este perfil. Imprime solo un token o JSON con access_token y caducidad. El perfil guarda el comando, nunca los tokens.',
    commandPlaceholder: 'Comando que imprime un token de acceso',
    savedCommandPlaceholder: 'Déjalo vacío para conservar el comando guardado'
  },
  zh: {
    authentication: '身份验证',
    azureCli: 'Azure CLI 登录',
    command: '凭据命令',
    commandLabel: '令牌命令',
    azureCliHint:
      '使用运行此配置文件的计算机上的 Azure CLI 账户。请先在该计算机安装 Azure CLI 并运行 az login。令牌会自动刷新。Foundry 默认使用 Responses API。',
    commandHint:
      '在托管此配置文件的计算机上运行。仅输出令牌，或包含 access_token 和过期时间的 JSON。配置文件保存命令，但不保存令牌。',
    commandPlaceholder: '输出访问令牌的命令',
    savedCommandPlaceholder: '留空以保留已保存的命令'
  },
  'zh-hant': {
    authentication: '身分驗證',
    azureCli: 'Azure CLI 登入',
    command: '憑證命令',
    commandLabel: '權杖命令',
    azureCliHint:
      '使用執行此設定檔的電腦上的 Azure CLI 帳戶。請先在該電腦安裝 Azure CLI 並執行 az login。權杖會自動更新。Foundry 預設使用 Responses API。',
    commandHint:
      '在託管此設定檔的電腦上執行。僅輸出權杖，或包含 access_token 和到期時間的 JSON。設定檔儲存命令，但不儲存權杖。',
    commandPlaceholder: '輸出存取權杖的命令',
    savedCommandPlaceholder: '留空以保留已儲存的命令'
  },
  ja: {
    authentication: '認証',
    azureCli: 'Azure CLI ログイン',
    command: '認証コマンド',
    commandLabel: 'トークンコマンド',
    azureCliHint:
      'このプロファイルを実行するマシンの Azure CLI アカウントを使います。そのマシンに Azure CLI をインストールし、az login を実行してください。トークンは自動更新されます。Foundry 用に Responses API が選択されます。',
    commandHint:
      'このプロファイルのホストで実行します。トークンのみ、または access_token と有効期限を含む JSON を出力します。コマンドはプロファイルに保存されますが、トークンは保存されません。',
    commandPlaceholder: 'アクセストークンを出力するコマンド',
    savedCommandPlaceholder: '空欄で保存済みのコマンドを維持'
  },
  ru: {
    authentication: 'Аутентификация',
    azureCli: 'Вход через Azure CLI',
    command: 'Команда авторизации',
    commandLabel: 'Команда получения токена',
    azureCliHint:
      'Используется аккаунт Azure CLI на машине этого профиля. Установите там Azure CLI и выполните az login. Токены обновляются автоматически. Для Foundry выбран Responses API.',
    commandHint:
      'Выполняется на машине этого профиля. Выводит только токен или JSON с access_token и сроком действия. Профиль сохраняет команду, но не токены.',
    commandPlaceholder: 'Команда, выводящая токен доступа',
    savedCommandPlaceholder: 'Оставьте пустым, чтобы сохранить текущую команду'
  },
  ar: {
    authentication: 'المصادقة',
    azureCli: 'تسجيل الدخول عبر Azure CLI',
    command: 'أمر بيانات الاعتماد',
    commandLabel: 'أمر الرمز',
    azureCliHint:
      'يستخدم حساب Azure CLI على الجهاز الذي يشغل هذا الملف الشخصي. ثبّت Azure CLI وشغّل az login هناك أولا. تتجدد الرموز تلقائيا. يتم اختيار Responses API لـ Foundry.',
    commandHint:
      'يعمل على الجهاز المضيف لهذا الملف الشخصي. يطبع رمزا فقط أو JSON يحتوي على access_token ووقت انتهاء الصلاحية. يحفظ الملف الشخصي الأمر ولا يحفظ الرموز.',
    commandPlaceholder: 'أمر يطبع رمز الوصول',
    savedCommandPlaceholder: 'اتركه فارغا للاحتفاظ بالأمر المحفوظ'
  }
}
