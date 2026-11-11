import { type BundledLocale, isBundledLocale, useI18n } from '@/i18n'

/** Copy the redesigned global Settings pages draw that classic never shows (Figma 856:*, 895:*):
 *  page descriptions, section names and the few row labels that differ from classic. */
export interface SettingsPagesCopy {
  archive: string
  newSessions: string
  version: string
  intro: { account: string; gateways: string; sessions: string; about: string }
  gateway: {
    connectionMode: string
    thisHost: string
    saved: string
    savedDesc: string
    manage: string
    advancedDesc: string
  }
  about: { whatsNew: string; checkForUpdates: string; gateway: string; detailsDesc: string }
  keybinds: { bindings: string; all: string; allDesc: string; reset: string }
  notifications: { advancedDesc: string }
  account: { planCredits: string }
  schedules: { scheduled: string; scheduledDesc: string }
  model: { fallbacks: string; fallbacksDesc: string }
  providers: { addKey: string }
  messaging: { channels: string; channelsDesc: string; addChannel: string; manage: string; reconnect: string }
  vault: { credentials: string; credentialsDesc: string; managers: string; managersDesc: string }
}

const COPY: Record<BundledLocale, SettingsPagesCopy> = {
  en: {
    archive: 'Archive',
    newSessions: 'New sessions',
    version: 'Version',
    intro: {
      account: 'Your Nous account: plan, credits and payment. The same on every device you sign in to.',
      gateways:
        'Where your agents run, and the machines Hermes can reach. Agents are discovered from the gateways you connect.',
      sessions: 'Chat history and where new sessions start, across all agents.',
      about: 'Version details for this app and your gateways.'
    },
    gateway: {
      connectionMode: 'Connection mode',
      thisHost: 'This host',
      saved: 'Saved connections',
      savedDesc: 'Every Hermes gateway this app can reach.',
      manage: 'Manage',
      advancedDesc: 'Remote updates, startup and diagnostics'
    },
    about: {
      whatsNew: 'What’s new',
      checkForUpdates: 'Check for updates',
      gateway: 'Gateway',
      detailsDesc: 'Build, runtime and install details'
    },
    keybinds: {
      bindings: 'Key bindings',
      all: 'All shortcuts',
      allDesc: 'The full, editable list of key bindings.',
      reset: 'Reset shortcuts to defaults'
    },
    notifications: { advancedDesc: 'Less common alerts' },
    account: { planCredits: 'Plan & credits' },
    schedules: { scheduled: 'Scheduled', scheduledDesc: 'Each run starts fresh and posts where you choose.' },
    model: { fallbacks: 'Fallbacks', fallbacksDesc: 'Tried in order if the main model fails.' },
    providers: { addKey: 'Add key' },
    messaging: {
      channels: 'Channels',
      channelsDesc: 'Messages sent here reach this agent.',
      addChannel: 'Add channel',
      manage: 'Manage',
      reconnect: 'Reconnect'
    },
    vault: {
      credentials: 'Saved credentials',
      credentialsDesc: 'Logins, cards and addresses, stored encrypted. The model never sees the password.',
      managers: 'Password managers',
      managersDesc: 'Unlock once per session to let the agent use them.'
    }
  },
  fr: {
    archive: 'Archives',
    newSessions: 'Nouvelles sessions',
    version: 'Version',
    intro: {
      account:
        'Votre compte Nous : forfait, crédits et paiement. Identique sur chaque appareil où vous vous connectez.',
      gateways:
        'Où vos agents s’exécutent, et les machines que Hermes peut joindre. Les agents sont découverts via les passerelles connectées.',
      sessions: 'L’historique des discussions et le point de départ des nouvelles sessions, pour tous les agents.',
      about: 'Les versions de cette app et de vos passerelles.'
    },
    gateway: {
      connectionMode: 'Mode de connexion',
      thisHost: 'Cet hôte',
      saved: 'Connexions enregistrées',
      savedDesc: 'Chaque passerelle Hermes que cette app peut joindre.',
      manage: 'Gérer',
      advancedDesc: 'Mises à jour distantes, démarrage et diagnostics'
    },
    about: {
      whatsNew: 'Nouveautés',
      checkForUpdates: 'Rechercher des mises à jour',
      gateway: 'Passerelle',
      detailsDesc: 'Détails de build, d’exécution et d’installation'
    },
    keybinds: {
      bindings: 'Raccourcis',
      all: 'Tous les raccourcis',
      allDesc: 'La liste complète et modifiable des raccourcis.',
      reset: 'Rétablir les raccourcis par défaut'
    },
    notifications: { advancedDesc: 'Alertes moins courantes' },
    account: { planCredits: 'Forfait et crédits' },
    schedules: {
      scheduled: 'Planifiées',
      scheduledDesc: 'Chaque exécution repart de zéro et publie où vous le choisissez.'
    },
    model: { fallbacks: 'Secours', fallbacksDesc: 'Essayés dans l’ordre si le modèle principal échoue.' },
    providers: { addKey: 'Ajouter une clé' },
    messaging: {
      channels: 'Canaux',
      channelsDesc: 'Les messages envoyés ici parviennent à cet agent.',
      addChannel: 'Ajouter un canal',
      manage: 'Gérer',
      reconnect: 'Reconnecter'
    },
    vault: {
      credentials: 'Identifiants enregistrés',
      credentialsDesc: 'Identifiants, cartes et adresses, stockés chiffrés. Le modèle ne voit jamais le mot de passe.',
      managers: 'Gestionnaires de mots de passe',
      managersDesc: 'Déverrouillez-les une fois par session pour que l’agent puisse les utiliser.'
    }
  },
  de: {
    archive: 'Archiv',
    newSessions: 'Neue Sitzungen',
    version: 'Version',
    intro: {
      account: 'Dein Nous-Konto: Tarif, Guthaben und Zahlung. Auf jedem Gerät gleich, auf dem du angemeldet bist.',
      gateways:
        'Wo deine Agenten laufen und welche Maschinen Hermes erreicht. Agenten werden über die verbundenen Gateways gefunden.',
      sessions: 'Chatverlauf und wo neue Sitzungen starten, für alle Agenten.',
      about: 'Versionen dieser App und deiner Gateways.'
    },
    gateway: {
      connectionMode: 'Verbindungsart',
      thisHost: 'Dieser Host',
      saved: 'Gespeicherte Verbindungen',
      savedDesc: 'Jedes Hermes-Gateway, das diese App erreicht.',
      manage: 'Verwalten',
      advancedDesc: 'Remote-Updates, Start und Diagnose'
    },
    about: {
      whatsNew: 'Neuigkeiten',
      checkForUpdates: 'Nach Updates suchen',
      gateway: 'Gateway',
      detailsDesc: 'Build-, Laufzeit- und Installationsdetails'
    },
    keybinds: {
      bindings: 'Tastenbelegung',
      all: 'Alle Tastenkürzel',
      allDesc: 'Die vollständige, bearbeitbare Liste der Tastenkürzel.',
      reset: 'Tastenkürzel zurücksetzen'
    },
    notifications: { advancedDesc: 'Seltenere Hinweise' },
    account: { planCredits: 'Tarif & Guthaben' },
    schedules: { scheduled: 'Geplant', scheduledDesc: 'Jeder Lauf startet frisch und postet, wo du es festlegst.' },
    model: { fallbacks: 'Ausweichmodelle', fallbacksDesc: 'Der Reihe nach versucht, wenn das Hauptmodell ausfällt.' },
    providers: { addKey: 'Schlüssel hinzufügen' },
    messaging: {
      channels: 'Kanäle',
      channelsDesc: 'Nachrichten hier erreichen diesen Agenten.',
      addChannel: 'Kanal hinzufügen',
      manage: 'Verwalten',
      reconnect: 'Neu verbinden'
    },
    vault: {
      credentials: 'Gespeicherte Zugangsdaten',
      credentialsDesc: 'Logins, Karten und Adressen, verschlüsselt gespeichert. Das Modell sieht nie das Passwort.',
      managers: 'Passwortmanager',
      managersDesc: 'Einmal pro Sitzung entsperren, damit der Agent sie nutzen kann.'
    }
  },
  es: {
    archive: 'Archivo',
    newSessions: 'Sesiones nuevas',
    version: 'Versión',
    intro: {
      account: 'Tu cuenta de Nous: plan, créditos y pago. Igual en cada dispositivo donde inicies sesión.',
      gateways:
        'Dónde se ejecutan tus agentes y qué máquinas puede alcanzar Hermes. Los agentes se descubren desde los gateways que conectas.',
      sessions: 'El historial de chats y dónde empiezan las sesiones nuevas, en todos los agentes.',
      about: 'Las versiones de esta app y de tus gateways.'
    },
    gateway: {
      connectionMode: 'Modo de conexión',
      thisHost: 'Este host',
      saved: 'Conexiones guardadas',
      savedDesc: 'Cada gateway de Hermes que esta app puede alcanzar.',
      manage: 'Gestionar',
      advancedDesc: 'Actualizaciones remotas, inicio y diagnóstico'
    },
    about: {
      whatsNew: 'Novedades',
      checkForUpdates: 'Buscar actualizaciones',
      gateway: 'Gateway',
      detailsDesc: 'Detalles de compilación, ejecución e instalación'
    },
    keybinds: {
      bindings: 'Atajos',
      all: 'Todos los atajos',
      allDesc: 'La lista completa y editable de atajos.',
      reset: 'Restablecer atajos predeterminados'
    },
    notifications: { advancedDesc: 'Alertas menos habituales' },
    account: { planCredits: 'Plan y créditos' },
    schedules: { scheduled: 'Programadas', scheduledDesc: 'Cada ejecución empieza de cero y publica donde elijas.' },
    model: { fallbacks: 'Alternativos', fallbacksDesc: 'Se prueban en orden si falla el modelo principal.' },
    providers: { addKey: 'Añadir clave' },
    messaging: {
      channels: 'Canales',
      channelsDesc: 'Los mensajes enviados aquí llegan a este agente.',
      addChannel: 'Añadir canal',
      manage: 'Gestionar',
      reconnect: 'Reconectar'
    },
    vault: {
      credentials: 'Credenciales guardadas',
      credentialsDesc: 'Inicios de sesión, tarjetas y direcciones, cifrados. El modelo nunca ve la contraseña.',
      managers: 'Gestores de contraseñas',
      managersDesc: 'Desbloquéalos una vez por sesión para que el agente pueda usarlos.'
    }
  },
  ja: {
    archive: 'アーカイブ',
    newSessions: '新しいセッション',
    version: 'バージョン',
    intro: {
      account: 'Nous アカウント：プラン、クレジット、支払い。サインインしたどのデバイスでも同じです。',
      gateways:
        'エージェントが動く場所と、Hermes が接続できるマシン。エージェントは接続したゲートウェイから検出されます。',
      sessions: 'すべてのエージェントのチャット履歴と、新しいセッションの開始場所。',
      about: 'このアプリとゲートウェイのバージョン情報。'
    },
    gateway: {
      connectionMode: '接続方法',
      thisHost: 'このホスト',
      saved: '保存済みの接続',
      savedDesc: 'このアプリが接続できる Hermes ゲートウェイ。',
      manage: '管理',
      advancedDesc: 'リモート更新、起動、診断'
    },
    about: {
      whatsNew: '新機能',
      checkForUpdates: 'アップデートを確認',
      gateway: 'ゲートウェイ',
      detailsDesc: 'ビルド、ランタイム、インストールの詳細'
    },
    keybinds: {
      bindings: 'キー割り当て',
      all: 'すべてのショートカット',
      allDesc: '編集できるキー割り当ての一覧。',
      reset: 'ショートカットを既定に戻す'
    },
    notifications: { advancedDesc: 'あまり使わない通知' },
    account: { planCredits: 'プランとクレジット' },
    schedules: { scheduled: 'スケジュール済み', scheduledDesc: '実行ごとに新しく始まり、選んだ場所に投稿します。' },
    model: { fallbacks: 'フォールバック', fallbacksDesc: 'メインモデルが失敗したときに順番に試します。' },
    providers: { addKey: 'キーを追加' },
    messaging: {
      channels: 'チャネル',
      channelsDesc: 'ここに送ったメッセージはこのエージェントに届きます。',
      addChannel: 'チャネルを追加',
      manage: '管理',
      reconnect: '再接続'
    },
    vault: {
      credentials: '保存済みの認証情報',
      credentialsDesc: 'ログイン、カード、住所を暗号化して保存します。モデルがパスワードを見ることはありません。',
      managers: 'パスワードマネージャー',
      managersDesc: 'セッションごとに一度ロックを解除すると、エージェントが利用できます。'
    }
  },
  zh: {
    archive: '归档',
    newSessions: '新会话',
    version: '版本',
    intro: {
      account: '你的 Nous 账户：套餐、额度和付款。在你登录的每台设备上都一样。',
      gateways: '你的智能体在哪里运行，以及 Hermes 能连接的机器。智能体从你连接的网关中发现。',
      sessions: '所有智能体的聊天记录，以及新会话的起始位置。',
      about: '此应用和你的网关的版本信息。'
    },
    gateway: {
      connectionMode: '连接方式',
      thisHost: '此主机',
      saved: '已保存的连接',
      savedDesc: '此应用能连接的每个 Hermes 网关。',
      manage: '管理',
      advancedDesc: '远程更新、启动和诊断'
    },
    about: {
      whatsNew: '新功能',
      checkForUpdates: '检查更新',
      gateway: '网关',
      detailsDesc: '构建、运行时和安装详情'
    },
    keybinds: {
      bindings: '按键绑定',
      all: '全部快捷键',
      allDesc: '完整且可编辑的按键绑定列表。',
      reset: '恢复默认快捷键'
    },
    notifications: { advancedDesc: '不常用的提醒' },
    account: { planCredits: '套餐与额度' },
    schedules: { scheduled: '已计划', scheduledDesc: '每次运行都会重新开始，并发布到你选择的位置。' },
    model: { fallbacks: '备用模型', fallbacksDesc: '主模型失败时按顺序尝试。' },
    providers: { addKey: '添加密钥' },
    messaging: {
      channels: '渠道',
      channelsDesc: '发送到这里的消息会到达此智能体。',
      addChannel: '添加渠道',
      manage: '管理',
      reconnect: '重新连接'
    },
    vault: {
      credentials: '已保存的凭据',
      credentialsDesc: '登录信息、卡片和地址，加密存储。模型永远看不到密码。',
      managers: '密码管理器',
      managersDesc: '每个会话解锁一次，智能体即可使用。'
    }
  },
  'zh-hant': {
    archive: '封存',
    newSessions: '新工作階段',
    version: '版本',
    intro: {
      account: '你的 Nous 帳戶：方案、點數和付款。在你登入的每部裝置上都相同。',
      gateways: '你的代理在哪裡執行，以及 Hermes 能連線的機器。代理會從你連線的閘道中找到。',
      sessions: '所有代理的聊天紀錄，以及新工作階段的起始位置。',
      about: '此 App 與你的閘道的版本資訊。'
    },
    gateway: {
      connectionMode: '連線方式',
      thisHost: '此主機',
      saved: '已儲存的連線',
      savedDesc: '此 App 能連線的每個 Hermes 閘道。',
      manage: '管理',
      advancedDesc: '遠端更新、啟動與診斷'
    },
    about: {
      whatsNew: '新功能',
      checkForUpdates: '檢查更新',
      gateway: '閘道',
      detailsDesc: '建置、執行環境與安裝詳細資料'
    },
    keybinds: {
      bindings: '按鍵綁定',
      all: '所有快速鍵',
      allDesc: '完整且可編輯的按鍵綁定清單。',
      reset: '還原預設快速鍵'
    },
    notifications: { advancedDesc: '較少用的提醒' },
    account: { planCredits: '方案與點數' },
    schedules: { scheduled: '已排程', scheduledDesc: '每次執行都會重新開始，並發布到你選擇的位置。' },
    model: { fallbacks: '備用模型', fallbacksDesc: '主要模型失敗時依序嘗試。' },
    providers: { addKey: '新增金鑰' },
    messaging: {
      channels: '頻道',
      channelsDesc: '傳送到這裡的訊息會送達此代理。',
      addChannel: '新增頻道',
      manage: '管理',
      reconnect: '重新連線'
    },
    vault: {
      credentials: '已儲存的憑證',
      credentialsDesc: '登入資訊、卡片和地址，加密儲存。模型永遠看不到密碼。',
      managers: '密碼管理工具',
      managersDesc: '每個工作階段解鎖一次，代理即可使用。'
    }
  },
  ru: {
    archive: 'Архив',
    newSessions: 'Новые сеансы',
    version: 'Версия',
    intro: {
      account: 'Ваш аккаунт Nous: тариф, кредиты и оплата. Одинаков на всех устройствах, где вы вошли.',
      gateways: 'Где работают ваши агенты и какие машины доступны Hermes. Агенты находятся через подключённые шлюзы.',
      sessions: 'История чатов и место, где начинаются новые сеансы, для всех агентов.',
      about: 'Версии этого приложения и ваших шлюзов.'
    },
    gateway: {
      connectionMode: 'Способ подключения',
      thisHost: 'Этот хост',
      saved: 'Сохранённые подключения',
      savedDesc: 'Все шлюзы Hermes, доступные этому приложению.',
      manage: 'Управлять',
      advancedDesc: 'Удалённые обновления, запуск и диагностика'
    },
    about: {
      whatsNew: 'Что нового',
      checkForUpdates: 'Проверить обновления',
      gateway: 'Шлюз',
      detailsDesc: 'Сборка, среда выполнения и установка'
    },
    keybinds: {
      bindings: 'Сочетания клавиш',
      all: 'Все сочетания',
      allDesc: 'Полный редактируемый список сочетаний клавиш.',
      reset: 'Сбросить сочетания по умолчанию'
    },
    notifications: { advancedDesc: 'Редкие уведомления' },
    account: { planCredits: 'Тариф и кредиты' },
    schedules: {
      scheduled: 'Запланированные',
      scheduledDesc: 'Каждый запуск начинается с нуля и публикует туда, куда вы выберете.'
    },
    model: { fallbacks: 'Резервные', fallbacksDesc: 'Пробуются по порядку, если основная модель не сработала.' },
    providers: { addKey: 'Добавить ключ' },
    messaging: {
      channels: 'Каналы',
      channelsDesc: 'Сообщения, отправленные сюда, получает этот агент.',
      addChannel: 'Добавить канал',
      manage: 'Управлять',
      reconnect: 'Переподключить'
    },
    vault: {
      credentials: 'Сохранённые данные входа',
      credentialsDesc: 'Логины, карты и адреса хранятся зашифрованными. Модель никогда не видит пароль.',
      managers: 'Менеджеры паролей',
      managersDesc: 'Разблокируйте один раз за сеанс, чтобы агент мог ими пользоваться.'
    }
  },
  ar: {
    archive: 'الأرشيف',
    newSessions: 'الجلسات الجديدة',
    version: 'الإصدار',
    intro: {
      account: 'حسابك في Nous: الخطة والأرصدة والدفع. هو نفسه على كل جهاز تسجّل الدخول منه.',
      gateways: 'أين تعمل وكلاؤك، والأجهزة التي يصل إليها Hermes. تُكتشف الوكلاء من البوابات التي توصلها.',
      sessions: 'سجل المحادثات وأين تبدأ الجلسات الجديدة، لكل الوكلاء.',
      about: 'تفاصيل إصدار هذا التطبيق وبواباتك.'
    },
    gateway: {
      connectionMode: 'طريقة الاتصال',
      thisHost: 'هذا المضيف',
      saved: 'الاتصالات المحفوظة',
      savedDesc: 'كل بوابة Hermes يمكن لهذا التطبيق الوصول إليها.',
      manage: 'إدارة',
      advancedDesc: 'التحديثات عن بُعد وبدء التشغيل والتشخيص'
    },
    about: {
      whatsNew: 'الجديد',
      checkForUpdates: 'التحقق من التحديثات',
      gateway: 'البوابة',
      detailsDesc: 'تفاصيل البناء وبيئة التشغيل والتثبيت'
    },
    keybinds: {
      bindings: 'اختصارات المفاتيح',
      all: 'كل الاختصارات',
      allDesc: 'القائمة الكاملة القابلة للتعديل لاختصارات المفاتيح.',
      reset: 'إعادة الاختصارات إلى الافتراضي'
    },
    notifications: { advancedDesc: 'تنبيهات أقل شيوعًا' },
    account: { planCredits: 'الخطة والأرصدة' },
    schedules: { scheduled: 'المجدولة', scheduledDesc: 'يبدأ كل تشغيل من جديد وينشر حيث تختار.' },
    model: { fallbacks: 'البدائل', fallbacksDesc: 'تُجرَّب بالترتيب إذا فشل النموذج الرئيسي.' },
    providers: { addKey: 'إضافة مفتاح' },
    messaging: {
      channels: 'القنوات',
      channelsDesc: 'الرسائل المرسلة هنا تصل إلى هذا الوكيل.',
      addChannel: 'إضافة قناة',
      manage: 'إدارة',
      reconnect: 'إعادة الاتصال'
    },
    vault: {
      credentials: 'بيانات الدخول المحفوظة',
      credentialsDesc: 'تسجيلات الدخول والبطاقات والعناوين مخزنة مشفرة. لا يرى النموذج كلمة المرور أبدًا.',
      managers: 'مديرو كلمات المرور',
      managersDesc: 'افتحها مرة واحدة لكل جلسة ليستخدمها الوكيل.'
    }
  }
}

export function settingsPagesCopy(locale: string): SettingsPagesCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useSettingsPagesCopy(): SettingsPagesCopy {
  return settingsPagesCopy(useI18n().locale)
}
