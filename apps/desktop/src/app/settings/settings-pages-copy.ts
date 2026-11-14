import { type BundledLocale, isBundledLocale, useI18n } from '@/i18n'
import { hostHas } from '@/lib/host-capabilities'

/** Copy the redesigned global Settings pages draw that classic never shows (Figma 856:*, 895:*):
 *  page descriptions, section names and the few row labels that differ from classic. */
export interface SettingsPagesCopy {
  archive: string
  newSessions: string
  version: string
  intro: { account: string; gateways: string; sessions: string; about: string }
  /** Device-scoped pages, worded for where this renderer runs (Figma web note 832:2953). */
  hostIntro: { notifications: HostWording; keybinds: HostWording }
  /** The quiet note on a section whose settings live in this renderer's storage (856:2641). */
  savedHere: HostWording
  archiveChats: {
    period: string
    off: string
    afterDays: (days: number) => string
    archived: string
    archivedDesc: string
    view: string
    hide: string
  }
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
  /** `browserAlerts` names the alerts group in a browser tab (856:4096); the app keeps its own. */
  notifications: { advancedDesc: string; browserAlerts: string; completionSound: string }
  account: { planCredits: string }
  schedules: { scheduled: string; scheduledDesc: string }
  model: { fallbacks: string; fallbacksDesc: string }
  providers: { addKey: string }
  messaging: { channels: string; channelsDesc: string; addChannel: string; manage: string; reconnect: string }
  vault: { credentials: string; credentialsDesc: string; managers: string; managersDesc: string }
  /** Export / import / reset of the profile's config, which classic keeps as rail-footer icons. */
  configuration: {
    title: string
    exportDesc: string
    importDesc: string
    resetDesc: string
    export: string
    import: string
    reset: string
  }
}

interface HostWording {
  browser: string
  device: string
}

/** The wording for where this renderer runs: a browser tab (the Webapp) or the desktop app. */
export function forHost(wording: HostWording): string {
  return hostHas('webappCopy') ? wording.browser : wording.device
}

const COPY: Record<BundledLocale, SettingsPagesCopy> = {
  en: {
    archive: 'Archive',
    newSessions: 'New chats',
    version: 'Version',
    hostIntro: {
      notifications: {
        browser: 'Alerts from this browser when Hermes needs you while the tab is in the background.',
        device: 'Alerts from this computer when Hermes needs you while the app is in the background.'
      },
      keybinds: {
        browser: 'Key bindings for this browser. Click a shortcut to change it.',
        device: 'Key bindings for this computer. Click a shortcut to change it.'
      }
    },
    savedHere: { browser: 'Saved to this browser', device: 'Saved on this computer' },
    archiveChats: {
      period: 'Archive after',
      off: 'Never',
      afterDays: days => (days === 1 ? 'After 1 day' : `After ${days} days`),
      archived: 'Archived chats',
      archivedDesc: 'Unarchive chats or delete them permanently.',
      view: 'View archive',
      hide: 'Hide archive'
    },
    intro: {
      account: 'Your Nous account: plan, credits and payment. The same on every device you sign in to.',
      gateways:
        'Where your agents run, and the machines Hermes can reach. Agents are discovered from the machines you connect.',
      sessions: 'Chat history and where new chats start, across all agents.',
      about: 'Version details for this app and your machines.'
    },
    gateway: {
      connectionMode: 'Connection mode',
      thisHost: 'This machine',
      saved: 'Saved connections',
      savedDesc: 'Every machine running Hermes that this app can reach.',
      manage: 'Manage',
      advancedDesc: 'Remote updates, startup and diagnostics'
    },
    about: {
      whatsNew: 'What’s new',
      checkForUpdates: 'Check for updates',
      gateway: 'Machine',
      detailsDesc: 'Build, runtime and install details'
    },
    keybinds: {
      bindings: 'Key bindings',
      all: 'All shortcuts',
      allDesc: 'The full, editable list of key bindings.',
      reset: 'Reset shortcuts to defaults'
    },
    notifications: {
      advancedDesc: 'Less common alerts',
      browserAlerts: 'Browser notifications',
      completionSound: 'Completion sound'
    },
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
    },
    configuration: {
      title: 'Configuration',
      exportDesc: "Download this agent's settings as a JSON file.",
      importDesc: 'Replace settings with a previously exported JSON file.',
      resetDesc: 'Return every setting to Hermes defaults. This cannot be undone.',
      export: 'Export',
      import: 'Import',
      reset: 'Reset…'
    }
  },
  fr: {
    archive: 'Archives',
    newSessions: 'Nouvelles discussions',
    version: 'Version',
    hostIntro: {
      notifications: {
        browser: 'Alertes de ce navigateur quand Hermes a besoin de vous et que l’onglet est en arrière-plan.',
        device: 'Alertes de cet ordinateur quand Hermes a besoin de vous et que l’app est en arrière-plan.'
      },
      keybinds: {
        browser: 'Raccourcis de ce navigateur. Cliquez sur un raccourci pour le modifier.',
        device: 'Raccourcis de cet ordinateur. Cliquez sur un raccourci pour le modifier.'
      }
    },
    savedHere: { browser: 'Enregistré dans ce navigateur', device: 'Enregistré sur cet ordinateur' },
    archiveChats: {
      period: 'Archiver après',
      off: 'Jamais',
      afterDays: days => (days === 1 ? 'Après 1 jour' : `Après ${days} jours`),
      archived: 'Discussions archivées',
      archivedDesc: 'Désarchivez des discussions ou supprimez-les définitivement.',
      view: 'Voir les archives',
      hide: 'Masquer les archives'
    },
    intro: {
      account:
        'Votre compte Nous : forfait, crédits et paiement. Identique sur chaque appareil où vous vous connectez.',
      gateways:
        'Où vos agents s’exécutent, et les machines que Hermes peut joindre. Les agents sont découverts via les machines connectées.',
      sessions: 'L’historique des discussions et le point de départ des nouvelles discussions, pour tous les agents.',
      about: 'Les versions de cette app et de vos machines.'
    },
    gateway: {
      connectionMode: 'Mode de connexion',
      thisHost: 'Cette machine',
      saved: 'Connexions enregistrées',
      savedDesc: 'Chaque machine exécutant Hermes que cette app peut joindre.',
      manage: 'Gérer',
      advancedDesc: 'Mises à jour distantes, démarrage et diagnostics'
    },
    about: {
      whatsNew: 'Nouveautés',
      checkForUpdates: 'Rechercher des mises à jour',
      gateway: 'Machine',
      detailsDesc: 'Détails de build, d’exécution et d’installation'
    },
    keybinds: {
      bindings: 'Raccourcis',
      all: 'Tous les raccourcis',
      allDesc: 'La liste complète et modifiable des raccourcis.',
      reset: 'Rétablir les raccourcis par défaut'
    },
    notifications: {
      advancedDesc: 'Alertes moins courantes',
      browserAlerts: 'Notifications du navigateur',
      completionSound: 'Son de fin'
    },
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
    },
    configuration: {
      title: 'Configuration',
      exportDesc: 'Télécharger les réglages de cet agent dans un fichier JSON.',
      importDesc: 'Remplacer les réglages par un fichier JSON exporté.',
      resetDesc: 'Rétablir tous les réglages par défaut de Hermes. Action irréversible.',
      export: 'Exporter',
      import: 'Importer',
      reset: 'Réinitialiser…'
    }
  },
  de: {
    archive: 'Archiv',
    newSessions: 'Neue Chats',
    version: 'Version',
    hostIntro: {
      notifications: {
        browser: 'Hinweise von diesem Browser, wenn Hermes dich braucht und der Tab im Hintergrund ist.',
        device: 'Hinweise von diesem Computer, wenn Hermes dich braucht und die App im Hintergrund ist.'
      },
      keybinds: {
        browser: 'Tastenkürzel für diesen Browser. Klicke auf ein Kürzel, um es zu ändern.',
        device: 'Tastenkürzel für diesen Computer. Klicke auf ein Kürzel, um es zu ändern.'
      }
    },
    savedHere: { browser: 'In diesem Browser gespeichert', device: 'Auf diesem Computer gespeichert' },
    archiveChats: {
      period: 'Archivieren nach',
      off: 'Nie',
      afterDays: days => (days === 1 ? 'Nach 1 Tag' : `Nach ${days} Tagen`),
      archived: 'Archivierte Chats',
      archivedDesc: 'Chats wiederherstellen oder endgültig löschen.',
      view: 'Archiv anzeigen',
      hide: 'Archiv ausblenden'
    },
    intro: {
      account: 'Dein Nous-Konto: Tarif, Guthaben und Zahlung. Auf jedem Gerät gleich, auf dem du angemeldet bist.',
      gateways:
        'Wo deine Agenten laufen und welche Rechner Hermes erreicht. Agenten werden über die verbundenen Rechner gefunden.',
      sessions: 'Chatverlauf und wo neue Chats starten, für alle Agenten.',
      about: 'Versionen dieser App und deiner Rechner.'
    },
    gateway: {
      connectionMode: 'Verbindungsart',
      thisHost: 'Dieser Rechner',
      saved: 'Gespeicherte Verbindungen',
      savedDesc: 'Jeder Rechner mit Hermes, den diese App erreicht.',
      manage: 'Verwalten',
      advancedDesc: 'Remote-Updates, Start und Diagnose'
    },
    about: {
      whatsNew: 'Neuigkeiten',
      checkForUpdates: 'Nach Updates suchen',
      gateway: 'Rechner',
      detailsDesc: 'Build-, Laufzeit- und Installationsdetails'
    },
    keybinds: {
      bindings: 'Tastenbelegung',
      all: 'Alle Tastenkürzel',
      allDesc: 'Die vollständige, bearbeitbare Liste der Tastenkürzel.',
      reset: 'Tastenkürzel zurücksetzen'
    },
    notifications: {
      advancedDesc: 'Seltenere Hinweise',
      browserAlerts: 'Browser-Benachrichtigungen',
      completionSound: 'Abschlusston'
    },
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
    },
    configuration: {
      title: 'Konfiguration',
      exportDesc: 'Die Einstellungen dieses Agenten als JSON-Datei herunterladen.',
      importDesc: 'Einstellungen durch eine exportierte JSON-Datei ersetzen.',
      resetDesc: 'Alle Einstellungen auf die Hermes-Standardwerte zurücksetzen. Nicht umkehrbar.',
      export: 'Exportieren',
      import: 'Importieren',
      reset: 'Zurücksetzen…'
    }
  },
  es: {
    archive: 'Archivo',
    newSessions: 'Chats nuevos',
    version: 'Versión',
    hostIntro: {
      notifications: {
        browser: 'Avisos de este navegador cuando Hermes te necesita y la pestaña está en segundo plano.',
        device: 'Avisos de este ordenador cuando Hermes te necesita y la app está en segundo plano.'
      },
      keybinds: {
        browser: 'Atajos de teclado de este navegador. Haz clic en un atajo para cambiarlo.',
        device: 'Atajos de teclado de este ordenador. Haz clic en un atajo para cambiarlo.'
      }
    },
    savedHere: { browser: 'Guardado en este navegador', device: 'Guardado en este ordenador' },
    archiveChats: {
      period: 'Archivar tras',
      off: 'Nunca',
      afterDays: days => (days === 1 ? 'Tras 1 día' : `Tras ${days} días`),
      archived: 'Chats archivados',
      archivedDesc: 'Desarchiva chats o elimínalos para siempre.',
      view: 'Ver archivo',
      hide: 'Ocultar archivo'
    },
    intro: {
      account: 'Tu cuenta de Nous: plan, créditos y pago. Igual en cada dispositivo donde inicies sesión.',
      gateways:
        'Dónde se ejecutan tus agentes y qué máquinas puede alcanzar Hermes. Los agentes se descubren desde las máquinas que conectas.',
      sessions: 'El historial de chats y dónde empiezan los chats nuevos, en todos los agentes.',
      about: 'Las versiones de esta app y de tus máquinas.'
    },
    gateway: {
      connectionMode: 'Modo de conexión',
      thisHost: 'Esta máquina',
      saved: 'Conexiones guardadas',
      savedDesc: 'Cada máquina con Hermes que esta app puede alcanzar.',
      manage: 'Gestionar',
      advancedDesc: 'Actualizaciones remotas, inicio y diagnóstico'
    },
    about: {
      whatsNew: 'Novedades',
      checkForUpdates: 'Buscar actualizaciones',
      gateway: 'Máquina',
      detailsDesc: 'Detalles de compilación, ejecución e instalación'
    },
    keybinds: {
      bindings: 'Atajos',
      all: 'Todos los atajos',
      allDesc: 'La lista completa y editable de atajos.',
      reset: 'Restablecer atajos predeterminados'
    },
    notifications: {
      advancedDesc: 'Alertas menos habituales',
      browserAlerts: 'Notificaciones del navegador',
      completionSound: 'Sonido de finalización'
    },
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
    },
    configuration: {
      title: 'Configuración',
      exportDesc: 'Descarga los ajustes de este agente como archivo JSON.',
      importDesc: 'Sustituye los ajustes por un archivo JSON exportado.',
      resetDesc: 'Restablece todos los ajustes a los valores de Hermes. No se puede deshacer.',
      export: 'Exportar',
      import: 'Importar',
      reset: 'Restablecer…'
    }
  },
  ja: {
    archive: 'アーカイブ',
    newSessions: '新しいチャット',
    version: 'バージョン',
    hostIntro: {
      notifications: {
        browser: 'タブがバックグラウンドのときに Hermes が呼びかける、このブラウザーの通知です。',
        device: 'アプリがバックグラウンドのときに Hermes が呼びかける、このコンピューターの通知です。'
      },
      keybinds: {
        browser: 'このブラウザーのキー割り当てです。ショートカットをクリックして変更します。',
        device: 'このコンピューターのキー割り当てです。ショートカットをクリックして変更します。'
      }
    },
    savedHere: { browser: 'このブラウザーに保存', device: 'このコンピューターに保存' },
    archiveChats: {
      period: 'アーカイブまでの期間',
      off: 'しない',
      afterDays: days => `${days}日後`,
      archived: 'アーカイブ済みのチャット',
      archivedDesc: 'チャットをアーカイブから戻すか、完全に削除します。',
      view: 'アーカイブを表示',
      hide: 'アーカイブを隠す'
    },
    intro: {
      account: 'Nous アカウント：プラン、クレジット、支払い。サインインしたどのデバイスでも同じです。',
      gateways:
        'エージェントが動く場所と、Hermes が接続できるマシン。エージェントは接続したマシンから検出されます。',
      sessions: 'すべてのエージェントのチャット履歴と、新しいチャットの開始場所。',
      about: 'このアプリとマシンのバージョン情報。'
    },
    gateway: {
      connectionMode: '接続方法',
      thisHost: 'このマシン',
      saved: '保存済みの接続',
      savedDesc: 'このアプリが接続できる、Hermes が動作しているすべてのマシン。',
      manage: '管理',
      advancedDesc: 'リモート更新、起動、診断'
    },
    about: {
      whatsNew: '新機能',
      checkForUpdates: 'アップデートを確認',
      gateway: 'マシン',
      detailsDesc: 'ビルド、ランタイム、インストールの詳細'
    },
    keybinds: {
      bindings: 'キー割り当て',
      all: 'すべてのショートカット',
      allDesc: '編集できるキー割り当ての一覧。',
      reset: 'ショートカットを既定に戻す'
    },
    notifications: { advancedDesc: 'あまり使わない通知', browserAlerts: 'ブラウザー通知', completionSound: '完了音' },
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
    },
    configuration: {
      title: '構成',
      exportDesc: 'このエージェントの設定を JSON ファイルとしてダウンロードします。',
      importDesc: 'エクスポートした JSON ファイルで設定を置き換えます。',
      resetDesc: 'すべての設定を Hermes の既定値に戻します。元に戻せません。',
      export: 'エクスポート',
      import: 'インポート',
      reset: 'リセット…'
    }
  },
  zh: {
    archive: '归档',
    newSessions: '新对话',
    version: '版本',
    hostIntro: {
      notifications: {
        browser: '标签页在后台、Hermes 需要你时，由此浏览器发出的提醒。',
        device: '应用在后台、Hermes 需要你时，由此电脑发出的提醒。'
      },
      keybinds: {
        browser: '此浏览器的快捷键。点击快捷键即可修改。',
        device: '此电脑的快捷键。点击快捷键即可修改。'
      }
    },
    savedHere: { browser: '保存在此浏览器', device: '保存在此电脑' },
    archiveChats: {
      period: '归档时间',
      off: '从不',
      afterDays: days => `${days} 天后`,
      archived: '已归档的对话',
      archivedDesc: '取消归档对话，或将其永久删除。',
      view: '查看归档',
      hide: '隐藏归档'
    },
    intro: {
      account: '你的 Nous 账户：套餐、额度和付款。在你登录的每台设备上都一样。',
      gateways: '你的智能体在哪里运行，以及 Hermes 能连接的机器。智能体从你连接的机器中发现。',
      sessions: '所有智能体的对话记录，以及新对话的起始位置。',
      about: '此应用和你的机器的版本信息。'
    },
    gateway: {
      connectionMode: '连接方式',
      thisHost: '此机器',
      saved: '已保存的连接',
      savedDesc: '此应用能连接的每台运行 Hermes 的机器。',
      manage: '管理',
      advancedDesc: '远程更新、启动和诊断'
    },
    about: {
      whatsNew: '新功能',
      checkForUpdates: '检查更新',
      gateway: '机器',
      detailsDesc: '构建、运行时和安装详情'
    },
    keybinds: {
      bindings: '按键绑定',
      all: '全部快捷键',
      allDesc: '完整且可编辑的按键绑定列表。',
      reset: '恢复默认快捷键'
    },
    notifications: { advancedDesc: '不常用的提醒', browserAlerts: '浏览器通知', completionSound: '完成提示音' },
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
    },
    configuration: {
      title: '配置',
      exportDesc: '将此智能体的设置下载为 JSON 文件。',
      importDesc: '用之前导出的 JSON 文件替换设置。',
      resetDesc: '将所有设置恢复为 Hermes 默认值。此操作无法撤销。',
      export: '导出',
      import: '导入',
      reset: '重置…'
    }
  },
  'zh-hant': {
    archive: '封存',
    newSessions: '新對話',
    version: '版本',
    hostIntro: {
      notifications: {
        browser: '分頁在背景、Hermes 需要你時，由此瀏覽器發出的提醒。',
        device: '應用程式在背景、Hermes 需要你時，由此電腦發出的提醒。'
      },
      keybinds: {
        browser: '此瀏覽器的快捷鍵。點選快捷鍵即可修改。',
        device: '此電腦的快捷鍵。點選快捷鍵即可修改。'
      }
    },
    savedHere: { browser: '儲存在此瀏覽器', device: '儲存在此電腦' },
    archiveChats: {
      period: '封存時間',
      off: '永不',
      afterDays: days => `${days} 天後`,
      archived: '已封存的對話',
      archivedDesc: '取消封存對話，或將其永久刪除。',
      view: '檢視封存',
      hide: '隱藏封存'
    },
    intro: {
      account: '你的 Nous 帳戶：方案、點數和付款。在你登入的每部裝置上都相同。',
      gateways: '你的智慧體在哪裡執行，以及 Hermes 能連線的機器。智慧體會從你連線的機器中找到。',
      sessions: '所有智慧體的對話紀錄，以及新對話的起始位置。',
      about: '此 App 與你的機器的版本資訊。'
    },
    gateway: {
      connectionMode: '連線方式',
      thisHost: '此機器',
      saved: '已儲存的連線',
      savedDesc: '此 App 能連線的每台執行 Hermes 的機器。',
      manage: '管理',
      advancedDesc: '遠端更新、啟動與診斷'
    },
    about: {
      whatsNew: '新功能',
      checkForUpdates: '檢查更新',
      gateway: '機器',
      detailsDesc: '建置、執行環境與安裝詳細資料'
    },
    keybinds: {
      bindings: '按鍵綁定',
      all: '所有快速鍵',
      allDesc: '完整且可編輯的按鍵綁定清單。',
      reset: '還原預設快速鍵'
    },
    notifications: { advancedDesc: '較少用的提醒', browserAlerts: '瀏覽器通知', completionSound: '完成提示音' },
    account: { planCredits: '方案與點數' },
    schedules: { scheduled: '已排程', scheduledDesc: '每次執行都會重新開始，並發布到你選擇的位置。' },
    model: { fallbacks: '備用模型', fallbacksDesc: '主要模型失敗時依序嘗試。' },
    providers: { addKey: '新增金鑰' },
    messaging: {
      channels: '頻道',
      channelsDesc: '傳送到這裡的訊息會送達此智慧體。',
      addChannel: '新增頻道',
      manage: '管理',
      reconnect: '重新連線'
    },
    vault: {
      credentials: '已儲存的憑證',
      credentialsDesc: '登入資訊、卡片和地址，加密儲存。模型永遠看不到密碼。',
      managers: '密碼管理工具',
      managersDesc: '每個工作階段解鎖一次，智慧體即可使用。'
    },
    configuration: {
      title: '設定',
      exportDesc: '將此智慧體的設定下載為 JSON 檔案。',
      importDesc: '以先前匯出的 JSON 檔案取代設定。',
      resetDesc: '將所有設定還原為 Hermes 預設值。此操作無法復原。',
      export: '匯出',
      import: '匯入',
      reset: '重設…'
    }
  },
  ru: {
    archive: 'Архив',
    newSessions: 'Новые чаты',
    version: 'Версия',
    hostIntro: {
      notifications: {
        browser: 'Оповещения этого браузера, когда Hermes нужен ваш ответ, а вкладка в фоне.',
        device: 'Оповещения этого компьютера, когда Hermes нужен ваш ответ, а приложение в фоне.'
      },
      keybinds: {
        browser: 'Сочетания клавиш для этого браузера. Нажмите на сочетание, чтобы изменить его.',
        device: 'Сочетания клавиш для этого компьютера. Нажмите на сочетание, чтобы изменить его.'
      }
    },
    savedHere: { browser: 'Сохранено в этом браузере', device: 'Сохранено на этом компьютере' },
    archiveChats: {
      period: 'Архивировать через',
      off: 'Никогда',
      afterDays: days => `Через ${days} дн.`,
      archived: 'Архивные чаты',
      archivedDesc: 'Верните чаты из архива или удалите их навсегда.',
      view: 'Открыть архив',
      hide: 'Скрыть архив'
    },
    intro: {
      account: 'Ваш аккаунт Nous: тариф, кредиты и оплата. Одинаков на всех устройствах, где вы вошли.',
      gateways: 'Где работают ваши агенты и какие компьютеры доступны Hermes. Агенты обнаруживаются на подключённых вами компьютерах.',
      sessions: 'История чатов и место, где начинаются новые чаты, для всех агентов.',
      about: 'Версии этого приложения и ваших компьютеров.'
    },
    gateway: {
      connectionMode: 'Способ подключения',
      thisHost: 'Этот компьютер',
      saved: 'Сохранённые подключения',
      savedDesc: 'Все компьютеры с Hermes, доступные этому приложению.',
      manage: 'Управлять',
      advancedDesc: 'Удалённые обновления, запуск и диагностика'
    },
    about: {
      whatsNew: 'Что нового',
      checkForUpdates: 'Проверить обновления',
      gateway: 'Компьютер',
      detailsDesc: 'Сборка, среда выполнения и установка'
    },
    keybinds: {
      bindings: 'Сочетания клавиш',
      all: 'Все сочетания',
      allDesc: 'Полный редактируемый список сочетаний клавиш.',
      reset: 'Сбросить сочетания по умолчанию'
    },
    notifications: {
      advancedDesc: 'Редкие уведомления',
      browserAlerts: 'Уведомления браузера',
      completionSound: 'Звук завершения'
    },
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
    },
    configuration: {
      title: 'Конфигурация',
      exportDesc: 'Скачать настройки этого агента в файле JSON.',
      importDesc: 'Заменить настройки ранее экспортированным файлом JSON.',
      resetDesc: 'Вернуть все настройки Hermes по умолчанию. Это нельзя отменить.',
      export: 'Экспорт',
      import: 'Импорт',
      reset: 'Сбросить…'
    }
  },
  ar: {
    archive: 'الأرشيف',
    newSessions: 'المحادثات الجديدة',
    version: 'الإصدار',
    hostIntro: {
      notifications: {
        browser: 'تنبيهات من هذا المتصفح عندما يحتاجك Hermes والتبويب في الخلفية.',
        device: 'تنبيهات من هذا الحاسوب عندما يحتاجك Hermes والتطبيق في الخلفية.'
      },
      keybinds: {
        browser: 'اختصارات لوحة المفاتيح لهذا المتصفح. انقر على اختصار لتغييره.',
        device: 'اختصارات لوحة المفاتيح لهذا الحاسوب. انقر على اختصار لتغييره.'
      }
    },
    savedHere: { browser: 'محفوظ في هذا المتصفح', device: 'محفوظ على هذا الحاسوب' },
    archiveChats: {
      period: 'الأرشفة بعد',
      off: 'أبدًا',
      afterDays: days => (days === 1 ? 'بعد يوم واحد' : `بعد ${days} أيام`),
      archived: 'المحادثات المؤرشفة',
      archivedDesc: 'ألغِ أرشفة المحادثات أو احذفها نهائيًا.',
      view: 'عرض الأرشيف',
      hide: 'إخفاء الأرشيف'
    },
    intro: {
      account: 'حسابك في Nous: الخطة والأرصدة والدفع. هو نفسه على كل جهاز تسجّل الدخول منه.',
      gateways: 'أين تعمل وكلاؤك، والأجهزة التي يصل إليها Hermes. يُكتشف الوكلاء من الأجهزة التي توصلها.',
      sessions: 'سجل المحادثات وأين تبدأ المحادثات الجديدة، لكل الوكلاء.',
      about: 'تفاصيل إصدار هذا التطبيق وأجهزتك.'
    },
    gateway: {
      connectionMode: 'طريقة الاتصال',
      thisHost: 'هذا الجهاز',
      saved: 'الاتصالات المحفوظة',
      savedDesc: 'كل جهاز يشغّل Hermes يمكن لهذا التطبيق الوصول إليه.',
      manage: 'إدارة',
      advancedDesc: 'التحديثات عن بُعد وبدء التشغيل والتشخيص'
    },
    about: {
      whatsNew: 'الجديد',
      checkForUpdates: 'التحقق من التحديثات',
      gateway: 'الجهاز',
      detailsDesc: 'تفاصيل البناء وبيئة التشغيل والتثبيت'
    },
    keybinds: {
      bindings: 'اختصارات المفاتيح',
      all: 'كل الاختصارات',
      allDesc: 'القائمة الكاملة القابلة للتعديل لاختصارات المفاتيح.',
      reset: 'إعادة الاختصارات إلى الافتراضي'
    },
    notifications: {
      advancedDesc: 'تنبيهات أقل شيوعًا',
      browserAlerts: 'إشعارات المتصفح',
      completionSound: 'صوت الإكمال'
    },
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
    },
    configuration: {
      title: 'الإعدادات',
      exportDesc: 'نزّل إعدادات هذا الوكيل كملف JSON.',
      importDesc: 'استبدل الإعدادات بملف JSON صُدّر سابقًا.',
      resetDesc: 'أعد كل الإعدادات إلى قيم Hermes الافتراضية. لا يمكن التراجع عن ذلك.',
      export: 'تصدير',
      import: 'استيراد',
      reset: 'إعادة الضبط…'
    }
  }
}

export function settingsPagesCopy(locale: string): SettingsPagesCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useSettingsPagesCopy(): SettingsPagesCopy {
  return settingsPagesCopy(useI18n().locale)
}
