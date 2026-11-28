import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

/** Agent › Settings › Tools / Skills / Connectors & MCP / Plugins (Figma 895:13886, 895:14967,
 *  895:15892, 895:16810): the page summary, the catalog banner and each section's one-line note. */
export interface AgentCapabilityCopy {
  showAll: (count: number) => string
  moreTools: (count: number) => string
  less: string
  notAdded: string
  notAddedDesc: string
  tools: {
    summary: (toolsets: number, tools: number) => string
    bannerTitle: string
    bannerBody: string
    onDesc: string
    offDesc: (shown: number, total: number) => string
    /** Tools › Advanced: keys no toolset on this agent calls. */
    otherKeys: string
  }
  skills: {
    summary: (count: number) => string
    bannerTitle: string
    /** `count` arrives formatted for the locale. */
    bannerBody: (count: string) => string
    browse: string
    builtIn: string
    builtInDesc: (shown: number, total: number) => string
    added: string
    addedDesc: string
  }
  connectors: {
    bannerTitle: string
    bannerBody: string
    browse: string
    connected: string
    connectedDesc: string
    servers: string
    serversDesc: string
    addServer: string
  }
  plugins: {
    bannerTitle: string
    bannerBody: (count: string) => string
    browse: string
    onAgent: string
    onAgentDesc: string
    inApp: string
    inAppDesc: string
  }
}

const COPY: Record<BundledLocale, AgentCapabilityCopy> = {
  en: {
    showAll: count => `Show all ${count}`,
    moreTools: count => `+${count} more`,
    less: 'Less',
    notAdded: 'Not added to this agent',
    notAddedDesc: 'In the catalog, not on this agent yet.',
    tools: {
      summary: (toolsets, tools) =>
        `${toolsets} ${toolsets === 1 ? 'toolset' : 'toolsets'} on · ${tools} ${tools === 1 ? 'tool' : 'tools'}. Turn off any this agent shouldn’t use.`,
      bannerTitle: 'More toolsets in the catalog',
      bannerBody: 'Search, media, desktop control and more. All built in.',
      onDesc: 'A toolset switches all of its tools on or off together.',
      offDesc: (shown, total) =>
        shown < total
          ? `Showing ${shown} of ${total}. Turn one on to give this agent its tools.`
          : 'Turn one on to give this agent its tools.',
      otherKeys: 'Other'
    },
    skills: {
      summary: count =>
        `${count} ${count === 1 ? 'skill' : 'skills'} on for this agent. Turn off any it shouldn’t use.`,
      bannerTitle: 'Give this agent new expertise',
      bannerBody: count => `${count} ready-made skills, one click to add.`,
      browse: 'Browse the skill catalog',
      builtIn: 'Built in',
      builtInDesc: (shown, total) =>
        shown < total ? `Ship with every agent. Showing ${shown} of ${total}.` : 'Ship with every agent.',
      added: 'Added from the catalog',
      addedDesc: 'Installed for this agent only. Removing one doesn’t affect other agents.'
    },
    connectors: {
      bannerTitle: 'Bring your apps into chat',
      bannerBody: 'Sign in once, use it in every agent.',
      browse: 'Connect apps & add MCP servers',
      connected: 'Connected apps',
      connectedDesc: 'Signed in once for every agent. Turn off any this agent shouldn’t use.',
      servers: 'MCP servers',
      serversDesc: 'Installed for this agent only, with its own server address and arguments.',
      addServer: 'Add MCP server'
    },
    plugins: {
      bannerTitle: 'Extend Hermes with plugins',
      bannerBody: count => `${count} plugins from Nous and the community.`,
      browse: 'Browse the plugin catalog',
      onAgent: 'On this agent',
      onAgentDesc: 'This agent’s copy of each plugin. Removing one here doesn’t affect other agents.',
      inApp: 'Runs in the app',
      inAppDesc: 'Desktop plugins work for every agent at once, so there’s nothing to change here.'
    }
  },
  fr: {
    showAll: count => `Tout afficher (${count})`,
    moreTools: count => `+${count} de plus`,
    less: 'Moins',
    notAdded: 'Non ajoutés à cet agent',
    notAddedDesc: 'Dans le catalogue, pas encore sur cet agent.',
    tools: {
      summary: (toolsets, tools) =>
        `${toolsets} ensemble(s) d’outils actif(s) · ${tools} outil(s). Désactivez ceux que cet agent ne doit pas utiliser.`,
      bannerTitle: 'Plus d’outils dans le catalogue',
      bannerBody: 'Recherche, médias, contrôle du bureau et plus. Tout est intégré.',
      onDesc: 'Un ensemble active ou désactive tous ses outils à la fois.',
      offDesc: (shown, total) =>
        shown < total
          ? `${shown} sur ${total} affichés. Activez-en un pour donner ses outils à cet agent.`
          : 'Activez-en un pour donner ses outils à cet agent.',
      otherKeys: 'Autres'
    },
    skills: {
      summary: count =>
        `${count} compétence(s) active(s) pour cet agent. Désactivez celles qu’il ne doit pas utiliser.`,
      bannerTitle: 'Donnez de nouvelles compétences à cet agent',
      bannerBody: count => `${count} compétences prêtes à l’emploi, à ajouter en un clic.`,
      browse: 'Parcourir les compétences',
      builtIn: 'Intégrées',
      builtInDesc: (shown, total) =>
        shown < total ? `Fournies avec chaque agent. ${shown} sur ${total} affichées.` : 'Fournies avec chaque agent.',
      added: 'Ajoutées depuis le catalogue',
      addedDesc: 'Installées pour cet agent seulement. En retirer une n’affecte pas les autres agents.'
    },
    connectors: {
      bannerTitle: 'Amenez vos apps dans la discussion',
      bannerBody: 'Connectez-vous une fois, utilisez-les avec chaque agent.',
      browse: 'Connecter des apps et serveurs MCP',
      connected: 'Apps connectées',
      connectedDesc: 'Connexion partagée par tous les agents. Désactivez celles que cet agent ne doit pas utiliser.',
      servers: 'Serveurs MCP',
      serversDesc: 'Installés pour cet agent seulement, avec leur propre adresse et leurs arguments.',
      addServer: 'Ajouter un serveur MCP'
    },
    plugins: {
      bannerTitle: 'Étendez Hermes avec des plugins',
      bannerBody: count => `${count} plugins de Nous et de la communauté.`,
      browse: 'Parcourir les plugins',
      onAgent: 'Sur cet agent',
      onAgentDesc: 'La copie de chaque plugin pour cet agent. En retirer un n’affecte pas les autres agents.',
      inApp: 'Dans l’application',
      inAppDesc: 'Les plugins de bureau valent pour tous les agents, il n’y a donc rien à changer ici.'
    }
  },
  de: {
    showAll: count => `Alle ${count} anzeigen`,
    moreTools: count => `+${count} weitere`,
    less: 'Weniger',
    notAdded: 'Nicht bei diesem Agenten',
    notAddedDesc: 'Im Katalog, aber noch nicht bei diesem Agenten.',
    tools: {
      summary: (toolsets, tools) =>
        `${toolsets} Toolsets an · ${tools} Tools. Schalte ab, was dieser Agent nicht nutzen soll.`,
      bannerTitle: 'Weitere Toolsets im Katalog',
      bannerBody: 'Suche, Medien, Desktopsteuerung und mehr. Alles eingebaut.',
      onDesc: 'Ein Toolset schaltet alle seine Tools gemeinsam an oder aus.',
      offDesc: (shown, total) =>
        shown < total
          ? `${shown} von ${total} angezeigt. Schalte eins an, um diesem Agenten seine Tools zu geben.`
          : 'Schalte eins an, um diesem Agenten seine Tools zu geben.',
      otherKeys: 'Sonstige'
    },
    skills: {
      summary: count => `${count} Skills für diesen Agenten an. Schalte ab, was er nicht nutzen soll.`,
      bannerTitle: 'Gib diesem Agenten neues Fachwissen',
      bannerBody: count => `${count} fertige Skills, mit einem Klick hinzugefügt.`,
      browse: 'Skill-Katalog durchsuchen',
      builtIn: 'Eingebaut',
      builtInDesc: (shown, total) =>
        shown < total ? `Bei jedem Agenten dabei. ${shown} von ${total} angezeigt.` : 'Bei jedem Agenten dabei.',
      added: 'Aus dem Katalog hinzugefügt',
      addedDesc: 'Nur für diesen Agenten installiert. Entfernen betrifft keine anderen Agenten.'
    },
    connectors: {
      bannerTitle: 'Hol deine Apps in den Chat',
      bannerBody: 'Einmal anmelden, in jedem Agenten nutzen.',
      browse: 'Apps verbinden & MCP-Server hinzufügen',
      connected: 'Verbundene Apps',
      connectedDesc: 'Einmal für alle Agenten angemeldet. Schalte ab, was dieser Agent nicht nutzen soll.',
      servers: 'MCP-Server',
      serversDesc: 'Nur für diesen Agenten installiert, mit eigener Serveradresse und Argumenten.',
      addServer: 'MCP-Server hinzufügen'
    },
    plugins: {
      bannerTitle: 'Erweitere Hermes mit Plugins',
      bannerBody: count => `${count} Plugins von Nous und der Community.`,
      browse: 'Plugin-Katalog durchsuchen',
      onAgent: 'Bei diesem Agenten',
      onAgentDesc: 'Die Kopie jedes Plugins für diesen Agenten. Entfernen betrifft keine anderen Agenten.',
      inApp: 'Läuft in der App',
      inAppDesc: 'Desktop-Plugins gelten für alle Agenten zugleich, hier gibt es nichts zu ändern.'
    }
  },
  es: {
    showAll: count => `Mostrar los ${count}`,
    moreTools: count => `+${count} más`,
    less: 'Menos',
    notAdded: 'No añadidos a este agente',
    notAddedDesc: 'En el catálogo, todavía no en este agente.',
    tools: {
      summary: (toolsets, tools) =>
        `${toolsets} conjuntos activos · ${tools} herramientas. Desactiva las que este agente no deba usar.`,
      bannerTitle: 'Más herramientas en el catálogo',
      bannerBody: 'Búsqueda, multimedia, control del escritorio y más. Todo integrado.',
      onDesc: 'Un conjunto activa o desactiva todas sus herramientas a la vez.',
      offDesc: (shown, total) =>
        shown < total
          ? `Mostrando ${shown} de ${total}. Activa uno para dar sus herramientas a este agente.`
          : 'Activa uno para dar sus herramientas a este agente.',
      otherKeys: 'Otras'
    },
    skills: {
      summary: count => `${count} habilidades activas para este agente. Desactiva las que no deba usar.`,
      bannerTitle: 'Dale nuevas habilidades a este agente',
      bannerBody: count => `${count} habilidades listas, se añaden con un clic.`,
      browse: 'Explorar habilidades',
      builtIn: 'Integradas',
      builtInDesc: (shown, total) =>
        shown < total ? `Vienen con cada agente. Mostrando ${shown} de ${total}.` : 'Vienen con cada agente.',
      added: 'Añadidas desde el catálogo',
      addedDesc: 'Instaladas solo para este agente. Quitar una no afecta a otros agentes.'
    },
    connectors: {
      bannerTitle: 'Trae tus apps al chat',
      bannerBody: 'Inicia sesión una vez y úsalas con cada agente.',
      browse: 'Conectar apps y añadir servidores MCP',
      connected: 'Apps conectadas',
      connectedDesc: 'Sesión compartida por todos los agentes. Desactiva las que este agente no deba usar.',
      servers: 'Servidores MCP',
      serversDesc: 'Instalados solo para este agente, con su propia dirección y argumentos.',
      addServer: 'Añadir servidor MCP'
    },
    plugins: {
      bannerTitle: 'Amplía Hermes con plugins',
      bannerBody: count => `${count} plugins de Nous y la comunidad.`,
      browse: 'Explorar plugins',
      onAgent: 'En este agente',
      onAgentDesc: 'La copia de cada plugin para este agente. Quitar uno no afecta a otros agentes.',
      inApp: 'Se ejecutan en la app',
      inAppDesc: 'Los plugins de escritorio valen para todos los agentes, así que aquí no hay nada que cambiar.'
    }
  },
  ja: {
    showAll: count => `すべて表示（${count}）`,
    moreTools: count => `+${count}件`,
    less: '閉じる',
    notAdded: 'このエージェントに未追加',
    notAddedDesc: 'カタログにありますが、このエージェントにはまだありません。',
    tools: {
      summary: (toolsets, tools) =>
        `${toolsets}個のツールセットがオン・ツール${tools}個。このエージェントに不要なものはオフにします。`,
      bannerTitle: 'カタログにはさらにツールセットがあります',
      bannerBody: '検索、メディア、デスクトップ操作など。すべて組み込み済みです。',
      onDesc: 'ツールセットは含まれるツールをまとめてオン・オフします。',
      offDesc: (shown, total) =>
        shown < total
          ? `${total}件中${shown}件を表示。オンにするとこのエージェントが使えるようになります。`
          : 'オンにするとこのエージェントが使えるようになります。',
      otherKeys: 'その他'
    },
    skills: {
      summary: count => `このエージェントでオンのスキルは${count}個です。不要なものはオフにします。`,
      bannerTitle: 'このエージェントに新しい専門知識を',
      bannerBody: count => `${count}個の既製スキルをワンクリックで追加できます。`,
      browse: 'スキルカタログを見る',
      builtIn: '組み込み',
      builtInDesc: (shown, total) =>
        shown < total ? `すべてのエージェントに付属。${total}件中${shown}件を表示。` : 'すべてのエージェントに付属。',
      added: 'カタログから追加',
      addedDesc: 'このエージェントにだけインストールされています。削除しても他のエージェントには影響しません。'
    },
    connectors: {
      bannerTitle: 'アプリをチャットに',
      bannerBody: '一度サインインすれば、どのエージェントでも使えます。',
      browse: 'アプリを接続・MCP サーバーを追加',
      connected: '接続済みのアプリ',
      connectedDesc: 'サインインはすべてのエージェントで共有されます。不要なものはオフにします。',
      servers: 'MCP サーバー',
      serversDesc: 'このエージェントにだけインストールされ、独自のアドレスと引数を持ちます。',
      addServer: 'MCP サーバーを追加'
    },
    plugins: {
      bannerTitle: 'プラグインで Hermes を拡張',
      bannerBody: count => `Nous とコミュニティによる${count}個のプラグイン。`,
      browse: 'プラグインカタログを見る',
      onAgent: 'このエージェント',
      onAgentDesc: '各プラグインのこのエージェント用コピー。削除しても他のエージェントには影響しません。',
      inApp: 'アプリで動作',
      inAppDesc: 'デスクトッププラグインはすべてのエージェントに共通のため、ここで変更するものはありません。'
    }
  },
  zh: {
    showAll: count => `显示全部 ${count} 项`,
    moreTools: count => `+${count} 个`,
    less: '收起',
    notAdded: '未添加到此智能体',
    notAddedDesc: '在目录中，但尚未添加到此智能体。',
    tools: {
      summary: (toolsets, tools) => `已开启 ${toolsets} 个工具集 · ${tools} 个工具。关闭此智能体不该使用的工具。`,
      bannerTitle: '目录中还有更多工具集',
      bannerBody: '搜索、媒体、桌面控制等，全部内置。',
      onDesc: '工具集会一起开启或关闭其中的所有工具。',
      offDesc: (shown, total) =>
        shown < total
          ? `显示 ${total} 个中的 ${shown} 个。开启一个即可让此智能体使用其工具。`
          : '开启一个即可让此智能体使用其工具。',
      otherKeys: '其他'
    },
    skills: {
      summary: count => `此智能体已开启 ${count} 个技能。关闭它不该使用的技能。`,
      bannerTitle: '为此智能体增添新专长',
      bannerBody: count => `${count} 个现成技能，一键添加。`,
      browse: '浏览技能目录',
      builtIn: '内置',
      builtInDesc: (shown, total) =>
        shown < total ? `每个智能体都自带。显示 ${total} 个中的 ${shown} 个。` : '每个智能体都自带。',
      added: '从目录添加',
      addedDesc: '仅为此智能体安装。移除不会影响其他智能体。'
    },
    connectors: {
      bannerTitle: '把你的应用带进对话',
      bannerBody: '登录一次，所有智能体都能使用。',
      browse: '连接应用并添加 MCP 服务器',
      connected: '已连接的应用',
      connectedDesc: '登录由所有智能体共享。关闭此智能体不该使用的应用。',
      servers: 'MCP 服务器',
      serversDesc: '仅为此智能体安装，使用自己的服务器地址和参数。',
      addServer: '添加 MCP 服务器'
    },
    plugins: {
      bannerTitle: '用插件扩展 Hermes',
      bannerBody: count => `来自 Nous 和社区的 ${count} 个插件。`,
      browse: '浏览插件目录',
      onAgent: '此智能体上的插件',
      onAgentDesc: '每个插件在此智能体上的副本。在这里移除不会影响其他智能体。',
      inApp: '在应用中运行',
      inAppDesc: '桌面插件同时作用于所有智能体，因此这里无需更改。'
    }
  },
  'zh-hant': {
    showAll: count => `顯示全部 ${count} 項`,
    moreTools: count => `+${count} 個`,
    less: '收合',
    notAdded: '未加入此智慧體',
    notAddedDesc: '在目錄中，但尚未加入此智慧體。',
    tools: {
      summary: (toolsets, tools) => `已開啟 ${toolsets} 個工具集 · ${tools} 個工具。關閉此智慧體不該使用的工具。`,
      bannerTitle: '目錄中還有更多工具集',
      bannerBody: '搜尋、媒體、桌面控制等，全部內建。',
      onDesc: '工具集會一起開啟或關閉其中的所有工具。',
      offDesc: (shown, total) =>
        shown < total
          ? `顯示 ${total} 個中的 ${shown} 個。開啟一個即可讓此智慧體使用其工具。`
          : '開啟一個即可讓此智慧體使用其工具。',
      otherKeys: '其他'
    },
    skills: {
      summary: count => `此智慧體已開啟 ${count} 個技能。關閉它不該使用的技能。`,
      bannerTitle: '為此智慧體增添新專長',
      bannerBody: count => `${count} 個現成技能，一鍵加入。`,
      browse: '瀏覽技能目錄',
      builtIn: '內建',
      builtInDesc: (shown, total) =>
        shown < total ? `每個智慧體都內附。顯示 ${total} 個中的 ${shown} 個。` : '每個智慧體都內附。',
      added: '從目錄加入',
      addedDesc: '僅為此智慧體安裝。移除不會影響其他智慧體。'
    },
    connectors: {
      bannerTitle: '把你的應用程式帶進對話',
      bannerBody: '登入一次，所有智慧體都能使用。',
      browse: '連接應用程式並新增 MCP 伺服器',
      connected: '已連接的應用程式',
      connectedDesc: '登入由所有智慧體共用。關閉此智慧體不該使用的應用程式。',
      servers: 'MCP 伺服器',
      serversDesc: '僅為此智慧體安裝，使用自己的伺服器位址和參數。',
      addServer: '新增 MCP 伺服器'
    },
    plugins: {
      bannerTitle: '用外掛擴充 Hermes',
      bannerBody: count => `來自 Nous 和社群的 ${count} 個外掛。`,
      browse: '瀏覽外掛目錄',
      onAgent: '此智慧體上的外掛',
      onAgentDesc: '每個外掛在此智慧體上的副本。在這裡移除不會影響其他智慧體。',
      inApp: '在應用程式中執行',
      inAppDesc: '桌面外掛同時作用於所有智慧體，因此這裡無需變更。'
    }
  },
  ru: {
    showAll: count => `Показать все (${count})`,
    moreTools: count => `ещё ${count}`,
    less: 'Свернуть',
    notAdded: 'Не добавлены этому агенту',
    notAddedDesc: 'Есть в каталоге, но ещё не у этого агента.',
    tools: {
      summary: (toolsets, tools) =>
        `Наборов включено: ${toolsets} · инструментов: ${tools}. Выключите те, что не нужны этому агенту.`,
      bannerTitle: 'Больше наборов в каталоге',
      bannerBody: 'Поиск, медиа, управление рабочим столом и другое. Всё встроено.',
      onDesc: 'Набор включает или выключает все свои инструменты сразу.',
      offDesc: (shown, total) =>
        shown < total
          ? `Показано ${shown} из ${total}. Включите набор, чтобы дать агенту его инструменты.`
          : 'Включите набор, чтобы дать агенту его инструменты.',
      otherKeys: 'Прочее'
    },
    skills: {
      summary: count => `Включено навыков для этого агента: ${count}. Выключите те, что ему не нужны.`,
      bannerTitle: 'Дайте агенту новые знания',
      bannerBody: count => `Готовых навыков: ${count}, добавляются в один клик.`,
      browse: 'Открыть каталог навыков',
      builtIn: 'Встроенные',
      builtInDesc: (shown, total) =>
        shown < total ? `Есть у каждого агента. Показано ${shown} из ${total}.` : 'Есть у каждого агента.',
      added: 'Добавлены из каталога',
      addedDesc: 'Установлены только для этого агента. Удаление не затронет других агентов.'
    },
    connectors: {
      bannerTitle: 'Подключите приложения к чату',
      bannerBody: 'Войдите один раз и пользуйтесь в любом агенте.',
      browse: 'Подключить приложения и MCP-серверы',
      connected: 'Подключённые приложения',
      connectedDesc: 'Вход общий для всех агентов. Выключите те, что не нужны этому агенту.',
      servers: 'MCP-серверы',
      serversDesc: 'Установлены только для этого агента, со своим адресом и аргументами.',
      addServer: 'Добавить MCP-сервер'
    },
    plugins: {
      bannerTitle: 'Расширьте Hermes плагинами',
      bannerBody: count => `Плагинов от Nous и сообщества: ${count}.`,
      browse: 'Открыть каталог плагинов',
      onAgent: 'У этого агента',
      onAgentDesc: 'Копия каждого плагина для этого агента. Удаление здесь не затронет других агентов.',
      inApp: 'Работают в приложении',
      inAppDesc: 'Плагины рабочего стола действуют для всех агентов сразу, здесь нечего менять.'
    }
  },
  ar: {
    showAll: count => `عرض الكل (${count})`,
    moreTools: count => `+${count} أخرى`,
    less: 'أقل',
    notAdded: 'غير مضافة لهذا الوكيل',
    notAddedDesc: 'موجودة في الكتالوج، وليست لدى هذا الوكيل بعد.',
    tools: {
      summary: (toolsets, tools) =>
        `${toolsets} مجموعات أدوات مفعّلة · ${tools} أدوات. أوقف ما لا ينبغي لهذا الوكيل استخدامه.`,
      bannerTitle: 'مجموعات أدوات أخرى في الكتالوج',
      bannerBody: 'البحث والوسائط والتحكم في سطح المكتب وغيرها. كلها مدمجة.',
      onDesc: 'تشغّل المجموعة كل أدواتها أو توقفها معًا.',
      offDesc: (shown, total) =>
        shown < total
          ? `عرض ${shown} من ${total}. فعّل واحدة لتمنح هذا الوكيل أدواتها.`
          : 'فعّل واحدة لتمنح هذا الوكيل أدواتها.',
      otherKeys: 'أخرى'
    },
    skills: {
      summary: count => `${count} مهارات مفعّلة لهذا الوكيل. أوقف ما لا ينبغي له استخدامه.`,
      bannerTitle: 'امنح هذا الوكيل خبرة جديدة',
      bannerBody: count => `${count} مهارة جاهزة، تُضاف بنقرة واحدة.`,
      browse: 'تصفح كتالوج المهارات',
      builtIn: 'مدمجة',
      builtInDesc: (shown, total) =>
        shown < total ? `تأتي مع كل وكيل. عرض ${shown} من ${total}.` : 'تأتي مع كل وكيل.',
      added: 'مضافة من الكتالوج',
      addedDesc: 'مثبتة لهذا الوكيل فقط. إزالة إحداها لا تؤثر في الوكلاء الآخرين.'
    },
    connectors: {
      bannerTitle: 'أحضر تطبيقاتك إلى المحادثة',
      bannerBody: 'سجّل الدخول مرة واحدة، واستخدمها مع كل وكيل.',
      browse: 'ربط التطبيقات وإضافة خوادم MCP',
      connected: 'التطبيقات المتصلة',
      connectedDesc: 'تسجيل الدخول مشترك بين كل الوكلاء. أوقف ما لا ينبغي لهذا الوكيل استخدامه.',
      servers: 'خوادم MCP',
      serversDesc: 'مثبتة لهذا الوكيل فقط، بعنوان خادم ووسائط خاصة بها.',
      addServer: 'إضافة خادم MCP'
    },
    plugins: {
      bannerTitle: 'وسّع Hermes بالإضافات',
      bannerBody: count => `${count} إضافة من Nous والمجتمع.`,
      browse: 'تصفح كتالوج الإضافات',
      onAgent: 'لدى هذا الوكيل',
      onAgentDesc: 'نسخة هذا الوكيل من كل إضافة. إزالة إحداها هنا لا تؤثر في الوكلاء الآخرين.',
      inApp: 'تعمل في التطبيق',
      inAppDesc: 'تعمل إضافات سطح المكتب لكل الوكلاء معًا، فلا شيء لتغييره هنا.'
    }
  }
}

export function agentCapabilityCopy(locale: string): AgentCapabilityCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useAgentCapabilityCopy(): AgentCapabilityCopy {
  return agentCapabilityCopy(useI18n().locale)
}
