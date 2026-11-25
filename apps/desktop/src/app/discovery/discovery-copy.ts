import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'
import type { OldNameAlias } from '@/lib/old-name-aliases'

/** The redesign's discovery surfaces (notifications tray, rename splash, new-version notice, the
 *  palette's old-word rows). Only the redesign renders them. The new nouns (Chats, Agents, ...)
 *  are the rail's own labels, read from the catalog where the rail reads them; this module holds
 *  only the classic words they replace and the surfaces' own sentences. */
export interface DiscoveryCopy {
  /** The words the classic interface used, as it showed them: the rename splash names them, and
   *  ⌘K matches them in this language. */
  classicNames: Record<OldNameAlias['code'], string>
  tray: {
    /** The bell's accessible name. */
    title: string
    /** The tray's own heading. */
    heading: string
    /** The bell's accessible description while something in the tray is new. */
    unseen: string
    empty: string
    newBadge: string
    /** Clears the New marks the tray opened with. */
    markAllRead: string
    /** A row's ×: takes it out of the tray until it has news again. */
    dismiss: string
    footnote: string
    /** Reopens the welcome screen (the rename splash, or the first-time one for a newcomer). */
    openIntroduction: string
  }
  renameItem: { title: string; description: string; action: string }
  updateItem: { title: string; description: string; action: string }
  newVersion: {
    eyebrow: string
    title: string
    description: string
    notice: string
    refresh: string
    /** Puts the notice in the chat away; the tray keeps it. */
    later: string
  }
  renameSplash: {
    eyebrow: string
    title: string
    subtitle: string
    /** The rows' list, for assistive tech. */
    listLabel: string
    /** The row's accessible name, as its arrow shows it. */
    renamed: (classic: string, current: string) => string
    lines: Record<'agents' | 'chats' | 'library' | 'machines' | 'routines', string>
    /** A count with more beyond the loaded page ("128+"). */
    atLeast: (count: string) => string
    footnote: string
    done: string
  }
  /** A ⌘K row's second line for a renamed place, found by its classic word ("Formerly Sessions"). */
  formerlyLine: (classic: string) => string
}

const en: DiscoveryCopy = {
  classicNames: {
    sessions: 'Sessions',
    profiles: 'Profiles',
    artifacts: 'Artifacts',
    cron: 'Cron jobs',
    gateways: 'Gateways'
  },
  tray: {
    title: 'Notifications',
    heading: 'Notifications & updates',
    unseen: 'Something new',
    empty: 'Nothing new right now.',
    newBadge: 'New',
    markAllRead: 'Mark all read',
    dismiss: 'Dismiss',
    footnote: 'Updates stay here until you clear them.',
    openIntroduction: 'Open introduction'
  },
  renameItem: {
    title: 'Hermes got a new look',
    description: 'See the new names and where your work lives',
    action: 'Open'
  },
  updateItem: { title: 'Update available', description: 'Install when you’re ready', action: 'Update' },
  newVersion: {
    eyebrow: 'Update ready',
    title: 'A new version of Hermes is ready',
    description: 'It loads when you start a new chat. Nothing you’re doing is lost.',
    notice: 'A new Hermes is ready. It loads when you start a new chat.',
    refresh: 'Refresh now',
    later: 'Later'
  },
  renameSplash: {
    eyebrow: 'Welcome back / What moved',
    title: 'Hermes got a new look',
    subtitle: 'New names, familiar concepts. Select a row to find its new home.',
    listLabel: 'What was renamed',
    renamed: (classic, current) => `${classic}, now ${current}`,
    lines: {
      chats: 'Each chat sits under the agent that ran it',
      agents: 'Your experts, with their settings and memory',
      library: 'Everything your agents make',
      routines: 'Tasks that keep running on a schedule',
      machines: 'Where your agents run'
    },
    atLeast: count => `${count}+`,
    footnote: 'Click a navigation item or press Esc to close. Reopen from Notifications.',
    done: 'Let’s get to work'
  },
  formerlyLine: classic => `Formerly ${classic}`
}

const fr: DiscoveryCopy = {
  classicNames: {
    sessions: 'Sessions',
    profiles: 'Profils',
    artifacts: 'Artefacts',
    cron: 'Tâches cron',
    gateways: 'Gateways'
  },
  tray: {
    title: 'Notifications',
    heading: 'Notifications et mises à jour',
    unseen: 'Du nouveau',
    empty: 'Rien de nouveau pour l’instant.',
    newBadge: 'Nouveau',
    markAllRead: 'Tout marquer comme lu',
    dismiss: 'Ignorer',
    footnote: 'Les mises à jour restent ici jusqu’à ce que vous les effaciez.',
    openIntroduction: 'Ouvrir la présentation'
  },
  renameItem: {
    title: 'Hermes fait peau neuve',
    description: 'Voyez les nouveaux noms et où se trouve votre travail',
    action: 'Ouvrir'
  },
  updateItem: { title: 'Mise à jour disponible', description: 'Installez quand vous voulez', action: 'Mettre à jour' },
  newVersion: {
    eyebrow: 'Mise à jour prête',
    title: 'Une nouvelle version de Hermes est prête',
    description: 'Elle se charge quand vous lancez une nouvelle discussion. Rien de ce que vous faites n’est perdu.',
    notice: 'Une nouvelle version de Hermes est prête. Elle se charge quand vous lancez une nouvelle discussion.',
    refresh: 'Actualiser',
    later: 'Plus tard'
  },
  renameSplash: {
    eyebrow: 'Bon retour / Ce qui a changé',
    title: 'Hermes fait peau neuve',
    subtitle: 'Nouveaux noms, concepts familiers. Choisissez une ligne pour trouver sa nouvelle place.',
    listLabel: 'Ce qui a été renommé',
    renamed: (classic, current) => `${classic}, désormais ${current}`,
    lines: {
      chats: 'Chaque discussion se range sous l’agent qui l’a menée',
      agents: 'Vos experts, avec leurs réglages et leur mémoire',
      library: 'Tout ce que créent vos agents',
      routines: 'Des tâches qui tournent selon un planning',
      machines: 'Là où tournent vos agents'
    },
    atLeast: count => `${count}+`,
    footnote: 'Cliquez dans la navigation ou appuyez sur Échap pour fermer. Rouvrez depuis les notifications.',
    done: 'Au travail'
  },
  formerlyLine: classic => `Anciennement ${classic}`
}

const de: DiscoveryCopy = {
  classicNames: {
    sessions: 'Sessions',
    profiles: 'Profile',
    artifacts: 'Artefakte',
    cron: 'Cron-Jobs',
    gateways: 'Gateways'
  },
  tray: {
    title: 'Benachrichtigungen',
    heading: 'Benachrichtigungen & Updates',
    unseen: 'Etwas Neues',
    empty: 'Gerade nichts Neues.',
    newBadge: 'Neu',
    markAllRead: 'Alle als gelesen markieren',
    dismiss: 'Ausblenden',
    footnote: 'Updates bleiben hier, bis du sie entfernst.',
    openIntroduction: 'Einführung öffnen'
  },
  renameItem: {
    title: 'Hermes sieht jetzt neu aus',
    description: 'Sieh dir die neuen Namen an und wo deine Arbeit liegt',
    action: 'Öffnen'
  },
  updateItem: { title: 'Update verfügbar', description: 'Installier es, wann du willst', action: 'Aktualisieren' },
  newVersion: {
    eyebrow: 'Update bereit',
    title: 'Eine neue Version von Hermes ist bereit',
    description: 'Sie lädt, wenn du einen neuen Chat startest. Nichts, woran du arbeitest, geht verloren.',
    notice: 'Eine neue Version von Hermes ist bereit. Sie lädt, wenn du einen neuen Chat startest.',
    refresh: 'Jetzt neu laden',
    later: 'Später'
  },
  renameSplash: {
    eyebrow: 'Willkommen zurück / Was sich geändert hat',
    title: 'Hermes sieht jetzt neu aus',
    subtitle: 'Neue Namen, vertraute Konzepte. Wähl eine Zeile, um zu sehen, wo sie jetzt liegt.',
    listLabel: 'Was umbenannt wurde',
    renamed: (classic, current) => `${classic}, jetzt ${current}`,
    lines: {
      chats: 'Jeder Chat liegt unter dem Agenten, der ihn geführt hat',
      agents: 'Deine Experten, mit ihren Einstellungen und Erinnerungen',
      library: 'Alles, was deine Agenten erstellen',
      routines: 'Aufgaben, die nach Zeitplan weiterlaufen',
      machines: 'Wo deine Agenten laufen'
    },
    atLeast: count => `${count}+`,
    footnote: 'Klick in die Navigation oder drück Esc, um zu schließen. Unter Benachrichtigungen findest du es wieder.',
    done: 'Los geht’s'
  },
  formerlyLine: classic => `Früher ${classic}`
}

const es: DiscoveryCopy = {
  classicNames: {
    sessions: 'Sesiones',
    profiles: 'Perfiles',
    artifacts: 'Artefactos',
    cron: 'Tareas cron',
    gateways: 'Gateways'
  },
  tray: {
    title: 'Notificaciones',
    heading: 'Notificaciones y actualizaciones',
    unseen: 'Hay novedades',
    empty: 'Nada nuevo por ahora.',
    newBadge: 'Nuevo',
    markAllRead: 'Marcar todo como leído',
    dismiss: 'Descartar',
    footnote: 'Las actualizaciones se quedan aquí hasta que las borres.',
    openIntroduction: 'Abrir la presentación'
  },
  renameItem: {
    title: 'Hermes tiene un nuevo aspecto',
    description: 'Mira los nuevos nombres y dónde está tu trabajo',
    action: 'Abrir'
  },
  updateItem: { title: 'Actualización disponible', description: 'Instálala cuando quieras', action: 'Actualizar' },
  newVersion: {
    eyebrow: 'Actualización lista',
    title: 'Hay una nueva versión de Hermes lista',
    description: 'Se carga cuando empiezas un chat nuevo. No se pierde nada de lo que estás haciendo.',
    notice: 'Hay una nueva versión de Hermes lista. Se carga cuando empiezas un chat nuevo.',
    refresh: 'Actualizar ahora',
    later: 'Más tarde'
  },
  renameSplash: {
    eyebrow: 'Bienvenido de nuevo / Qué ha cambiado',
    title: 'Hermes tiene un nuevo aspecto',
    subtitle: 'Nombres nuevos, conceptos conocidos. Elige una fila para ver dónde está ahora.',
    listLabel: 'Qué cambió de nombre',
    renamed: (classic, current) => `${classic}, ahora ${current}`,
    lines: {
      chats: 'Cada chat está bajo el agente que lo llevó',
      agents: 'Tus expertos, con su configuración y su memoria',
      library: 'Todo lo que crean tus agentes',
      routines: 'Tareas que siguen funcionando según un horario',
      machines: 'Donde funcionan tus agentes'
    },
    atLeast: count => `${count}+`,
    footnote: 'Haz clic en la navegación o pulsa Esc para cerrar. Vuelve a abrirlo desde Notificaciones.',
    done: 'Manos a la obra'
  },
  formerlyLine: classic => `Antes ${classic}`
}

const ja: DiscoveryCopy = {
  classicNames: {
    sessions: 'セッション',
    profiles: 'プロファイル',
    artifacts: 'アーティファクト',
    cron: 'Cron ジョブ',
    gateways: 'ゲートウェイ'
  },
  tray: {
    title: '通知',
    heading: '通知とアップデート',
    unseen: '新着あり',
    empty: '今は新しいお知らせはありません。',
    newBadge: '新着',
    markAllRead: 'すべて既読にする',
    dismiss: '閉じる',
    footnote: 'アップデートは消去するまでここに残ります。',
    openIntroduction: '紹介を開く'
  },
  renameItem: {
    title: 'Hermes が新しくなりました',
    description: '新しい名前と、作業の置き場所を確認',
    action: '開く'
  },
  updateItem: { title: 'アップデートあり', description: '準備ができたらインストール', action: 'アップデート' },
  newVersion: {
    eyebrow: 'アップデート準備完了',
    title: 'Hermes の新しいバージョンの準備ができました',
    description: '新しいチャットを始めたときに読み込まれます。作業中の内容は失われません。',
    notice: 'Hermes の新しいバージョンの準備ができました。新しいチャットを始めると読み込まれます。',
    refresh: '今すぐ更新',
    later: 'あとで'
  },
  renameSplash: {
    eyebrow: 'おかえりなさい / 変わったところ',
    title: 'Hermes が新しくなりました',
    subtitle: '名前は新しく、考え方はそのまま。行を選ぶと新しい場所がわかります。',
    listLabel: '名前が変わったもの',
    renamed: (classic, current) => `${classic} は ${current} になりました`,
    lines: {
      chats: '各チャットは、それを実行したエージェントの下に並びます',
      agents: '設定とメモリーを持つ、あなたのエキスパート',
      library: 'エージェントが作ったものすべて',
      routines: 'スケジュールに沿って動き続けるタスク',
      machines: 'エージェントが動く場所'
    },
    atLeast: count => `${count}+`,
    footnote: 'ナビゲーションをクリックするか Esc キーで閉じます。通知からもう一度開けます。',
    done: 'さっそく始める'
  },
  formerlyLine: classic => `旧: ${classic}`
}

const zh: DiscoveryCopy = {
  classicNames: {
    sessions: '会话',
    profiles: '配置档案',
    artifacts: '产物',
    cron: 'Cron 任务',
    gateways: '网关'
  },
  tray: {
    title: '通知',
    heading: '通知与更新',
    unseen: '有新内容',
    empty: '暂时没有新内容。',
    newBadge: '新',
    markAllRead: '全部标为已读',
    dismiss: '忽略',
    footnote: '更新会一直留在这里，直到你清除。',
    openIntroduction: '打开介绍'
  },
  renameItem: {
    title: 'Hermes 换了新面貌',
    description: '看看新名字，以及你的工作在哪里',
    action: '打开'
  },
  updateItem: { title: '有可用更新', description: '准备好时再安装', action: '更新' },
  newVersion: {
    eyebrow: '更新已就绪',
    title: 'Hermes 新版本已就绪',
    description: '开始新对话时就会加载。你正在做的事不会丢失。',
    notice: 'Hermes 新版本已就绪。开始新对话时就会加载。',
    refresh: '立即刷新',
    later: '稍后'
  },
  renameSplash: {
    eyebrow: '欢迎回来 / 有哪些变化',
    title: 'Hermes 换了新面貌',
    subtitle: '名字是新的，概念还是熟悉的。选择一行，看看它的新位置。',
    listLabel: '改名的内容',
    renamed: (classic, current) => `${classic}，现在叫 ${current}`,
    lines: {
      chats: '每个对话都归在运行它的智能体下面',
      agents: '你的专家，带着各自的设置和记忆',
      library: '智能体做出的一切',
      routines: '按计划持续运行的任务',
      machines: '智能体运行的地方'
    },
    atLeast: count => `${count}+`,
    footnote: '点击导航中的项目或按 Esc 关闭。可从通知中再次打开。',
    done: '开始工作'
  },
  formerlyLine: classic => `原 ${classic}`
}

const zhHant: DiscoveryCopy = {
  classicNames: {
    sessions: '工作階段',
    profiles: '設定檔',
    artifacts: '成品',
    cron: 'Cron 工作',
    gateways: '閘道'
  },
  tray: {
    title: '通知',
    heading: '通知與更新',
    unseen: '有新內容',
    empty: '目前沒有新內容。',
    newBadge: '新',
    markAllRead: '全部標為已讀',
    dismiss: '忽略',
    footnote: '更新會一直留在這裡，直到你清除。',
    openIntroduction: '開啟介紹'
  },
  renameItem: {
    title: 'Hermes 換了新面貌',
    description: '看看新名稱，以及你的工作在哪裡',
    action: '開啟'
  },
  updateItem: { title: '有可用更新', description: '準備好時再安裝', action: '更新' },
  newVersion: {
    eyebrow: '更新已就緒',
    title: 'Hermes 新版本已就緒',
    description: '開始新對話時就會載入。你正在做的事不會遺失。',
    notice: 'Hermes 新版本已就緒。開始新對話時就會載入。',
    refresh: '立即重新整理',
    later: '稍後'
  },
  renameSplash: {
    eyebrow: '歡迎回來 / 有哪些變化',
    title: 'Hermes 換了新面貌',
    subtitle: '名稱是新的，概念還是熟悉的。選一列，看看它的新位置。',
    listLabel: '改名的內容',
    renamed: (classic, current) => `${classic}，現在叫 ${current}`,
    lines: {
      chats: '每個對話都歸在執行它的智慧體底下',
      agents: '你的專家，帶著各自的設定和記憶',
      library: '智慧體做出的一切',
      routines: '依排程持續執行的工作',
      machines: '智慧體執行的地方'
    },
    atLeast: count => `${count}+`,
    footnote: '點一下導覽列的項目或按 Esc 關閉。可從通知再次開啟。',
    done: '開始工作'
  },
  formerlyLine: classic => `原 ${classic}`
}

const ru: DiscoveryCopy = {
  classicNames: {
    sessions: 'Сеансы',
    profiles: 'Профили',
    artifacts: 'Артефакты',
    cron: 'Задачи cron',
    gateways: 'Шлюзы'
  },
  tray: {
    title: 'Уведомления',
    heading: 'Уведомления и обновления',
    unseen: 'Есть новое',
    empty: 'Пока ничего нового.',
    newBadge: 'Новое',
    markAllRead: 'Отметить всё прочитанным',
    dismiss: 'Скрыть',
    footnote: 'Обновления остаются здесь, пока вы их не уберёте.',
    openIntroduction: 'Открыть знакомство'
  },
  renameItem: {
    title: 'Hermes выглядит по-новому',
    description: 'Новые названия и где теперь ваша работа',
    action: 'Открыть'
  },
  updateItem: { title: 'Доступно обновление', description: 'Установите, когда будете готовы', action: 'Обновить' },
  newVersion: {
    eyebrow: 'Обновление готово',
    title: 'Новая версия Hermes готова',
    description: 'Она загрузится, когда вы начнёте новый чат. Ничего из того, что вы делаете, не потеряется.',
    notice: 'Новая версия Hermes готова. Она загрузится, когда вы начнёте новый чат.',
    refresh: 'Обновить сейчас',
    later: 'Позже'
  },
  renameSplash: {
    eyebrow: 'С возвращением / Что изменилось',
    title: 'Hermes выглядит по-новому',
    subtitle: 'Новые названия, знакомые понятия. Выберите строку, чтобы найти её новое место.',
    listLabel: 'Что переименовано',
    renamed: (classic, current) => `${classic}, теперь ${current}`,
    lines: {
      chats: 'Каждый чат находится под агентом, который его вёл',
      agents: 'Ваши эксперты, с их настройками и памятью',
      library: 'Всё, что создают ваши агенты',
      routines: 'Задачи, которые работают по расписанию',
      machines: 'Где работают ваши агенты'
    },
    atLeast: count => `${count}+`,
    footnote: 'Нажмите на пункт навигации или Esc, чтобы закрыть. Открыть снова можно из уведомлений.',
    done: 'За работу'
  },
  formerlyLine: classic => `Раньше: ${classic}`
}

const ar: DiscoveryCopy = {
  classicNames: {
    sessions: 'الجلسات',
    profiles: 'الملفات الشخصية',
    artifacts: 'المخرجات',
    cron: 'مهام cron',
    gateways: 'البوابات'
  },
  tray: {
    title: 'الإشعارات',
    heading: 'الإشعارات والتحديثات',
    unseen: 'يوجد جديد',
    empty: 'لا جديد الآن.',
    newBadge: 'جديد',
    markAllRead: 'تعليم الكل كمقروء',
    dismiss: 'تجاهل',
    footnote: 'تبقى التحديثات هنا حتى تمسحها.',
    openIntroduction: 'فتح التعريف'
  },
  renameItem: {
    title: 'Hermes بشكل جديد',
    description: 'اطّلع على الأسماء الجديدة وأين يوجد عملك',
    action: 'فتح'
  },
  updateItem: { title: 'تحديث متاح', description: 'ثبّته عندما تكون جاهزًا', action: 'تحديث' },
  newVersion: {
    eyebrow: 'التحديث جاهز',
    title: 'إصدار جديد من Hermes جاهز',
    description: 'يُحمَّل عندما تبدأ محادثة جديدة. لن يضيع شيء مما تعمل عليه.',
    notice: 'إصدار جديد من Hermes جاهز. يُحمَّل عندما تبدأ محادثة جديدة.',
    refresh: 'التحديث الآن',
    later: 'لاحقًا'
  },
  renameSplash: {
    eyebrow: 'مرحبًا بعودتك / ما الذي تغيّر',
    title: 'Hermes بشكل جديد',
    subtitle: 'أسماء جديدة، ومفاهيم مألوفة. اختر صفًا لتجد مكانه الجديد.',
    listLabel: 'ما الذي تغيّر اسمه',
    renamed: (classic, current) => `${classic}، أصبحت ${current}`,
    lines: {
      chats: 'كل محادثة تحت الوكيل الذي أجراها',
      agents: 'خبراؤك، بإعداداتهم وذاكرتهم',
      library: 'كل ما يصنعه وكلاؤك',
      routines: 'مهام تستمر في العمل وفق جدول',
      machines: 'حيث يعمل وكلاؤك'
    },
    atLeast: count => `${count}+`,
    footnote: 'انقر عنصرًا في شريط التنقل أو اضغط Esc للإغلاق. أعد فتحها من الإشعارات.',
    done: 'لنبدأ العمل'
  },
  formerlyLine: classic => `سابقًا ${classic}`
}

const COPY: Record<BundledLocale, DiscoveryCopy> = { en, fr, de, es, ja, zh, 'zh-hant': zhHant, ru, ar }

export function discoveryCopy(locale: string): DiscoveryCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useDiscoveryCopy(): DiscoveryCopy {
  return discoveryCopy(useI18n().locale)
}
