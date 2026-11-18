import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'
import type { OldNameAlias } from '@/lib/old-name-aliases'

/** The redesign's discovery surfaces (notifications tray, rename splash, new-version notice, the
 *  palette's old-word rows). Only the redesign renders them. The new nouns (Chats, Agents, ...)
 *  are the rail's own labels, read from the catalog where the rail reads them; this module holds
 *  only the classic words they replace and the surfaces' own sentences. */
export interface DiscoveryCopy {
  /** The words the classic interface used, as it showed them: the rename splash strikes them
   *  through, and ⌘K matches them in this language. */
  classicNames: Record<OldNameAlias['code'], string>
  tray: {
    title: string
    /** The bell's accessible description while something in the tray is new. */
    unseen: string
    empty: string
    newBadge: string
  }
  renameItem: { title: string; description: string; action: string }
  updateItem: { eyebrow: string; title: string; action: string }
  newVersion: { eyebrow: string; title: string; description: string; notice: string; refresh: string }
  renameSplash: {
    eyebrow: string
    title: string
    subtitle: string
    /** The rows' list, for assistive tech. */
    listLabel: string
    /** The row's accessible name, as its strike-through and arrow show it. */
    renamed: (classic: string, current: string) => string
    lines: Record<'agents' | 'chats' | 'library' | 'machines' | 'notifications' | 'routines', string>
    notifications: string
    newBadge: string
    /** A count with more beyond the loaded page ("128+"). */
    atLeast: (count: string) => string
    footnote: string
    done: string
  }
  /** A ⌘K row for a renamed place, found by its classic word. */
  formerly: (current: string, classic: string) => string
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
    unseen: 'Something new',
    empty: 'Nothing new right now.',
    newBadge: 'New'
  },
  renameItem: {
    title: 'New names, same stuff',
    description: 'See where everything lives in the new look.',
    action: 'Open'
  },
  updateItem: { eyebrow: 'Update available', title: 'A new version of Hermes is ready', action: 'Update' },
  newVersion: {
    eyebrow: 'Update ready',
    title: 'A new version of Hermes is ready',
    description: 'It loads when you start a new chat. Nothing you’re doing is lost.',
    notice: 'A new version of Hermes is ready. It loads when you start a new chat, nothing you’re doing is lost.',
    refresh: 'Refresh now'
  },
  renameSplash: {
    eyebrow: '// Hermes update',
    title: 'Hermes got a new look',
    subtitle: 'New names, same concepts. Hover or click below to see where they live now.',
    listLabel: 'What was renamed',
    renamed: (classic, current) => `${classic}, now ${current}`,
    lines: {
      chats: 'Each chat now sits under the agent that ran it.',
      agents: 'Each agent keeps its own settings, skills and memory.',
      library: 'Everything Hermes made, from every chat, in one place.',
      routines: 'Tasks your agents run on a schedule, on their own.',
      machines: 'The computers your agents run on.',
      notifications: 'Finished routines and updates land here, including this page.'
    },
    notifications: 'Notifications',
    newBadge: 'New',
    atLeast: count => `${count}+`,
    footnote: 'Click anywhere in the nav to close this. You can come back anytime from notifications.',
    done: 'Let’s get to work'
  },
  formerly: (current, classic) => `${current}, formerly ${classic}`
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
    unseen: 'Du nouveau',
    empty: 'Rien de nouveau pour l’instant.',
    newBadge: 'Nouveau'
  },
  renameItem: {
    title: 'Nouveaux noms, mêmes choses',
    description: 'Voyez où tout se trouve dans la nouvelle interface.',
    action: 'Ouvrir'
  },
  updateItem: {
    eyebrow: 'Mise à jour disponible',
    title: 'Une nouvelle version de Hermes est prête',
    action: 'Mettre à jour'
  },
  newVersion: {
    eyebrow: 'Mise à jour prête',
    title: 'Une nouvelle version de Hermes est prête',
    description: 'Elle se charge quand vous lancez une nouvelle discussion. Rien de ce que vous faites n’est perdu.',
    notice:
      'Une nouvelle version de Hermes est prête. Elle se charge quand vous lancez une nouvelle discussion, rien de ce que vous faites n’est perdu.',
    refresh: 'Actualiser'
  },
  renameSplash: {
    eyebrow: '// Mise à jour de Hermes',
    title: 'Hermes fait peau neuve',
    subtitle: 'Nouveaux noms, mêmes concepts. Survolez ou cliquez ci-dessous pour voir où ils se trouvent.',
    listLabel: 'Ce qui a été renommé',
    renamed: (classic, current) => `${classic}, désormais ${current}`,
    lines: {
      chats: 'Chaque discussion se range sous l’agent qui l’a menée.',
      agents: 'Chaque agent garde ses propres réglages, compétences et mémoire.',
      library: 'Tout ce que Hermes a créé, de chaque discussion, au même endroit.',
      routines: 'Des tâches que vos agents exécutent seuls, selon un planning.',
      machines: 'Les ordinateurs sur lesquels tournent vos agents.',
      notifications: 'Les routines terminées et les mises à jour arrivent ici, y compris cette page.'
    },
    notifications: 'Notifications',
    newBadge: 'Nouveau',
    atLeast: count => `${count}+`,
    footnote:
      'Cliquez n’importe où dans la navigation pour fermer. Vous pouvez y revenir à tout moment depuis les notifications.',
    done: 'Au travail'
  },
  formerly: (current, classic) => `${current}, anciennement ${classic}`
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
    unseen: 'Etwas Neues',
    empty: 'Gerade nichts Neues.',
    newBadge: 'Neu'
  },
  renameItem: {
    title: 'Neue Namen, gleiche Dinge',
    description: 'Sieh nach, wo jetzt alles zu finden ist.',
    action: 'Öffnen'
  },
  updateItem: {
    eyebrow: 'Update verfügbar',
    title: 'Eine neue Version von Hermes ist bereit',
    action: 'Aktualisieren'
  },
  newVersion: {
    eyebrow: 'Update bereit',
    title: 'Eine neue Version von Hermes ist bereit',
    description: 'Sie lädt, wenn du einen neuen Chat startest. Nichts, woran du arbeitest, geht verloren.',
    notice:
      'Eine neue Version von Hermes ist bereit. Sie lädt, wenn du einen neuen Chat startest, nichts, woran du arbeitest, geht verloren.',
    refresh: 'Jetzt neu laden'
  },
  renameSplash: {
    eyebrow: '// Hermes-Update',
    title: 'Hermes sieht jetzt neu aus',
    subtitle: 'Neue Namen, gleiche Konzepte. Zeig auf eine Zeile oder klick sie an, um zu sehen, wo sie jetzt liegen.',
    listLabel: 'Was umbenannt wurde',
    renamed: (classic, current) => `${classic}, jetzt ${current}`,
    lines: {
      chats: 'Jeder Chat liegt jetzt unter dem Agenten, der ihn geführt hat.',
      agents: 'Jeder Agent behält seine eigenen Einstellungen, Skills und Erinnerungen.',
      library: 'Alles, was Hermes erstellt hat, aus jedem Chat, an einem Ort.',
      routines: 'Aufgaben, die deine Agenten selbstständig nach Zeitplan ausführen.',
      machines: 'Die Computer, auf denen deine Agenten laufen.',
      notifications: 'Fertige Routinen und Updates landen hier, auch diese Seite.'
    },
    notifications: 'Benachrichtigungen',
    newBadge: 'Neu',
    atLeast: count => `${count}+`,
    footnote: 'Klick irgendwo in die Navigation, um das zu schließen. Unter Benachrichtigungen findest du es jederzeit wieder.',
    done: 'Los geht’s'
  },
  formerly: (current, classic) => `${current}, früher ${classic}`
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
    unseen: 'Hay novedades',
    empty: 'Nada nuevo por ahora.',
    newBadge: 'Nuevo'
  },
  renameItem: {
    title: 'Nombres nuevos, lo mismo de siempre',
    description: 'Mira dónde está todo con el nuevo diseño.',
    action: 'Abrir'
  },
  updateItem: {
    eyebrow: 'Actualización disponible',
    title: 'Hay una nueva versión de Hermes lista',
    action: 'Actualizar'
  },
  newVersion: {
    eyebrow: 'Actualización lista',
    title: 'Hay una nueva versión de Hermes lista',
    description: 'Se carga cuando empiezas un chat nuevo. No se pierde nada de lo que estás haciendo.',
    notice:
      'Hay una nueva versión de Hermes lista. Se carga cuando empiezas un chat nuevo, no se pierde nada de lo que estás haciendo.',
    refresh: 'Actualizar ahora'
  },
  renameSplash: {
    eyebrow: '// Actualización de Hermes',
    title: 'Hermes tiene un nuevo aspecto',
    subtitle: 'Nombres nuevos, mismos conceptos. Pasa el cursor o haz clic abajo para ver dónde están ahora.',
    listLabel: 'Qué cambió de nombre',
    renamed: (classic, current) => `${classic}, ahora ${current}`,
    lines: {
      chats: 'Cada chat ahora está bajo el agente que lo llevó.',
      agents: 'Cada agente conserva su propia configuración, habilidades y memoria.',
      library: 'Todo lo que Hermes creó, de cada chat, en un solo lugar.',
      routines: 'Tareas que tus agentes ejecutan solos, según un horario.',
      machines: 'Los ordenadores en los que funcionan tus agentes.',
      notifications: 'Las rutinas terminadas y las actualizaciones llegan aquí, incluida esta página.'
    },
    notifications: 'Notificaciones',
    newBadge: 'Nuevo',
    atLeast: count => `${count}+`,
    footnote: 'Haz clic en cualquier parte de la navegación para cerrar. Puedes volver cuando quieras desde las notificaciones.',
    done: 'Manos a la obra'
  },
  formerly: (current, classic) => `${current}, antes ${classic}`
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
    unseen: '新着あり',
    empty: '今は新しいお知らせはありません。',
    newBadge: '新着'
  },
  renameItem: {
    title: '名前が変わっても、中身は同じ',
    description: '新しいデザインでどこに何があるかを確認できます。',
    action: '開く'
  },
  updateItem: { eyebrow: 'アップデートあり', title: 'Hermes の新しいバージョンの準備ができました', action: 'アップデート' },
  newVersion: {
    eyebrow: 'アップデート準備完了',
    title: 'Hermes の新しいバージョンの準備ができました',
    description: '新しいチャットを始めたときに読み込まれます。作業中の内容は失われません。',
    notice: 'Hermes の新しいバージョンの準備ができました。新しいチャットを始めたときに読み込まれ、作業中の内容は失われません。',
    refresh: '今すぐ更新'
  },
  renameSplash: {
    eyebrow: '// Hermes アップデート',
    title: 'Hermes が新しくなりました',
    subtitle: '名前は新しく、考え方はそのまま。下の項目にカーソルを合わせるかクリックすると、今の場所がわかります。',
    listLabel: '名前が変わったもの',
    renamed: (classic, current) => `${classic} は ${current} になりました`,
    lines: {
      chats: '各チャットは、それを実行したエージェントの下に並びます。',
      agents: '各エージェントは自分の設定、スキル、メモリーを持ちます。',
      library: 'Hermes がすべてのチャットで作ったものを、一か所に。',
      routines: 'エージェントがスケジュールに沿って自動で行うタスク。',
      machines: 'エージェントが動くコンピューター。',
      notifications: '完了したルーティンやアップデートはここに届きます。このページもここから開けます。'
    },
    notifications: '通知',
    newBadge: '新着',
    atLeast: count => `${count}+`,
    footnote: 'ナビゲーションのどこかをクリックすると閉じます。通知からいつでも戻れます。',
    done: 'さっそく始める'
  },
  formerly: (current, classic) => `${current}（旧: ${classic}）`
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
    unseen: '有新内容',
    empty: '暂时没有新内容。',
    newBadge: '新'
  },
  renameItem: {
    title: '名字变了，内容没变',
    description: '看看新界面里各项内容在哪里。',
    action: '打开'
  },
  updateItem: { eyebrow: '有可用更新', title: 'Hermes 新版本已就绪', action: '更新' },
  newVersion: {
    eyebrow: '更新已就绪',
    title: 'Hermes 新版本已就绪',
    description: '开始新对话时就会加载。你正在做的事不会丢失。',
    notice: 'Hermes 新版本已就绪。开始新对话时就会加载，你正在做的事不会丢失。',
    refresh: '立即刷新'
  },
  renameSplash: {
    eyebrow: '// Hermes 更新',
    title: 'Hermes 换了新面貌',
    subtitle: '名字是新的，概念不变。将鼠标悬停或点击下方条目，看看它们现在在哪里。',
    listLabel: '改名的内容',
    renamed: (classic, current) => `${classic}，现在叫 ${current}`,
    lines: {
      chats: '每个对话现在都归在运行它的智能体下面。',
      agents: '每个智能体都有自己的设置、技能和记忆。',
      library: 'Hermes 在所有对话中做出的内容，集中在一处。',
      routines: '智能体按计划自动执行的任务。',
      machines: '运行智能体的电脑。',
      notifications: '完成的例行任务和更新都会出现在这里，包括这一页。'
    },
    notifications: '通知',
    newBadge: '新',
    atLeast: count => `${count}+`,
    footnote: '点击导航中的任意位置即可关闭。你随时可以从通知中再次打开。',
    done: '开始工作'
  },
  formerly: (current, classic) => `${current}（原 ${classic}）`
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
    unseen: '有新內容',
    empty: '目前沒有新內容。',
    newBadge: '新'
  },
  renameItem: {
    title: '名稱變了，內容沒變',
    description: '看看新介面裡各項內容在哪裡。',
    action: '開啟'
  },
  updateItem: { eyebrow: '有可用更新', title: 'Hermes 新版本已就緒', action: '更新' },
  newVersion: {
    eyebrow: '更新已就緒',
    title: 'Hermes 新版本已就緒',
    description: '開始新對話時就會載入。你正在做的事不會遺失。',
    notice: 'Hermes 新版本已就緒。開始新對話時就會載入，你正在做的事不會遺失。',
    refresh: '立即重新整理'
  },
  renameSplash: {
    eyebrow: '// Hermes 更新',
    title: 'Hermes 換了新面貌',
    subtitle: '名稱是新的，概念不變。將游標停在下方項目上或點一下，看看它們現在在哪裡。',
    listLabel: '改名的內容',
    renamed: (classic, current) => `${classic}，現在叫 ${current}`,
    lines: {
      chats: '每個對話現在都歸在執行它的智慧體底下。',
      agents: '每個智慧體都有自己的設定、技能和記憶。',
      library: 'Hermes 在所有對話中做出的內容，集中在一處。',
      routines: '智慧體依排程自動執行的工作。',
      machines: '執行智慧體的電腦。',
      notifications: '完成的例行任務和更新都會出現在這裡，包括這一頁。'
    },
    notifications: '通知',
    newBadge: '新',
    atLeast: count => `${count}+`,
    footnote: '點一下導覽列的任何地方即可關閉。你隨時可以從通知再次開啟。',
    done: '開始工作'
  },
  formerly: (current, classic) => `${current}（原 ${classic}）`
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
    unseen: 'Есть новое',
    empty: 'Пока ничего нового.',
    newBadge: 'Новое'
  },
  renameItem: {
    title: 'Новые названия, всё то же самое',
    description: 'Посмотрите, где что находится в новом интерфейсе.',
    action: 'Открыть'
  },
  updateItem: { eyebrow: 'Доступно обновление', title: 'Новая версия Hermes готова', action: 'Обновить' },
  newVersion: {
    eyebrow: 'Обновление готово',
    title: 'Новая версия Hermes готова',
    description: 'Она загрузится, когда вы начнёте новый чат. Ничего из того, что вы делаете, не потеряется.',
    notice:
      'Новая версия Hermes готова. Она загрузится, когда вы начнёте новый чат, ничего из того, что вы делаете, не потеряется.',
    refresh: 'Обновить сейчас'
  },
  renameSplash: {
    eyebrow: '// Обновление Hermes',
    title: 'Hermes выглядит по-новому',
    subtitle: 'Новые названия, те же понятия. Наведите курсор или нажмите на строку ниже, чтобы увидеть, где они теперь.',
    listLabel: 'Что переименовано',
    renamed: (classic, current) => `${classic}, теперь ${current}`,
    lines: {
      chats: 'Каждый чат теперь находится под агентом, который его вёл.',
      agents: 'У каждого агента свои настройки, навыки и память.',
      library: 'Всё, что Hermes создал во всех чатах, в одном месте.',
      routines: 'Задачи, которые ваши агенты сами выполняют по расписанию.',
      machines: 'Компьютеры, на которых работают ваши агенты.',
      notifications: 'Завершённые рутины и обновления приходят сюда, включая эту страницу.'
    },
    notifications: 'Уведомления',
    newBadge: 'Новое',
    atLeast: count => `${count}+`,
    footnote: 'Нажмите в любом месте навигации, чтобы закрыть. Вернуться сюда можно в любой момент из уведомлений.',
    done: 'За работу'
  },
  formerly: (current, classic) => `${current}, раньше ${classic}`
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
    unseen: 'يوجد جديد',
    empty: 'لا جديد الآن.',
    newBadge: 'جديد'
  },
  renameItem: {
    title: 'أسماء جديدة، والأشياء نفسها',
    description: 'اعرف مكان كل شيء في الشكل الجديد.',
    action: 'فتح'
  },
  updateItem: { eyebrow: 'تحديث متاح', title: 'إصدار جديد من Hermes جاهز', action: 'تحديث' },
  newVersion: {
    eyebrow: 'التحديث جاهز',
    title: 'إصدار جديد من Hermes جاهز',
    description: 'يُحمَّل عندما تبدأ محادثة جديدة. لن يضيع شيء مما تعمل عليه.',
    notice: 'إصدار جديد من Hermes جاهز. يُحمَّل عندما تبدأ محادثة جديدة، ولن يضيع شيء مما تعمل عليه.',
    refresh: 'التحديث الآن'
  },
  renameSplash: {
    eyebrow: '// تحديث Hermes',
    title: 'Hermes بشكل جديد',
    subtitle: 'أسماء جديدة، والمفاهيم نفسها. مرّر المؤشر أو انقر أدناه لترى مكانها الآن.',
    listLabel: 'ما الذي تغيّر اسمه',
    renamed: (classic, current) => `${classic}، أصبحت ${current}`,
    lines: {
      chats: 'كل محادثة أصبحت تحت الوكيل الذي أجراها.',
      agents: 'لكل وكيل إعداداته ومهاراته وذاكرته.',
      library: 'كل ما صنعه Hermes، من كل محادثة، في مكان واحد.',
      routines: 'مهام ينفّذها وكلاؤك وحدهم وفق جدول.',
      machines: 'الحواسيب التي يعمل عليها وكلاؤك.',
      notifications: 'الروتينات المكتملة والتحديثات تصل إلى هنا، ومنها هذه الصفحة.'
    },
    notifications: 'الإشعارات',
    newBadge: 'جديد',
    atLeast: count => `${count}+`,
    footnote: 'انقر في أي مكان من شريط التنقل لإغلاق هذه الصفحة. يمكنك العودة إليها متى شئت من الإشعارات.',
    done: 'لنبدأ العمل'
  },
  formerly: (current, classic) => `${current}، سابقًا ${classic}`
}

const COPY: Record<BundledLocale, DiscoveryCopy> = { en, fr, de, es, ja, zh, 'zh-hant': zhHant, ru, ar }

export function discoveryCopy(locale: string): DiscoveryCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useDiscoveryCopy(): DiscoveryCopy {
  return discoveryCopy(useI18n().locale)
}
