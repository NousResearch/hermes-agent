import type { BundledLocale } from './types'

export interface SimplifiedAgentsCopy {
  title: string
  description: string
  offerTitle: string
  offerDescription: string
  offerRevert: string
  yes: string
  no: string
  chats: string
  agents: string
  settings: string
  library: string
  routines: string
  feedbackTitle: string
  feedbackPlaceholder: string
  feedbackThanks: string
  feedbackReasons: {
    missing_chats: string
    hard_to_navigate: string
    missing_features: string
    prefer_classic: string
    performance: string
    other: string
  }
}

export const simplifiedAgentsCopy: Record<BundledLocale, SimplifiedAgentsCopy> = {
  en: {
    title: 'Simplified agent mode',
    description: 'Show chats and agents in a simpler sidebar.',
    offerTitle: 'Try the new Hermes sidebar?',
    offerDescription:
      'Hermes is developing a simpler sidebar with pinned chats, conversations grouped by agent, and easier access to your tools. Would you like to opt in and try it?',
    offerRevert: 'You can switch back at any time by turning off this setting:',
    yes: 'Yes, try it',
    no: 'No, keep current sidebar',
    chats: 'Chats',
    agents: 'Agents',
    settings: 'Agent settings',
    library: 'Library',
    routines: 'Routines',
    feedbackTitle: 'Why switch back? (optional)',
    feedbackPlaceholder: 'Choose a reason',
    feedbackThanks: 'Thanks for the feedback.',
    feedbackReasons: {
      missing_chats: 'I couldn’t find my chats',
      hard_to_navigate: 'It was harder to navigate',
      missing_features: 'A feature I need was missing',
      prefer_classic: 'I prefer the original sidebar',
      performance: 'It felt slow or buggy',
      other: 'Something else'
    }
  },
  fr: {
    title: 'Mode agents simplifié',
    description: 'Afficher les discussions et les agents dans une barre latérale simplifiée.',
    offerTitle: 'Essayer la nouvelle barre latérale Hermes ?',
    offerDescription:
      'Hermes développe une barre latérale plus simple avec des discussions épinglées, des conversations regroupées par agent et un accès plus facile à vos outils. Voulez-vous l’essayer ?',
    offerRevert: 'Vous pouvez revenir en arrière à tout moment en désactivant ce réglage :',
    yes: 'Oui, essayer',
    no: 'Non, conserver la barre actuelle',
    chats: 'Discussions',
    agents: 'Agents',
    settings: 'Paramètres de l’agent',
    library: 'Bibliothèque',
    routines: 'Routines',
    feedbackTitle: 'Pourquoi revenir en arrière ? (facultatif)',
    feedbackPlaceholder: 'Choisir une raison',
    feedbackThanks: 'Merci pour votre retour.',
    feedbackReasons: {
      missing_chats: 'Je ne retrouvais pas mes discussions',
      hard_to_navigate: 'La navigation était plus difficile',
      missing_features: 'Une fonctionnalité me manquait',
      prefer_classic: 'Je préfère la barre latérale originale',
      performance: 'C’était lent ou instable',
      other: 'Autre raison'
    }
  },
  de: {
    title: 'Vereinfachter Agentenmodus',
    description: 'Chats und Agenten in einer einfacheren Seitenleiste anzeigen.',
    offerTitle: 'Die neue Hermes-Seitenleiste ausprobieren?',
    offerDescription:
      'Hermes entwickelt eine einfachere Seitenleiste mit angehefteten Chats, nach Agenten gruppierten Gesprächen und leichterem Zugriff auf deine Werkzeuge. Möchtest du sie ausprobieren?',
    offerRevert: 'Du kannst jederzeit zurückwechseln, indem du diese Einstellung ausschaltest:',
    yes: 'Ja, ausprobieren',
    no: 'Nein, aktuelle Seitenleiste behalten',
    chats: 'Chats',
    agents: 'Agenten',
    settings: 'Agenteneinstellungen',
    library: 'Bibliothek',
    routines: 'Routinen',
    feedbackTitle: 'Warum zurückwechseln? (optional)',
    feedbackPlaceholder: 'Grund auswählen',
    feedbackThanks: 'Danke für dein Feedback.',
    feedbackReasons: {
      missing_chats: 'Ich konnte meine Chats nicht finden',
      hard_to_navigate: 'Die Navigation war schwieriger',
      missing_features: 'Eine benötigte Funktion fehlte',
      prefer_classic: 'Ich bevorzuge die bisherige Seitenleiste',
      performance: 'Es war langsam oder fehlerhaft',
      other: 'Etwas anderes'
    }
  },
  es: {
    title: 'Modo de agentes simplificado',
    description: 'Mostrar chats y agentes en una barra lateral más sencilla.',
    offerTitle: '¿Probar la nueva barra lateral de Hermes?',
    offerDescription:
      'Hermes está desarrollando una barra lateral más sencilla con chats fijados, conversaciones agrupadas por agente y acceso más fácil a tus herramientas. ¿Quieres probarla?',
    offerRevert: 'Puedes volver cuando quieras desactivando este ajuste:',
    yes: 'Sí, probar',
    no: 'No, mantener la barra actual',
    chats: 'Chats',
    agents: 'Agentes',
    settings: 'Ajustes del agente',
    library: 'Biblioteca',
    routines: 'Rutinas',
    feedbackTitle: '¿Por qué volver? (opcional)',
    feedbackPlaceholder: 'Elige un motivo',
    feedbackThanks: 'Gracias por tus comentarios.',
    feedbackReasons: {
      missing_chats: 'No encontraba mis chats',
      hard_to_navigate: 'Era más difícil navegar',
      missing_features: 'Faltaba una función que necesito',
      prefer_classic: 'Prefiero la barra lateral original',
      performance: 'Era lento o tenía errores',
      other: 'Otro motivo'
    }
  },
  ja: {
    title: 'シンプルなエージェントモード',
    description: 'チャットとエージェントをシンプルなサイドバーに表示します。',
    offerTitle: 'Hermesの新しいサイドバーを試しますか？',
    offerDescription:
      'Hermesでは、ピン留めしたチャット、エージェントごとにまとめた会話、ツールへのアクセスを備えた、よりシンプルなサイドバーを開発しています。試してみませんか？',
    offerRevert: '次の設定をオフにすると、いつでも元に戻せます：',
    yes: 'はい、試す',
    no: 'いいえ、現在の表示を維持',
    chats: 'チャット',
    agents: 'エージェント',
    settings: 'エージェント設定',
    library: 'ライブラリ',
    routines: 'ルーチン',
    feedbackTitle: '元に戻した理由は？（任意）',
    feedbackPlaceholder: '理由を選択',
    feedbackThanks: 'ご意見ありがとうございます。',
    feedbackReasons: {
      missing_chats: 'チャットが見つからなかった',
      hard_to_navigate: '操作が分かりにくかった',
      missing_features: '必要な機能がなかった',
      prefer_classic: '元のサイドバーの方が好き',
      performance: '動作が遅い、または不具合があった',
      other: 'その他'
    }
  },
  zh: {
    title: '简化智能体模式',
    description: '在更简洁的侧边栏中显示聊天和智能体。',
    offerTitle: '试用 Hermes 的新侧边栏？',
    offerDescription:
      'Hermes 正在开发更简洁的侧边栏，包含置顶聊天、按智能体分组的会话，以及更便捷的工具入口。要选择试用吗？',
    offerRevert: '您可以随时关闭以下设置，切回原来的侧边栏：',
    yes: '是，试用',
    no: '否，保留当前侧边栏',
    chats: '聊天',
    agents: '智能体',
    settings: '智能体设置',
    library: '资源库',
    routines: '例行任务',
    feedbackTitle: '为什么切换回来？（可选）',
    feedbackPlaceholder: '选择原因',
    feedbackThanks: '感谢你的反馈。',
    feedbackReasons: {
      missing_chats: '找不到我的聊天',
      hard_to_navigate: '导航更难用了',
      missing_features: '缺少需要的功能',
      prefer_classic: '更喜欢原来的侧边栏',
      performance: '运行缓慢或出现问题',
      other: '其他原因'
    }
  },
  'zh-hant': {
    title: '簡化代理模式',
    description: '在更簡潔的側邊欄中顯示聊天與代理。',
    offerTitle: '試用 Hermes 的新側邊欄？',
    offerDescription:
      'Hermes 正在開發更簡潔的側邊欄，包含釘選聊天、依代理分組的對話，以及更便利的工具入口。要選擇試用嗎？',
    offerRevert: '您可以隨時關閉以下設定，切回原本的側邊欄：',
    yes: '是，試用',
    no: '否，保留目前側邊欄',
    chats: '聊天',
    agents: '代理',
    settings: '代理設定',
    library: '資料庫',
    routines: '例行任務',
    feedbackTitle: '為什麼切換回來？（選填）',
    feedbackPlaceholder: '選擇原因',
    feedbackThanks: '感謝你的回饋。',
    feedbackReasons: {
      missing_chats: '找不到我的聊天',
      hard_to_navigate: '導覽更難用了',
      missing_features: '缺少需要的功能',
      prefer_classic: '比較喜歡原本的側邊欄',
      performance: '執行緩慢或出現問題',
      other: '其他原因'
    }
  },
  ru: {
    title: 'Упрощённый режим агентов',
    description: 'Показывать чаты и агентов в упрощённой боковой панели.',
    offerTitle: 'Попробовать новую боковую панель Hermes?',
    offerDescription:
      'Hermes разрабатывает более простую боковую панель с закреплёнными чатами, беседами по агентам и удобным доступом к инструментам. Хотите попробовать?',
    offerRevert: 'Вы можете в любой момент вернуться, выключив эту настройку:',
    yes: 'Да, попробовать',
    no: 'Нет, оставить текущую панель',
    chats: 'Чаты',
    agents: 'Агенты',
    settings: 'Настройки агента',
    library: 'Библиотека',
    routines: 'Регулярные задачи',
    feedbackTitle: 'Почему вернулись? (необязательно)',
    feedbackPlaceholder: 'Выберите причину',
    feedbackThanks: 'Спасибо за отзыв.',
    feedbackReasons: {
      missing_chats: 'Не удалось найти чаты',
      hard_to_navigate: 'Навигация стала сложнее',
      missing_features: 'Не хватало нужной функции',
      prefer_classic: 'Предпочитаю прежнюю боковую панель',
      performance: 'Медленно работает или есть ошибки',
      other: 'Другая причина'
    }
  },
  ar: {
    title: 'وضع الوكلاء المبسط',
    description: 'عرض المحادثات والوكلاء في شريط جانبي أبسط.',
    offerTitle: 'هل تريد تجربة الشريط الجانبي الجديد من Hermes؟',
    offerDescription:
      'يطوّر Hermes شريطًا جانبيًا أبسط يضم المحادثات المثبتة والمحادثات المجمّعة حسب الوكيل، مع وصول أسهل إلى أدواتك. هل تريد الاشتراك في تجربته؟',
    offerRevert: 'يمكنك الرجوع في أي وقت بإيقاف هذا الإعداد:',
    yes: 'نعم، جرّبه',
    no: 'لا، احتفظ بالشريط الحالي',
    chats: 'المحادثات',
    agents: 'الوكلاء',
    settings: 'إعدادات الوكيل',
    library: 'المكتبة',
    routines: 'المهام الدورية',
    feedbackTitle: 'لماذا عدت للوضع السابق؟ (اختياري)',
    feedbackPlaceholder: 'اختر سببًا',
    feedbackThanks: 'شكرًا على ملاحظاتك.',
    feedbackReasons: {
      missing_chats: 'لم أجد محادثاتي',
      hard_to_navigate: 'أصبح التنقل أصعب',
      missing_features: 'ميزة أحتاجها غير متوفرة',
      prefer_classic: 'أفضل الشريط الجانبي الأصلي',
      performance: 'كان بطيئًا أو به أخطاء',
      other: 'سبب آخر'
    }
  }
}
