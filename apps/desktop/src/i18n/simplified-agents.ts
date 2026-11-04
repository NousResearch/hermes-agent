import type { BundledLocale } from './types'

export interface SimplifiedAgentsCopy {
  title: string
  description: string
  offerTitle: string
  offerDescription: string
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
    offerTitle: 'Try simplified agent mode?',
    offerDescription:
      'We have a new setting for a simpler sidebar with pinned chats, chats, and agents. Would you like to try it? You can change this later in Settings → Appearance → Window layout.',
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
    offerTitle: 'Essayer le mode agents simplifié ?',
    offerDescription:
      'Un nouveau réglage simplifie la barre latérale : discussions épinglées, discussions et agents. Voulez-vous essayer ? Vous pourrez revenir en arrière dans Paramètres → Apparence → Disposition de la fenêtre.',
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
    offerTitle: 'Vereinfachten Agentenmodus ausprobieren?',
    offerDescription:
      'Eine neue Einstellung vereinfacht die Seitenleiste mit angehefteten Chats, Chats und Agenten. Möchten Sie sie ausprobieren? Änderbar unter Einstellungen → Erscheinungsbild → Fensterlayout.',
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
    offerTitle: '¿Probar el modo de agentes simplificado?',
    offerDescription:
      'Una nueva opción simplifica la barra lateral con chats fijados, chats y agentes. ¿Quieres probarla? Puedes cambiarla en Ajustes → Apariencia → Diseño de ventana.',
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
    offerTitle: 'シンプルなエージェントモードを試しますか？',
    offerDescription:
      '固定したチャット、チャット、エージェントを表示する新しいサイドバー設定です。試しますか？設定 → 外観 → ウィンドウレイアウトで変更できます。',
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
    offerTitle: '试用简化智能体模式？',
    offerDescription: '新增侧边栏设置，显示置顶聊天、聊天和智能体。要试用吗？之后可在设置 → 外观 → 窗口布局中更改。',
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
    offerTitle: '試用簡化代理模式？',
    offerDescription: '新增側邊欄設定，顯示釘選聊天、聊天與代理。要試用嗎？之後可在設定 → 外觀 → 視窗配置中變更。',
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
    offerTitle: 'Попробовать упрощённый режим агентов?',
    offerDescription:
      'Новая настройка упрощает боковую панель: закреплённые чаты, чаты и агенты. Попробовать? Изменить выбор можно в Настройки → Оформление → Макет окна.',
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
    offerTitle: 'هل تريد تجربة وضع الوكلاء المبسط؟',
    offerDescription:
      'إعداد جديد لشريط جانبي أبسط يعرض المحادثات المثبتة والمحادثات والوكلاء. هل تريد تجربته؟ يمكنك تغييره لاحقًا في الإعدادات ← المظهر ← تخطيط النافذة.',
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
