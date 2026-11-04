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
    offerTitle: 'Try the new sidebar',
    offerDescription:
      'We’re developing a new sidebar for Hermes, with chats grouped by agent, pinned conversations at the top, and quick access to agent settings.',
    offerRevert: 'You can switch back at any time by turning off this setting:',
    yes: 'Try the new sidebar',
    no: 'Keep current sidebar',
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
    offerTitle: 'Essayez la nouvelle barre latérale',
    offerDescription:
      'Nous développons une nouvelle barre latérale pour Hermes, avec des discussions regroupées par agent, des conversations épinglées en haut et un accès rapide aux paramètres des agents.',
    offerRevert: 'Vous pouvez revenir en arrière à tout moment en désactivant ce réglage :',
    yes: 'Essayer la nouvelle barre latérale',
    no: 'Conserver la barre latérale actuelle',
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
    offerTitle: 'Die neue Seitenleiste ausprobieren',
    offerDescription:
      'Wir entwickeln eine neue Seitenleiste für Hermes: Chats sind nach Agenten gruppiert, angeheftete Gespräche stehen oben und die Agenteneinstellungen sind schnell erreichbar.',
    offerRevert: 'Du kannst jederzeit zurückwechseln, indem du diese Einstellung ausschaltest:',
    yes: 'Neue Seitenleiste ausprobieren',
    no: 'Aktuelle Seitenleiste behalten',
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
    offerTitle: 'Prueba la nueva barra lateral',
    offerDescription:
      'Estamos desarrollando una nueva barra lateral para Hermes, con chats agrupados por agente, conversaciones fijadas en la parte superior y acceso rápido a los ajustes de los agentes.',
    offerRevert: 'Puedes volver cuando quieras desactivando este ajuste:',
    yes: 'Probar la nueva barra lateral',
    no: 'Mantener la barra lateral actual',
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
    offerTitle: '新しいサイドバーを試す',
    offerDescription:
      'Hermesの新しいサイドバーを開発しています。チャットはエージェントごとにまとまり、ピン留めした会話は上部に表示され、エージェント設定にもすぐにアクセスできます。',
    offerRevert: '次の設定をオフにすると、いつでも元に戻せます：',
    yes: '新しいサイドバーを試す',
    no: '現在のサイドバーを使う',
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
    offerTitle: '试用新侧边栏',
    offerDescription:
      '我们正在为 Hermes 开发新的侧边栏，聊天按智能体分组，置顶会话显示在顶部，还可快速访问智能体设置。',
    offerRevert: '您可以随时关闭以下设置，切回原来的侧边栏：',
    yes: '试用新侧边栏',
    no: '保留当前侧边栏',
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
    offerTitle: '試用新側邊欄',
    offerDescription:
      '我們正在為 Hermes 開發新的側邊欄，聊天依代理分組，釘選對話顯示在頂部，還可快速存取代理設定。',
    offerRevert: '您可以隨時關閉以下設定，切回原本的側邊欄：',
    yes: '試用新側邊欄',
    no: '保留目前側邊欄',
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
    offerTitle: 'Попробуйте новую боковую панель',
    offerDescription:
      'Мы разрабатываем новую боковую панель для Hermes: чаты сгруппированы по агентам, закреплённые беседы находятся вверху, а настройки агентов доступны в один шаг.',
    offerRevert: 'Вы можете в любой момент вернуться, выключив эту настройку:',
    yes: 'Попробовать новую панель',
    no: 'Оставить текущую панель',
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
    offerTitle: 'جرّب الشريط الجانبي الجديد',
    offerDescription:
      'نعمل على تطوير شريط جانبي جديد لـ Hermes، يجمع المحادثات حسب الوكيل، ويعرض المحادثات المثبتة في الأعلى، مع وصول سريع إلى إعدادات الوكلاء.',
    offerRevert: 'يمكنك الرجوع في أي وقت بإيقاف هذا الإعداد:',
    yes: 'جرّب الشريط الجانبي الجديد',
    no: 'الاحتفاظ بالشريط الجانبي الحالي',
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
