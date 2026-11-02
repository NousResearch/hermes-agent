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
    settings: 'Agent settings'
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
    settings: 'Paramètres de l’agent'
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
    settings: 'Agenteneinstellungen'
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
    settings: 'Ajustes del agente'
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
    settings: 'エージェント設定'
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
    settings: '智能体设置'
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
    settings: '代理設定'
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
    settings: 'Настройки агента'
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
    settings: 'إعدادات الوكيل'
  }
}
