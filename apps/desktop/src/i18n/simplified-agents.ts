import type { BundledLocale } from './types'

export interface SimplifiedAgentsCopy {
  title: string
  description: string
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
    title: 'Redesigned interface',
    description: 'Use the new navigation, sidebar, and chat layout. Turn off to restore the classic interface.',
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
      prefer_classic: 'I prefer the classic interface',
      performance: 'It felt slow or buggy',
      other: 'Something else'
    }
  },
  fr: {
    title: 'Interface repensée',
    description: 'Utiliser la nouvelle navigation, la barre latérale et la disposition des discussions. Désactiver pour retrouver l’interface classique.',
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
      prefer_classic: 'Je préfère l’interface classique',
      performance: 'C’était lent ou instable',
      other: 'Autre raison'
    }
  },
  de: {
    title: 'Neue Benutzeroberfläche',
    description: 'Neue Navigation, Seitenleiste und Chatansicht verwenden. Ausschalten, um zur klassischen Oberfläche zurückzukehren.',
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
      prefer_classic: 'Ich bevorzuge die klassische Oberfläche',
      performance: 'Es war langsam oder fehlerhaft',
      other: 'Etwas anderes'
    }
  },
  es: {
    title: 'Interfaz rediseñada',
    description: 'Usar la nueva navegación, barra lateral y distribución del chat. Desactivar para volver a la interfaz clásica.',
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
      prefer_classic: 'Prefiero la interfaz clásica',
      performance: 'Era lento o tenía errores',
      other: 'Otro motivo'
    }
  },
  ja: {
    title: '新しいインターフェース',
    description: '新しいナビゲーション、サイドバー、チャットレイアウトを使用します。オフにすると従来のインターフェースに戻ります。',
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
      prefer_classic: '従来のインターフェースの方が好き',
      performance: '動作が遅い、または不具合があった',
      other: 'その他'
    }
  },
  zh: {
    title: '全新界面',
    description: '使用全新的导航、侧边栏和聊天布局。关闭后恢复经典界面。',
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
      prefer_classic: '更喜欢经典界面',
      performance: '运行缓慢或出现问题',
      other: '其他原因'
    }
  },
  'zh-hant': {
    title: '全新介面',
    description: '使用全新的導覽、側邊欄和聊天版面。關閉後恢復經典介面。',
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
      prefer_classic: '比較喜歡經典介面',
      performance: '執行緩慢或出現問題',
      other: '其他原因'
    }
  },
  ru: {
    title: 'Обновлённый интерфейс',
    description: 'Новая навигация, боковая панель и оформление чата. Выключите, чтобы вернуть классический интерфейс.',
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
      prefer_classic: 'Предпочитаю классический интерфейс',
      performance: 'Медленно работает или есть ошибки',
      other: 'Другая причина'
    }
  },
  ar: {
    title: 'الواجهة الجديدة',
    description: 'استخدم التنقل والشريط الجانبي وتخطيط المحادثات الجديد. أوقف الخيار لاستعادة الواجهة الكلاسيكية.',
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
      prefer_classic: 'أفضل الواجهة الكلاسيكية',
      performance: 'كان بطيئًا أو به أخطاء',
      other: 'سبب آخر'
    }
  }
}
