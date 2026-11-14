import { RU_NOUN } from './ru_plural'
import type { BundledLocale } from './types'

export interface SimplifiedAgentsCopy {
  title: string
  description: string
  chats: string
  agents: string
  settings: string
  library: string
  /** The open Chats rail destination toggles the chat sidebar (Figma 1057:8047 / 1057:8049). */
  hideChats: string
  showChats: string
  /** The open Agents, Schedules or Settings destination toggles its page panel the same way. */
  hidePanel: string
  showPanel: string
  /** An agent card's second line when nothing else describes it. */
  noChats: string
  chatCount: (count: number) => string
  /** Under an agent's most recent chats: reveal the rest ("Show all (15)") or fold them back. */
  showAllChats: (count: number) => string
  showFewerChats: string
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
    hideChats: 'Hide chats',
    showChats: 'Show chats',
    hidePanel: 'Hide panel',
    showPanel: 'Show panel',
    noChats: 'No chats yet',
    chatCount: n => (n === 1 ? '1 chat' : `${n} chats`),
    showAllChats: n => `Show all (${n})`,
    showFewerChats: 'Show less',
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
    description:
      'Utiliser la nouvelle navigation, la barre latérale et la disposition des discussions. Désactiver pour retrouver l’interface classique.',
    chats: 'Discussions',
    agents: 'Agents',
    settings: 'Paramètres de l’agent',
    library: 'Bibliothèque',
    hideChats: 'Masquer les discussions',
    showChats: 'Afficher les discussions',
    hidePanel: 'Masquer le panneau',
    showPanel: 'Afficher le panneau',
    noChats: 'Aucune discussion',
    chatCount: n => (n === 1 ? '1 discussion' : `${n} discussions`),
    showAllChats: n => `Tout afficher (${n})`,
    showFewerChats: 'Afficher moins',
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
    description:
      'Neue Navigation, Seitenleiste und Chatansicht verwenden. Ausschalten, um zur klassischen Oberfläche zurückzukehren.',
    chats: 'Chats',
    agents: 'Agenten',
    settings: 'Agenteneinstellungen',
    library: 'Bibliothek',
    hideChats: 'Chats ausblenden',
    showChats: 'Chats einblenden',
    hidePanel: 'Seitenleiste ausblenden',
    showPanel: 'Seitenleiste einblenden',
    noChats: 'Noch keine Chats',
    chatCount: n => (n === 1 ? '1 Chat' : `${n} Chats`),
    showAllChats: n => `Alle anzeigen (${n})`,
    showFewerChats: 'Weniger anzeigen',
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
    description:
      'Usar la nueva navegación, barra lateral y distribución del chat. Desactivar para volver a la interfaz clásica.',
    chats: 'Chats',
    agents: 'Agentes',
    settings: 'Ajustes del agente',
    library: 'Biblioteca',
    hideChats: 'Ocultar chats',
    showChats: 'Mostrar chats',
    hidePanel: 'Ocultar panel',
    showPanel: 'Mostrar panel',
    noChats: 'Aún no hay chats',
    chatCount: n => (n === 1 ? '1 chat' : `${n} chats`),
    showAllChats: n => `Mostrar todo (${n})`,
    showFewerChats: 'Mostrar menos',
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
    description:
      '新しいナビゲーション、サイドバー、チャットレイアウトを使用します。オフにすると従来のインターフェースに戻ります。',
    chats: 'チャット',
    agents: 'エージェント',
    settings: 'エージェント設定',
    library: 'ライブラリ',
    hideChats: 'チャットを隠す',
    showChats: 'チャットを表示',
    hidePanel: 'パネルを隠す',
    showPanel: 'パネルを表示',
    noChats: 'チャットはまだありません',
    chatCount: n => `${n} 件のチャット`,
    showAllChats: n => `すべて表示 (${n})`,
    showFewerChats: '表示を減らす',
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
    description: '使用全新的导航、侧边栏和对话布局。关闭后恢复经典界面。',
    chats: '对话',
    agents: '智能体',
    settings: '智能体设置',
    library: '资源库',
    hideChats: '隐藏对话',
    showChats: '显示对话',
    hidePanel: '隐藏面板',
    showPanel: '显示面板',
    noChats: '还没有对话',
    chatCount: n => `${n} 个对话`,
    showAllChats: n => `显示全部 (${n})`,
    showFewerChats: '收起',
    feedbackTitle: '为什么切换回来？（可选）',
    feedbackPlaceholder: '选择原因',
    feedbackThanks: '感谢你的反馈。',
    feedbackReasons: {
      missing_chats: '找不到我的对话',
      hard_to_navigate: '导航更难用了',
      missing_features: '缺少需要的功能',
      prefer_classic: '更喜欢经典界面',
      performance: '运行缓慢或出现问题',
      other: '其他原因'
    }
  },
  'zh-hant': {
    title: '全新介面',
    description: '使用全新的導覽、側邊欄和對話版面。關閉後恢復經典介面。',
    chats: '對話',
    agents: '智慧體',
    settings: '智慧體設定',
    library: '資源庫',
    hideChats: '隱藏對話',
    showChats: '顯示對話',
    hidePanel: '隱藏面板',
    showPanel: '顯示面板',
    noChats: '尚無對話',
    chatCount: n => `${n} 個對話`,
    showAllChats: n => `顯示全部 (${n})`,
    showFewerChats: '收合',
    feedbackTitle: '為什麼切換回來？（選填）',
    feedbackPlaceholder: '選擇原因',
    feedbackThanks: '感謝你的回饋。',
    feedbackReasons: {
      missing_chats: '找不到我的對話',
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
    hideChats: 'Скрыть чаты',
    showChats: 'Показать чаты',
    hidePanel: 'Скрыть панель',
    showPanel: 'Показать панель',
    noChats: 'Чатов пока нет',
    chatCount: n => `${n} ${RU_NOUN(n, 'чат', 'чата', 'чатов')}`,
    showAllChats: n => `Показать все (${n})`,
    showFewerChats: 'Показать меньше',
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
    hideChats: 'إخفاء المحادثات',
    showChats: 'إظهار المحادثات',
    hidePanel: 'إخفاء اللوحة',
    showPanel: 'إظهار اللوحة',
    noChats: 'لا توجد محادثات بعد',
    chatCount: n => (n === 1 ? 'محادثة واحدة' : `${n} محادثات`),
    showAllChats: n => `عرض الكل (${n})`,
    showFewerChats: 'عرض أقل',
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
