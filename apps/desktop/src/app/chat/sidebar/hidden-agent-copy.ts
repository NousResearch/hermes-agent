import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

const en = {
  hiddenAgents: 'Hidden agents',
  hide: 'Hide from chat list',
  show: 'Show',
  showInChatList: 'Show in chat list',
  newChat: 'New chat with this agent',
  openAgent: 'Open agent',
  actions: 'Agent actions',
  hiddenNotMuted: 'Hidden, not muted',
  hideDescription: 'Removes this agent from your chat list only. Its chats stay in search, and it keeps working.',
  notificationsContinue: 'Notifications still come through.',
  attentionExplanation: 'If it needs you, like an approval or a failed routine, a dot shows under Hidden agents.',
  restoreExplanation: 'Undo anytime from Hidden agents at the bottom of your chat list.',
  approval: 'Waiting for your approval',
  input: 'Needs input',
  failed: 'Failed',
  running: 'Running',
  unread: 'Unread'
}

type HiddenAgentCopy = { [Key in keyof typeof en]: string }

const copy = {
  en,
  fr: {
    hiddenAgents: 'Agents masqués',
    hide: 'Masquer de la liste des discussions',
    show: 'Afficher',
    showInChatList: 'Afficher dans la liste des discussions',
    newChat: 'Nouvelle discussion avec cet agent',
    openAgent: 'Ouvrir l’agent',
    actions: 'Actions de l’agent',
    hiddenNotMuted: 'Masqué, pas silencieux',
    hideDescription:
      'Retire cet agent de votre liste de discussions uniquement. Ses discussions restent accessibles dans la recherche et il continue de travailler.',
    notificationsContinue: 'Les notifications continuent.',
    attentionExplanation:
      'Si cet agent attend une approbation ou si une routine échoue, un point apparaît à côté des agents masqués.',
    restoreExplanation: 'Vous pouvez le réafficher depuis les agents masqués en bas de la liste des discussions.',
    approval: 'En attente de votre approbation',
    input: 'Réponse requise',
    failed: 'Échec',
    running: 'En cours',
    unread: 'Non lu'
  },
  de: {
    hiddenAgents: 'Ausgeblendete Agenten',
    hide: 'Aus Chatliste ausblenden',
    show: 'Anzeigen',
    showInChatList: 'In Chatliste anzeigen',
    newChat: 'Neuer Chat mit diesem Agenten',
    openAgent: 'Agent öffnen',
    actions: 'Agentenaktionen',
    hiddenNotMuted: 'Ausgeblendet, nicht stummgeschaltet',
    hideDescription:
      'Entfernt diesen Agenten nur aus deiner Chatliste. Seine Chats bleiben über die Suche erreichbar und er arbeitet weiter.',
    notificationsContinue: 'Benachrichtigungen bleiben aktiv.',
    attentionExplanation:
      'Wenn dieser Agent eine Freigabe benötigt oder eine Routine fehlschlägt, erscheint ein Punkt neben den ausgeblendeten Agenten.',
    restoreExplanation: 'Du kannst ihn unter den ausgeblendeten Agenten am Ende der Chatliste wieder anzeigen.',
    approval: 'Wartet auf deine Freigabe',
    input: 'Eingabe erforderlich',
    failed: 'Fehlgeschlagen',
    running: 'Läuft',
    unread: 'Ungelesen'
  },
  es: {
    hiddenAgents: 'Agentes ocultos',
    hide: 'Ocultar de la lista de chats',
    show: 'Mostrar',
    showInChatList: 'Mostrar en la lista de chats',
    newChat: 'Nuevo chat con este agente',
    openAgent: 'Abrir agente',
    actions: 'Acciones del agente',
    hiddenNotMuted: 'Oculto, no silenciado',
    hideDescription:
      'Quita este agente solo de tu lista de chats. Sus chats siguen en la búsqueda y continúa trabajando.',
    notificationsContinue: 'Las notificaciones continúan.',
    attentionExplanation:
      'Si este agente necesita aprobación o falla una rutina, aparece un punto junto a los agentes ocultos.',
    restoreExplanation: 'Puedes volver a mostrarlo desde los agentes ocultos al final de la lista de chats.',
    approval: 'Esperando tu aprobación',
    input: 'Necesita una respuesta',
    failed: 'Falló',
    running: 'En ejecución',
    unread: 'Sin leer'
  },
  ja: {
    hiddenAgents: '非表示のエージェント',
    hide: 'チャット一覧から非表示にする',
    show: '表示',
    showInChatList: 'チャット一覧に表示',
    newChat: 'このエージェントと新しいチャット',
    openAgent: 'エージェントを開く',
    actions: 'エージェントの操作',
    hiddenNotMuted: '非表示でも通知は有効',
    hideDescription:
      'このエージェントをチャット一覧からのみ非表示にします。チャットは引き続き検索でき、作業も継続します。',
    notificationsContinue: '通知は引き続き届きます。',
    attentionExplanation:
      '承認が必要な場合やルーティンが失敗した場合、非表示のエージェントの横にドットが表示されます。',
    restoreExplanation: 'チャット一覧の下部にある非表示のエージェントから再表示できます。',
    approval: 'あなたの承認を待っています',
    input: '入力が必要',
    failed: '失敗',
    running: '実行中',
    unread: '未読'
  },
  zh: {
    hiddenAgents: '已隐藏的智能体',
    hide: '从对话列表隐藏',
    show: '显示',
    showInChatList: '在对话列表中显示',
    newChat: '与此智能体开始新对话',
    openAgent: '打开智能体',
    actions: '智能体操作',
    hiddenNotMuted: '隐藏，但不静音',
    hideDescription: '仅从对话列表中隐藏此智能体。它的对话仍可搜索，工作也会继续。',
    notificationsContinue: '通知仍会继续。',
    attentionExplanation: '如果此智能体需要审批或例行任务失败，已隐藏的智能体旁会出现圆点。',
    restoreExplanation: '你可以从对话列表底部的已隐藏智能体中将其重新显示。',
    approval: '等待你的审批',
    input: '需要输入',
    failed: '失败',
    running: '运行中',
    unread: '未读'
  },
  'zh-hant': {
    hiddenAgents: '已隱藏的智慧體',
    hide: '從對話清單隱藏',
    show: '顯示',
    showInChatList: '在對話清單中顯示',
    newChat: '與此智慧體開始新對話',
    openAgent: '開啟智慧體',
    actions: '智慧體操作',
    hiddenNotMuted: '隱藏，但不靜音',
    hideDescription: '僅從對話清單隱藏此智慧體。它的對話仍可搜尋，工作也會繼續。',
    notificationsContinue: '通知仍會繼續。',
    attentionExplanation: '如果此智慧體需要核准或例行任務失敗，已隱藏的智慧體旁會出現圓點。',
    restoreExplanation: '你可以從對話清單底部的已隱藏智慧體中將其重新顯示。',
    approval: '等待你的核准',
    input: '需要輸入',
    failed: '失敗',
    running: '執行中',
    unread: '未讀'
  },
  ru: {
    hiddenAgents: 'Скрытые агенты',
    hide: 'Скрыть из списка чатов',
    show: 'Показать',
    showInChatList: 'Показать в списке чатов',
    newChat: 'Новый чат с этим агентом',
    openAgent: 'Открыть агента',
    actions: 'Действия агента',
    hiddenNotMuted: 'Скрыт, но уведомления включены',
    hideDescription: 'Убирает агента только из списка чатов. Его чаты остаются в поиске, и он продолжает работать.',
    notificationsContinue: 'Уведомления продолжают поступать.',
    attentionExplanation:
      'Если агенту нужно разрешение или регулярная задача завершилась с ошибкой, рядом со скрытыми агентами появится точка.',
    restoreExplanation: 'Агента можно снова показать в разделе скрытых агентов внизу списка чатов.',
    approval: 'Ожидает вашего разрешения',
    input: 'Нужен ответ',
    failed: 'Ошибка',
    running: 'Выполняется',
    unread: 'Не прочитано'
  },
  ar: {
    hiddenAgents: 'الوكلاء المخفيون',
    hide: 'إخفاء من قائمة المحادثات',
    show: 'إظهار',
    showInChatList: 'إظهار في قائمة المحادثات',
    newChat: 'محادثة جديدة مع هذا الوكيل',
    openAgent: 'فتح الوكيل',
    actions: 'إجراءات الوكيل',
    hiddenNotMuted: 'مخفي، وليس مكتومًا',
    hideDescription: 'يزيل هذا الوكيل من قائمة المحادثات فقط. تظل محادثاته متاحة في البحث ويواصل عمله.',
    notificationsContinue: 'تستمر الإشعارات.',
    attentionExplanation: 'إذا احتاج هذا الوكيل إلى موافقة أو فشلت مهمة روتينية، تظهر نقطة بجانب الوكلاء المخفيين.',
    restoreExplanation: 'يمكنك إظهاره مجددًا من الوكلاء المخفيين أسفل قائمة المحادثات.',
    approval: 'بانتظار موافقتك',
    input: 'يحتاج إلى إدخال',
    failed: 'فشل',
    running: 'قيد التشغيل',
    unread: 'غير مقروء'
  }
} satisfies Record<BundledLocale, HiddenAgentCopy>

export function useHiddenAgentCopy(): HiddenAgentCopy {
  const { locale } = useI18n()

  return isBundledLocale(locale) ? copy[locale] : copy.en
}
