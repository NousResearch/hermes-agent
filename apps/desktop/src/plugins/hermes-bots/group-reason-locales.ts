export interface GroupReasonMessages {
  reasonMemberUnavailable: string
  reasonRuntimeOffline: string
  reasonQueuedExpired: string
  reasonDeliveryTimeout: string
  reasonAgentBlocked: string
  reasonCancelled: string
  reasonProviderAuth: string
  reasonProviderQuota: string
  reasonProviderRate: string
  reasonProviderServer: string
  reasonContextOverflow: string
  reasonMissingConfig: string
  reasonModelUnavailable: string
  reasonTargetScope: string
  reasonSuperseded: string
  reasonDeadline: string
  reasonPrepareFailed: string
  reasonSlotWait: string
  reasonTargetBusy: string
  reasonGatewayRestarted: string
}

export const GROUP_REASON_LOCALES = {
  en: {
    reasonMemberUnavailable: 'The agent is unavailable',
    reasonRuntimeOffline: 'The agent is offline',
    reasonQueuedExpired: 'The queued turn expired',
    reasonDeliveryTimeout: 'The reply did not arrive in time',
    reasonAgentBlocked: 'The agent is blocked',
    reasonCancelled: 'The turn was cancelled',
    reasonProviderAuth: 'The provider refused access',
    reasonProviderQuota: 'The provider quota has been reached',
    reasonProviderRate: 'The provider is receiving too many requests',
    reasonProviderServer: 'The provider encountered an error',
    reasonContextOverflow: 'The conversation is too long for the model',
    reasonMissingConfig: 'Required settings are missing',
    reasonModelUnavailable: 'The selected model is unavailable',
    reasonTargetScope: 'The agent could not access its credentials',
    reasonSuperseded: 'A newer message replaced this turn',
    reasonDeadline: 'The turn exceeded its time limit',
    reasonPrepareFailed: 'The agents could not be started',
    reasonSlotWait: 'Too many agents are running',
    reasonTargetBusy: 'The agent is busy',
    reasonGatewayRestarted: 'Hermes on the agent’s machine restarted'
  },
  ja: {
    reasonMemberUnavailable: 'エージェントを利用できません',
    reasonRuntimeOffline: 'エージェントはオフラインです',
    reasonQueuedExpired: '待機中のターンの有効期限が切れました',
    reasonDeliveryTimeout: '時間内に返信が届きませんでした',
    reasonAgentBlocked: 'エージェントはブロックされています',
    reasonCancelled: 'ターンはキャンセルされました',
    reasonProviderAuth: 'プロバイダーがアクセスを拒否しました',
    reasonProviderQuota: 'プロバイダーの利用枠に達しました',
    reasonProviderRate: 'プロバイダーへのリクエストが多すぎます',
    reasonProviderServer: 'プロバイダーでエラーが発生しました',
    reasonContextOverflow: '会話がモデルの長さの上限を超えています',
    reasonMissingConfig: '必要な設定がありません',
    reasonModelUnavailable: '選択したモデルを利用できません',
    reasonTargetScope: 'エージェントが認証情報にアクセスできませんでした',
    reasonSuperseded: '新しいメッセージがこのターンに代わりました',
    reasonDeadline: 'ターンの制限時間を超えました',
    reasonPrepareFailed: 'エージェントを起動できませんでした',
    reasonSlotWait: '実行中のエージェントが多すぎます',
    reasonTargetBusy: 'エージェントは処理中です',
    reasonGatewayRestarted: 'エージェントのマシンの Hermes が再起動しました'
  },
  zh: {
    reasonMemberUnavailable: '智能体不可用',
    reasonRuntimeOffline: '智能体已离线',
    reasonQueuedExpired: '排队的回合已过期',
    reasonDeliveryTimeout: '回复未能及时送达',
    reasonAgentBlocked: '智能体已被阻止运行',
    reasonCancelled: '回合已取消',
    reasonProviderAuth: '服务提供商拒绝了访问',
    reasonProviderQuota: '已达到服务提供商的配额上限',
    reasonProviderRate: '发送给服务提供商的请求过多',
    reasonProviderServer: '服务提供商发生错误',
    reasonContextOverflow: '对话长度超出模型限制',
    reasonMissingConfig: '缺少必要设置',
    reasonModelUnavailable: '所选模型不可用',
    reasonTargetScope: '智能体无法访问其凭据',
    reasonSuperseded: '新消息取代了此回合',
    reasonDeadline: '回合超出了时间限制',
    reasonPrepareFailed: '无法启动智能体',
    reasonSlotWait: '运行中的智能体过多',
    reasonTargetBusy: '智能体正忙',
    reasonGatewayRestarted: '智能体所在机器上的 Hermes 已重启'
  },
  'zh-hant': {
    reasonMemberUnavailable: '智慧體無法使用',
    reasonRuntimeOffline: '智慧體已離線',
    reasonQueuedExpired: '排隊的回合已過期',
    reasonDeliveryTimeout: '回覆未能及時送達',
    reasonAgentBlocked: '智慧體已被阻止執行',
    reasonCancelled: '回合已取消',
    reasonProviderAuth: '服務供應商拒絕了存取',
    reasonProviderQuota: '已達服務供應商的配額上限',
    reasonProviderRate: '傳送給服務供應商的請求過多',
    reasonProviderServer: '服務供應商發生錯誤',
    reasonContextOverflow: '對話長度超出模型限制',
    reasonMissingConfig: '缺少必要設定',
    reasonModelUnavailable: '所選模型無法使用',
    reasonTargetScope: '智慧體無法存取其憑證',
    reasonSuperseded: '新訊息取代了此回合',
    reasonDeadline: '回合超出了時間限制',
    reasonPrepareFailed: '無法啟動智慧體',
    reasonSlotWait: '執行中的智慧體過多',
    reasonTargetBusy: '智慧體正忙',
    reasonGatewayRestarted: '智慧體所在機器上的 Hermes 已重新啟動'
  },
  ar: {
    reasonMemberUnavailable: 'الوكيل غير متاح',
    reasonRuntimeOffline: 'الوكيل غير متصل',
    reasonQueuedExpired: 'انتهت صلاحية الدور المنتظر',
    reasonDeliveryTimeout: 'لم يصل الرد في الوقت المحدد',
    reasonAgentBlocked: 'الوكيل محظور',
    reasonCancelled: 'أُلغي الدور',
    reasonProviderAuth: 'رفض مزوّد الخدمة الوصول',
    reasonProviderQuota: 'تم بلوغ حصة الاستخدام لدى مزوّد الخدمة',
    reasonProviderRate: 'يتلقى مزوّد الخدمة طلبات كثيرة جدًا',
    reasonProviderServer: 'واجه مزوّد الخدمة خطأً',
    reasonContextOverflow: 'المحادثة أطول مما يدعمه النموذج',
    reasonMissingConfig: 'بعض الإعدادات المطلوبة مفقودة',
    reasonModelUnavailable: 'النموذج المحدد غير متاح',
    reasonTargetScope: 'تعذّر على الوكيل الوصول إلى بيانات اعتماده',
    reasonSuperseded: 'حلّت رسالة أحدث محل هذا الدور',
    reasonDeadline: 'تجاوز الدور المهلة المحددة',
    reasonPrepareFailed: 'تعذّر بدء الوكلاء',
    reasonSlotWait: 'هناك وكلاء كثيرة قيد التشغيل',
    reasonTargetBusy: 'الوكيل مشغول',
    reasonGatewayRestarted: 'أُعيد تشغيل Hermes على جهاز الوكيل'
  },
  ru: {
    reasonMemberUnavailable: 'Агент недоступен',
    reasonRuntimeOffline: 'Агент не в сети',
    reasonQueuedExpired: 'Истекло время ожидания хода в очереди',
    reasonDeliveryTimeout: 'Ответ не поступил вовремя',
    reasonAgentBlocked: 'Агент заблокирован',
    reasonCancelled: 'Ход отменён',
    reasonProviderAuth: 'Провайдер отказал в доступе',
    reasonProviderQuota: 'Достигнут лимит использования провайдера',
    reasonProviderRate: 'Провайдер получает слишком много запросов',
    reasonProviderServer: 'У провайдера произошла ошибка',
    reasonContextOverflow: 'Разговор превышает допустимую длину для модели',
    reasonMissingConfig: 'Отсутствуют необходимые настройки',
    reasonModelUnavailable: 'Выбранная модель недоступна',
    reasonTargetScope: 'Агент не смог получить доступ к своим учётным данным',
    reasonSuperseded: 'Более новое сообщение заменило этот ход',
    reasonDeadline: 'Ход превысил лимит времени',
    reasonPrepareFailed: 'Не удалось запустить агентов',
    reasonSlotWait: 'Работает слишком много агентов',
    reasonTargetBusy: 'Агент занят',
    reasonGatewayRestarted: 'Hermes на компьютере агента перезапущен'
  },
  fr: {
    reasonMemberUnavailable: 'L’agent est indisponible',
    reasonRuntimeOffline: 'L’agent est hors ligne',
    reasonQueuedExpired: 'Le tour en attente a expiré',
    reasonDeliveryTimeout: 'La réponse n’est pas arrivée à temps',
    reasonAgentBlocked: 'L’agent est bloqué',
    reasonCancelled: 'Le tour a été annulé',
    reasonProviderAuth: 'Le fournisseur a refusé l’accès',
    reasonProviderQuota: 'Le quota du fournisseur est atteint',
    reasonProviderRate: 'Le fournisseur reçoit trop de requêtes',
    reasonProviderServer: 'Le fournisseur a rencontré une erreur',
    reasonContextOverflow: 'La conversation dépasse la longueur acceptée par le modèle',
    reasonMissingConfig: 'Des paramètres requis sont manquants',
    reasonModelUnavailable: 'Le modèle sélectionné est indisponible',
    reasonTargetScope: 'L’agent n’a pas pu accéder à ses identifiants',
    reasonSuperseded: 'Un message plus récent a remplacé ce tour',
    reasonDeadline: 'Le tour a dépassé le délai autorisé',
    reasonPrepareFailed: 'Les agents n’ont pas pu démarrer',
    reasonSlotWait: 'Trop de agents sont en cours d’exécution',
    reasonTargetBusy: 'L’agent est occupé',
    reasonGatewayRestarted: 'Hermes a redémarré sur la machine de l’agent'
  },
  de: {
    reasonMemberUnavailable: 'Der Agent ist nicht verfügbar',
    reasonRuntimeOffline: 'Der Agent ist offline',
    reasonQueuedExpired: 'Der wartende Durchgang ist abgelaufen',
    reasonDeliveryTimeout: 'Die Antwort ist nicht rechtzeitig eingetroffen',
    reasonAgentBlocked: 'Der Agent ist blockiert',
    reasonCancelled: 'Der Durchgang wurde abgebrochen',
    reasonProviderAuth: 'Der Anbieter hat den Zugriff verweigert',
    reasonProviderQuota: 'Das Kontingent des Anbieters ist aufgebraucht',
    reasonProviderRate: 'Der Anbieter erhält zu viele Anfragen',
    reasonProviderServer: 'Beim Anbieter ist ein Fehler aufgetreten',
    reasonContextOverflow: 'Die Unterhaltung ist zu lang für das Modell',
    reasonMissingConfig: 'Erforderliche Einstellungen fehlen',
    reasonModelUnavailable: 'Das gewählte Modell ist nicht verfügbar',
    reasonTargetScope: 'Der Agent konnte nicht auf seine Zugangsdaten zugreifen',
    reasonSuperseded: 'Eine neuere Nachricht hat diesen Durchgang ersetzt',
    reasonDeadline: 'Der Durchgang hat das Zeitlimit überschritten',
    reasonPrepareFailed: 'Die Agenten konnten nicht gestartet werden',
    reasonSlotWait: 'Es laufen zu viele Agenten',
    reasonTargetBusy: 'Der Agent ist beschäftigt',
    reasonGatewayRestarted: 'Hermes auf dem Rechner des Agenten wurde neu gestartet'
  },
  es: {
    reasonMemberUnavailable: 'El agente no está disponible',
    reasonRuntimeOffline: 'El agente está desconectado',
    reasonQueuedExpired: 'El turno en espera caducó',
    reasonDeliveryTimeout: 'La respuesta no llegó a tiempo',
    reasonAgentBlocked: 'El agente está bloqueado',
    reasonCancelled: 'El turno se canceló',
    reasonProviderAuth: 'El proveedor denegó el acceso',
    reasonProviderQuota: 'Se alcanzó la cuota del proveedor',
    reasonProviderRate: 'El proveedor está recibiendo demasiadas solicitudes',
    reasonProviderServer: 'El proveedor encontró un error',
    reasonContextOverflow: 'La conversación supera la longitud admitida por el modelo',
    reasonMissingConfig: 'Faltan ajustes necesarios',
    reasonModelUnavailable: 'El modelo seleccionado no está disponible',
    reasonTargetScope: 'El agente no pudo acceder a sus credenciales',
    reasonSuperseded: 'Un mensaje más reciente sustituyó este turno',
    reasonDeadline: 'El turno superó el límite de tiempo',
    reasonPrepareFailed: 'No se pudieron iniciar los agentes',
    reasonSlotWait: 'Hay demasiados agentes en ejecución',
    reasonTargetBusy: 'El agente está ocupado',
    reasonGatewayRestarted: 'Hermes se reinició en la máquina del agente'
  }
} satisfies Record<string, GroupReasonMessages>
