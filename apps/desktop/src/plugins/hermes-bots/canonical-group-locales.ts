export interface CanonicalGroupMessages {
  legacyRoom: string
  checkingDriver: string
  startGatewayGroup: string
  classicCount: string
  classicConnection: string
  classicMembers: string
  createRefused: string
  hostedProfileOwners: string
  refreshGroups: string
  loadingGroup: string
  loadingGroups: string
  emptyGroups: string
  driverUnavailable: string
  invalidLogCursor: string
  allowOnce: string
  deny: string
  discardUnknown: string
  discardWarning: string
  confirmDiscard: string
  unconfirmedSend: string
  restoredPendingSend: string
  groupMessage: string
  attachFiles: string
  removeAttachment: string
  uploadFailed: string
}

export const HOSTED_PROFILE_OWNERS_URL = 'https://hermes-agent.nousresearch.com/docs/developer-guide/hosted-profile-owners'

export const CANONICAL_GROUP_LOCALES = {
  en: {
    legacyRoom: 'This is a legacy Desktop room. Start a gateway-owned group with these members; the old history stays here and is not replayed.',
    checkingDriver: 'Checking group driver…',
    startGatewayGroup: 'Start gateway group',
    classicCount: 'This roster stays classic: gateway groups need two to six Bots.',
    classicConnection: 'This roster stays classic: it includes Bots on another connection.',
    classicMembers: 'This roster stays classic: gateway groups need unique profiles and unique, non-reserved handles.',
    createRefused: 'The gateway refused to create this group. On a default install, list each Bot under hosted_rooms.profiles in the owning gateway config, then try again.',
    hostedProfileOwners: 'Hosted profile configuration guide',
    refreshGroups: 'Refresh gateway groups',
    loadingGroup: 'Loading group…',
    loadingGroups: 'Loading gateway groups…',
    emptyGroups: 'No gateway groups found.',
    driverUnavailable: 'Group driver unavailable. Update or reconnect the owning gateway.',
    invalidLogCursor: 'Invalid room log cursor',
    allowOnce: 'Allow once',
    deny: 'Deny',
    discardUnknown: 'Discard unknown work',
    discardWarning: 'Side effects may already have occurred. Discarding does not undo them.',
    confirmDiscard: 'Confirm discard',
    unconfirmedSend: 'The previous send is unconfirmed. Retry its original text before sending another message.',
    restoredPendingSend: 'An unconfirmed send was restored. Retry it before sending another message.',
    groupMessage: 'Group message',
    attachFiles: 'Attach files',
    removeAttachment: 'Remove attachment',
    uploadFailed: 'Upload failed'
  },
  ja: {
    legacyRoom: 'これは従来のDesktopルームです。このメンバーでゲートウェイ管理のグループを開始できます。過去の履歴はここに残り、再実行されません。',
    checkingDriver: 'グループの実行機能を確認中…',
    startGatewayGroup: 'ゲートウェイのグループを開始',
    classicCount: 'このメンバー構成は従来方式のままです。ゲートウェイのグループには2〜6体のBotが必要です。',
    classicConnection: '別の接続上のBotが含まれるため、このメンバー構成は従来方式のままです。',
    classicMembers: 'このメンバー構成は従来方式のままです。ゲートウェイのグループには重複しないプロフィールと、予約語ではない固有のハンドルが必要です。',
    createRefused: 'ゲートウェイがグループの作成を拒否しました。標準構成では、管理元のゲートウェイ設定のhosted_rooms.profilesに各Botを登録してから、再試行してください。',
    hostedProfileOwners: 'ホストするプロフィールの設定ガイド',
    refreshGroups: 'ゲートウェイのグループを更新',
    loadingGroup: 'グループを読み込み中…',
    loadingGroups: 'ゲートウェイのグループを読み込み中…',
    emptyGroups: 'ゲートウェイのグループが見つかりません。',
    driverUnavailable: 'グループの実行機能を利用できません。管理元のゲートウェイを更新するか、再接続してください。',
    invalidLogCursor: 'ルームのログカーソルが無効です',
    allowOnce: '今回のみ許可',
    deny: '拒否',
    discardUnknown: '結果不明の処理を破棄',
    discardWarning: 'すでに変更が行われている可能性があります。破棄しても元には戻りません。',
    confirmDiscard: '破棄を確定',
    unconfirmedSend: '前回の送信は未確認です。別のメッセージを送信する前に、元のテキストで再試行してください。',
    restoredPendingSend: '未確認の送信を復元しました。別のメッセージを送信する前に再試行してください。',
    groupMessage: 'グループメッセージ',
    attachFiles: 'ファイルを添付',
    removeAttachment: '添付ファイルを削除',
    uploadFailed: 'アップロードに失敗しました'
  },
  zh: {
    legacyRoom: '这是旧版Desktop群组。可使用这些成员创建由网关管理的群组；旧记录会保留在此处，不会重新执行。',
    checkingDriver: '正在检查群组运行程序…',
    startGatewayGroup: '启动网关群组',
    classicCount: '此成员组合保留经典模式：网关群组需要2至6个Bot。',
    classicConnection: '此成员组合保留经典模式：其中包含其他连接上的Bot。',
    classicMembers: '此成员组合保留经典模式：网关群组需要不同的配置档和唯一且非保留的昵称。',
    createRefused: '网关拒绝创建此群组。默认安装下，请在所属网关配置的hosted_rooms.profiles中列出每个Bot，然后重试。',
    hostedProfileOwners: '托管配置档设置指南',
    refreshGroups: '刷新网关群组',
    loadingGroup: '正在加载群组…',
    loadingGroups: '正在加载网关群组…',
    emptyGroups: '未找到网关群组。',
    driverUnavailable: '群组运行程序不可用。请更新或重新连接所属网关。',
    invalidLogCursor: '群组日志游标无效',
    allowOnce: '仅允许一次',
    deny: '拒绝',
    discardUnknown: '丢弃结果未知的任务',
    discardWarning: '可能已产生实际影响。丢弃任务不会撤销这些影响。',
    confirmDiscard: '确认丢弃',
    unconfirmedSend: '上次发送尚未确认。请先重试发送原始文本，再发送其他消息。',
    restoredPendingSend: '已恢复尚未确认的发送。请先重试，再发送其他消息。',
    groupMessage: '群组消息',
    attachFiles: '附加文件',
    removeAttachment: '移除附件',
    uploadFailed: '上传失败'
  },
  'zh-hant': {
    legacyRoom: '這是舊版Desktop群組。可使用這些成員建立由閘道管理的群組；舊記錄會保留在此處，不會重新執行。',
    checkingDriver: '正在檢查群組執行程式…',
    startGatewayGroup: '啟動閘道群組',
    classicCount: '此成員組合保留經典模式：閘道群組需要2至6個Bot。',
    classicConnection: '此成員組合保留經典模式：其中包含其他連線上的Bot。',
    classicMembers: '此成員組合保留經典模式：閘道群組需要不同的設定檔和唯一且非保留的暱稱。',
    createRefused: '閘道拒絕建立此群組。預設安裝下，請在所屬閘道設定的hosted_rooms.profiles中列出每個Bot，然後重試。',
    hostedProfileOwners: '託管設定檔設定指南',
    refreshGroups: '重新整理閘道群組',
    loadingGroup: '正在載入群組…',
    loadingGroups: '正在載入閘道群組…',
    emptyGroups: '找不到閘道群組。',
    driverUnavailable: '群組執行程式無法使用。請更新或重新連線至所屬閘道。',
    invalidLogCursor: '群組記錄游標無效',
    allowOnce: '僅允許一次',
    deny: '拒絕',
    discardUnknown: '捨棄結果不明的工作',
    discardWarning: '可能已產生實際影響。捨棄工作不會復原這些影響。',
    confirmDiscard: '確認捨棄',
    unconfirmedSend: '上次傳送尚未確認。請先重試傳送原始文字，再傳送其他訊息。',
    restoredPendingSend: '已還原尚未確認的傳送。請先重試，再傳送其他訊息。',
    groupMessage: '群組訊息',
    attachFiles: '附加檔案',
    removeAttachment: '移除附件',
    uploadFailed: '上傳失敗'
  },
  ar: {
    legacyRoom: 'هذه غرفة Desktop قديمة. ابدأ مجموعة تديرها البوابة بهؤلاء الأعضاء؛ يبقى السجل القديم هنا ولا يُعاد تشغيله.',
    checkingDriver: 'جارٍ التحقق من مشغّل المجموعة…',
    startGatewayGroup: 'بدء مجموعة البوابة',
    classicCount: 'تبقى هذه التشكيلة بالنمط الكلاسيكي: تحتاج مجموعات البوابة إلى بوتين إلى ستة بوتات.',
    classicConnection: 'تبقى هذه التشكيلة بالنمط الكلاسيكي: تتضمن بوتات على اتصال آخر.',
    classicMembers: 'تبقى هذه التشكيلة بالنمط الكلاسيكي: تحتاج مجموعات البوابة إلى ملفات تعريف فريدة وأسماء مستخدم فريدة وغير محجوزة.',
    createRefused: 'رفضت البوابة إنشاء هذه المجموعة. في التثبيت الافتراضي، أدرج كل بوت ضمن hosted_rooms.profiles في إعدادات البوابة المالكة، ثم أعد المحاولة.',
    hostedProfileOwners: 'دليل إعداد ملفات التعريف المستضافة',
    refreshGroups: 'تحديث مجموعات البوابة',
    loadingGroup: 'جارٍ تحميل المجموعة…',
    loadingGroups: 'جارٍ تحميل مجموعات البوابة…',
    emptyGroups: 'لم يتم العثور على مجموعات في البوابة.',
    driverUnavailable: 'مشغّل المجموعة غير متاح. حدّث البوابة المالكة أو أعد الاتصال بها.',
    invalidLogCursor: 'مؤشر سجل الغرفة غير صالح',
    allowOnce: 'السماح مرة واحدة',
    deny: 'رفض',
    discardUnknown: 'تجاهل العمل ذي النتيجة المجهولة',
    discardWarning: 'قد تكون تغييرات قد حدثت بالفعل. تجاهل العمل لا يتراجع عنها.',
    confirmDiscard: 'تأكيد التجاهل',
    unconfirmedSend: 'الإرسال السابق غير مؤكّد. أعد المحاولة بالنص الأصلي قبل إرسال رسالة أخرى.',
    restoredPendingSend: 'تمت استعادة إرسال غير مؤكّد. أعد محاولته قبل إرسال رسالة أخرى.',
    groupMessage: 'رسالة المجموعة',
    attachFiles: 'إرفاق ملفات',
    removeAttachment: 'إزالة المرفق',
    uploadFailed: 'فشل الرفع'
  },
  ru: {
    legacyRoom: 'Это старая комната Desktop. Создайте группу под управлением шлюза с этими участниками; прежняя история останется здесь и не будет выполнена заново.',
    checkingDriver: 'Проверка исполнителя группы…',
    startGatewayGroup: 'Создать группу шлюза',
    classicCount: 'Эта группа останется классической: группе шлюза нужны от двух до шести ботов.',
    classicConnection: 'Эта группа останется классической: в ней есть боты на другом подключении.',
    classicMembers: 'Эта группа останется классической: группе шлюза нужны уникальные профили и уникальные незарезервированные имена пользователей.',
    createRefused: 'Шлюз отказался создать группу. При стандартной установке укажите каждого бота в hosted_rooms.profiles в конфигурации шлюза-владельца и повторите попытку.',
    hostedProfileOwners: 'Руководство по настройке размещённых профилей',
    refreshGroups: 'Обновить группы шлюза',
    loadingGroup: 'Загрузка группы…',
    loadingGroups: 'Загрузка групп шлюза…',
    emptyGroups: 'Группы шлюза не найдены.',
    driverUnavailable:
      'Исполнитель группы недоступен. Обновите шлюз, которому принадлежит группа, или подключитесь к нему заново.',
    invalidLogCursor: 'Недопустимый курсор журнала комнаты',
    allowOnce: 'Разрешить один раз',
    deny: 'Отклонить',
    discardUnknown: 'Отбросить работу с неизвестным результатом',
    discardWarning: 'Изменения могли уже произойти. Отмена работы не отменяет эти изменения.',
    confirmDiscard: 'Подтвердить отмену работы',
    unconfirmedSend:
      'Предыдущая отправка не подтверждена. Повторите её с исходным текстом, прежде чем отправлять другое сообщение.',
    restoredPendingSend:
      'Восстановлена неподтверждённая отправка. Повторите её, прежде чем отправлять другое сообщение.',
    groupMessage: 'Сообщение группе',
    attachFiles: 'Прикрепить файлы',
    removeAttachment: 'Удалить вложение',
    uploadFailed: 'Не удалось загрузить файл'
  },
  fr: {
    legacyRoom: 'Ceci est un ancien salon Desktop. Démarrez un groupe géré par la passerelle avec ces membres ; l’ancien historique reste ici et n’est pas rejoué.',
    checkingDriver: 'Vérification du pilote de groupe…',
    startGatewayGroup: 'Démarrer un groupe de passerelle',
    classicCount: 'Ce groupe reste classique : les groupes de passerelle nécessitent deux à six Bots.',
    classicConnection: 'Ce groupe reste classique : il inclut des Bots sur une autre connexion.',
    classicMembers: 'Ce groupe reste classique : les groupes de passerelle nécessitent des profils uniques et des identifiants uniques non réservés.',
    createRefused: 'La passerelle a refusé de créer ce groupe. Sur une installation par défaut, indiquez chaque Bot dans hosted_rooms.profiles dans la configuration de la passerelle propriétaire, puis réessayez.',
    hostedProfileOwners: 'Guide de configuration des profils hébergés',
    refreshGroups: 'Actualiser les groupes de la passerelle',
    loadingGroup: 'Chargement du groupe…',
    loadingGroups: 'Chargement des groupes de la passerelle…',
    emptyGroups: 'Aucun groupe de passerelle trouvé.',
    driverUnavailable:
      'Pilote de groupe indisponible. Mettez à jour ou reconnectez la passerelle propriétaire.',
    invalidLogCursor: 'Curseur de journal de salon invalide',
    allowOnce: 'Autoriser une fois',
    deny: 'Refuser',
    discardUnknown: 'Abandonner le travail au résultat inconnu',
    discardWarning: 'Des effets de bord ont peut-être déjà eu lieu. Abandonner ne les annule pas.',
    confirmDiscard: "Confirmer l'abandon",
    unconfirmedSend:
      "L'envoi précédent n'est pas confirmé. Réessayez son texte d'origine avant d'envoyer un autre message.",
    restoredPendingSend: 'Un envoi non confirmé a été restauré. Réessayez-le avant d\'envoyer un autre message.',
    groupMessage: 'Message de groupe',
    attachFiles: 'Joindre des fichiers',
    removeAttachment: 'Retirer la pièce jointe',
    uploadFailed: 'Échec du téléversement'
  },
  de: {
    legacyRoom: 'Dies ist ein alter Desktop-Raum. Starten Sie mit diesen Mitgliedern eine vom Gateway verwaltete Gruppe; der bisherige Verlauf bleibt hier und wird nicht erneut ausgeführt.',
    checkingDriver: 'Gruppentreiber wird geprüft…',
    startGatewayGroup: 'Gateway-Gruppe starten',
    classicCount: 'Diese Gruppe bleibt klassisch: Gateway-Gruppen benötigen zwei bis sechs Bots.',
    classicConnection: 'Diese Gruppe bleibt klassisch: Sie enthält Bots auf einer anderen Verbindung.',
    classicMembers: 'Diese Gruppe bleibt klassisch: Gateway-Gruppen benötigen eindeutige Profile und eindeutige, nicht reservierte Nutzernamen.',
    createRefused: 'Das Gateway hat die Erstellung dieser Gruppe abgelehnt. Tragen Sie bei einer Standardinstallation jeden Bot unter hosted_rooms.profiles in der Konfiguration des zuständigen Gateways ein und versuchen Sie es erneut.',
    hostedProfileOwners: 'Anleitung zur Konfiguration gehosteter Profile',
    refreshGroups: 'Gateway-Gruppen aktualisieren',
    loadingGroup: 'Gruppe wird geladen…',
    loadingGroups: 'Gateway-Gruppen werden geladen…',
    emptyGroups: 'Keine Gateway-Gruppen gefunden.',
    driverUnavailable:
      'Gruppentreiber nicht verfügbar. Aktualisieren Sie das zuständige Gateway oder verbinden Sie es neu.',
    invalidLogCursor: 'Ungültiger Raumprotokoll-Cursor',
    allowOnce: 'Einmal erlauben',
    deny: 'Ablehnen',
    discardUnknown: 'Arbeit mit unbekanntem Ergebnis verwerfen',
    discardWarning: 'Nebenwirkungen können bereits eingetreten sein. Verwerfen macht sie nicht rückgängig.',
    confirmDiscard: 'Verwerfen bestätigen',
    unconfirmedSend:
      'Der vorherige Versand ist unbestätigt. Wiederholen Sie seinen ursprünglichen Text, bevor Sie eine weitere Nachricht senden.',
    restoredPendingSend:
      'Ein unbestätigter Versand wurde wiederhergestellt. Wiederholen Sie ihn, bevor Sie eine weitere Nachricht senden.',
    groupMessage: 'Gruppennachricht',
    attachFiles: 'Dateien anhängen',
    removeAttachment: 'Anhang entfernen',
    uploadFailed: 'Hochladen fehlgeschlagen'
  },
  es: {
    legacyRoom: 'Esta es una sala Desktop antigua. Inicia un grupo gestionado por la pasarela con estos miembros; el historial anterior permanece aquí y no se vuelve a ejecutar.',
    checkingDriver: 'Comprobando el controlador de grupo…',
    startGatewayGroup: 'Iniciar grupo de pasarela',
    classicCount: 'Este grupo sigue siendo clásico: los grupos de pasarela necesitan entre dos y seis Bots.',
    classicConnection: 'Este grupo sigue siendo clásico: incluye Bots en otra conexión.',
    classicMembers: 'Este grupo sigue siendo clásico: los grupos de pasarela necesitan perfiles únicos e identificadores únicos no reservados.',
    createRefused: 'La pasarela rechazó crear este grupo. En una instalación predeterminada, incluye cada Bot en hosted_rooms.profiles en la configuración de la pasarela propietaria y vuelve a intentarlo.',
    hostedProfileOwners: 'Guía de configuración de perfiles alojados',
    refreshGroups: 'Actualizar grupos de la pasarela',
    loadingGroup: 'Cargando grupo…',
    loadingGroups: 'Cargando grupos de la pasarela…',
    emptyGroups: 'No se encontraron grupos de la pasarela.',
    driverUnavailable:
      'Controlador de grupo no disponible. Actualiza o vuelve a conectar la pasarela propietaria.',
    invalidLogCursor: 'Cursor de registro de sala no válido',
    allowOnce: 'Permitir una vez',
    deny: 'Denegar',
    discardUnknown: 'Descartar trabajo con resultado desconocido',
    discardWarning: 'Puede que ya se hayan producido efectos secundarios. Descartar no los deshace.',
    confirmDiscard: 'Confirmar descarte',
    unconfirmedSend:
      'El envío anterior no está confirmado. Reintenta su texto original antes de enviar otro mensaje.',
    restoredPendingSend: 'Se restauró un envío sin confirmar. Reintántalo antes de enviar otro mensaje.',
    groupMessage: 'Mensaje de grupo',
    attachFiles: 'Adjuntar archivos',
    removeAttachment: 'Quitar adjunto',
    uploadFailed: 'Error al subir'
  }
} satisfies Record<string, CanonicalGroupMessages>
