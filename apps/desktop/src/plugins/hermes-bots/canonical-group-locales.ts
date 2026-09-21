export interface CanonicalGroupMessages {
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
  cleanupWaiting: string
  cleanupPending: string
  cleanupBlocked: string
  cleanupInventoryLimit: string
  cleanupScanPending: string
  cleanupUnknownWork: string
  cleanupUnknown: string
  outputPending: string
  outputBlocked: string
  outputUnknown: string
  outputMember: string
  outputTask: string
  outputGeneration: string
  historyHeld: string
  membersHeading: string
  memberReady: string
  memberAuthorizationRequired: string
  memberLocalUnavailable: string
  memberAmbiguous: string
  memberFormer: string
  memberUnknown: string
  memberActivate: string
  memberRetire: string
  upgradeChecking: string
  upgradePreparing: string
  upgradeReconnect: string
  upgradeStorage: string
  unknownMember: string

}

export const CANONICAL_GROUP_LOCALES = {
  en: {
    cleanupWaiting: 'Output cleanup is waiting for the original work to stop.',
    cleanupPending: 'Output cleanup is pending.',
    cleanupBlocked: 'Output cleanup is blocked.',
    cleanupInventoryLimit: 'Output cleanup is blocked by the inventory limit.',
    cleanupScanPending: 'Output cleanup scan is pending. More work may remain.',
    cleanupUnknownWork: 'The original work’s outcome is unknown. Output cleanup is waiting.',
    cleanupUnknown: 'Output cleanup status is unavailable.',
    outputPending: 'Output processing is pending.',
    outputBlocked: 'Output processing is blocked.',
    outputUnknown: 'Output processing status is unavailable.',
    outputMember: 'Member:',
    outputTask: 'Task:',
    outputGeneration: 'Generation:',
    historyHeld: 'Held for review:',
    membersHeading: 'Members',
    memberReady: 'Ready for new work',
    memberAuthorizationRequired: 'Authorization required before this Bot can receive new work.',
    memberLocalUnavailable: 'This Bot profile is not available on the owning gateway.',
    memberAmbiguous: 'This Bot profile is ambiguous on the owning gateway.',
    memberFormer: 'Former member — history retained.',
    memberUnknown: 'Member status unavailable.',
    memberActivate: 'Re-add',
    memberRetire: 'Remove from new work',
    upgradeChecking: 'Checking this Group Chat…',
    upgradePreparing: 'Finishing this Group Chat upgrade. Its history and members will appear here automatically.',
    upgradeReconnect: 'Reconnect this Group Chat’s original connection to continue. Its history and members are still here.',
    upgradeStorage: 'Hermes could not save this Group Chat before upgrading it. Free some storage and restart; the original history was kept.',
    unknownMember: 'Unknown member',

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
    cleanupWaiting: '元の処理が停止するまで出力のクリーンアップを待機しています。',
    cleanupPending: '出力のクリーンアップは保留中です。',
    cleanupBlocked: '出力のクリーンアップはブロックされています。',
    cleanupInventoryLimit: '出力一覧の上限によりクリーンアップがブロックされています。',
    cleanupScanPending: 'クリーンアップ対象の走査は保留中です。対象がまだ残っている可能性があります。',
    cleanupUnknownWork: '元の処理の結果は不明です。出力のクリーンアップを待機しています。',
    cleanupUnknown: '出力のクリーンアップ状況を取得できません。',
    outputPending: '出力処理は保留中です。',
    outputBlocked: '出力処理はブロックされています。',
    outputUnknown: '出力処理の状況を取得できません。',
    outputMember: 'メンバー：',
    outputTask: 'タスク：',
    outputGeneration: '世代：',
    historyHeld: '確認が必要な保留：',
    membersHeading: 'メンバー',
    memberReady: '新しい作業を受け取れます',
    memberAuthorizationRequired: 'このBotが新しい作業を受け取るには認証が必要です。',
    memberLocalUnavailable: 'このBotプロフィールは管理元のゲートウェイで利用できません。',
    memberAmbiguous: '管理元のゲートウェイでこのBotプロフィールを一意に特定できません。',
    memberFormer: '以前のメンバー — 履歴は保持されています。',
    memberUnknown: 'メンバーの状態を取得できません。',
    memberActivate: '再追加',
    memberRetire: '新しい作業から外す',
    upgradeChecking: 'このグループチャットを確認しています…',
    upgradePreparing: 'このグループチャットの更新を完了しています。履歴とメンバーは自動的にここへ表示されます。',
    upgradeReconnect: '続行するには、このグループチャットの元の接続を再接続してください。履歴とメンバーは保持されています。',
    upgradeStorage: '更新前にこのグループチャットを保存できませんでした。空き容量を増やして再起動してください。元の履歴は保持されています。',
    unknownMember: '不明なメンバー',

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
    cleanupWaiting: '输出清理正在等待原任务停止。',
    cleanupPending: '输出清理待处理。',
    cleanupBlocked: '输出清理受阻。',
    cleanupInventoryLimit: '输出清理因清单数量上限而受阻。',
    cleanupScanPending: '清理扫描待处理，可能仍有未列出的任务。',
    cleanupUnknownWork: '原任务的结果未知，输出清理正在等待。',
    cleanupUnknown: '输出清理状态不可用。',
    outputPending: '输出处理待进行。',
    outputBlocked: '输出处理受阻。',
    outputUnknown: '输出处理状态不可用。',
    outputMember: '成员：',
    outputTask: '任务：',
    outputGeneration: '执行代次：',
    historyHeld: '待审核：',
    membersHeading: '成员',
    memberReady: '可接收新任务',
    memberAuthorizationRequired: '此 Bot 需要授权后才能接收新任务。',
    memberLocalUnavailable: '所属网关上没有此 Bot 配置。',
    memberAmbiguous: '所属网关上无法唯一确定此 Bot 配置。',
    memberFormer: '曾经的成员 — 历史记录已保留。',
    memberUnknown: '成员状态不可用。',
    memberActivate: '重新加入',
    memberRetire: '不再接收新任务',
    upgradeChecking: '正在检查此群聊…',
    upgradePreparing: '正在完成此群聊的升级。历史记录和成员将自动显示在这里。',
    upgradeReconnect: '请重新连接此群聊原来的连接以继续。历史记录和成员仍已保留。',
    upgradeStorage: '升级前无法保存此群聊。请释放一些存储空间并重启；原历史记录已保留。',
    unknownMember: '未知成员',

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
    cleanupWaiting: '輸出清理正在等待原工作停止。',
    cleanupPending: '輸出清理待處理。',
    cleanupBlocked: '輸出清理受阻。',
    cleanupInventoryLimit: '輸出清理因清單數量上限而受阻。',
    cleanupScanPending: '清理掃描待處理，可能仍有未列出的工作。',
    cleanupUnknownWork: '原工作的結果未知，輸出清理正在等待。',
    cleanupUnknown: '輸出清理狀態無法取得。',
    outputPending: '輸出處理待進行。',
    outputBlocked: '輸出處理受阻。',
    outputUnknown: '輸出處理狀態無法取得。',
    outputMember: '成員：',
    outputTask: '工作：',
    outputGeneration: '執行代次：',
    historyHeld: '待檢視：',
    membersHeading: '成員',
    memberReady: '可接收新工作',
    memberAuthorizationRequired: '此 Bot 需要授權後才能接收新工作。',
    memberLocalUnavailable: '所屬閘道上沒有此 Bot 設定檔。',
    memberAmbiguous: '所屬閘道上無法唯一辨識此 Bot 設定檔。',
    memberFormer: '先前的成員 — 歷史記錄已保留。',
    memberUnknown: '成員狀態無法取得。',
    memberActivate: '重新加入',
    memberRetire: '不再接收新工作',
    upgradeChecking: '正在檢查此群組聊天…',
    upgradePreparing: '正在完成此群組聊天的升級。歷史記錄和成員會自動顯示在這裡。',
    upgradeReconnect: '請重新連線此群組聊天原本的連線以繼續。歷史記錄和成員仍已保留。',
    upgradeStorage: '升級前無法儲存此群組聊天。請釋放一些儲存空間並重新啟動；原始歷史記錄已保留。',
    unknownMember: '未知成員',

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
    cleanupWaiting: 'تنظيف المخرجات ينتظر توقف العمل الأصلي.',
    cleanupPending: 'تنظيف المخرجات قيد الانتظار.',
    cleanupBlocked: 'تنظيف المخرجات محظور.',
    cleanupInventoryLimit: 'تنظيف المخرجات محظور بسبب حد الجرد.',
    cleanupScanPending: 'فحص عناصر التنظيف قيد الانتظار. قد يتبقى عمل آخر.',
    cleanupUnknownWork: 'نتيجة العمل الأصلي غير معروفة. تنظيف المخرجات قيد الانتظار.',
    cleanupUnknown: 'حالة تنظيف المخرجات غير متاحة.',
    outputPending: 'معالجة المخرجات قيد الانتظار.',
    outputBlocked: 'معالجة المخرجات محظورة.',
    outputUnknown: 'حالة معالجة المخرجات غير متاحة.',
    outputMember: 'العضو:',
    outputTask: 'المهمة:',
    outputGeneration: 'جيل التنفيذ:',
    historyHeld: 'معلّق للمراجعة:',
    membersHeading: 'الأعضاء',
    memberReady: 'جاهز للعمل الجديد',
    memberAuthorizationRequired: 'يحتاج هذا الروبوت إلى تفويض قبل استلام عمل جديد.',
    memberLocalUnavailable: 'ملف هذا الروبوت غير متاح على البوابة المالكة.',
    memberAmbiguous: 'لا يمكن تحديد ملف هذا الروبوت بشكل فريد على البوابة المالكة.',
    memberFormer: 'عضو سابق — تم الاحتفاظ بالسجل.',
    memberUnknown: 'حالة العضو غير متاحة.',
    memberActivate: 'إعادة الإضافة',
    memberRetire: 'إزالته من العمل الجديد',
    upgradeChecking: 'جارٍ التحقق من هذه المحادثة الجماعية…',
    upgradePreparing: 'يتم إكمال ترقية هذه المحادثة الجماعية. سيظهر السجل والأعضاء هنا تلقائيًا.',
    upgradeReconnect: 'أعد توصيل الاتصال الأصلي لهذه المحادثة الجماعية للمتابعة. ما زال السجل والأعضاء محفوظين.',
    upgradeStorage: 'تعذر حفظ هذه المحادثة الجماعية قبل ترقيتها. حرر بعض مساحة التخزين وأعد التشغيل؛ تم الاحتفاظ بالسجل الأصلي.',
    unknownMember: 'عضو غير معروف',

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
    cleanupWaiting: 'Очистка вывода ожидает остановки исходной работы.',
    cleanupPending: 'Очистка вывода ожидается.',
    cleanupBlocked: 'Очистка вывода заблокирована.',
    cleanupInventoryLimit: 'Очистка вывода заблокирована лимитом списка объектов.',
    cleanupScanPending: 'Сканирование для очистки ожидается. Может оставаться другая работа.',
    cleanupUnknownWork: 'Результат исходной работы неизвестен. Очистка вывода ожидает.',
    cleanupUnknown: 'Статус очистки вывода недоступен.',
    outputPending: 'Обработка вывода ожидается.',
    outputBlocked: 'Обработка вывода заблокирована.',
    outputUnknown: 'Статус обработки вывода недоступен.',
    outputMember: 'Участник:',
    outputTask: 'Задача:',
    outputGeneration: 'Поколение:',
    historyHeld: 'Ожидает проверки:',
    membersHeading: 'Участники',
    memberReady: 'Готов к новой работе',
    memberAuthorizationRequired: 'Нужна авторизация, прежде чем этот бот сможет получать новую работу.',
    memberLocalUnavailable: 'Профиль этого бота недоступен на управляющем шлюзе.',
    memberAmbiguous: 'Профиль этого бота неоднозначен на управляющем шлюзе.',
    memberFormer: 'Бывший участник — история сохранена.',
    memberUnknown: 'Статус участника недоступен.',
    memberActivate: 'Добавить снова',
    memberRetire: 'Убрать из новой работы',
    upgradeChecking: 'Проверка этого группового чата…',
    upgradePreparing: 'Завершается обновление этого группового чата. История и участники появятся здесь автоматически.',
    upgradeReconnect: 'Переподключите исходное соединение этого группового чата. История и участники сохранены.',
    upgradeStorage: 'Не удалось сохранить групповой чат перед обновлением. Освободите место и перезапустите Hermes; исходная история сохранена.',
    unknownMember: 'Неизвестный участник',

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
  }
} satisfies Record<string, CanonicalGroupMessages>
