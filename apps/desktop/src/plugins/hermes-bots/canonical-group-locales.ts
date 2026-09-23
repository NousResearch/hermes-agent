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
}

export const CANONICAL_GROUP_LOCALES = {
  en: {
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
