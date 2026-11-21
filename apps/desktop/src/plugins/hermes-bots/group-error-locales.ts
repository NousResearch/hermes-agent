import type { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'

export interface GroupChatErrorMessages {
  sendRouteRequired: string
  savedMessagesUnreadable: string
  savedMessageUnreadable: string
  sendNeedsUpdate: string
  attachmentTooLarge: (name: string) => string
  attachmentFallback: string
  pastedImage: string
}

export const GROUP_CHAT_ERRORS = {
  'en': {
    sendRouteRequired: "Sending to this group chat requires its connection, profile and group chat ID",
    savedMessagesUnreadable: "Saved group chat messages could not be read",
    savedMessageUnreadable: "A saved group chat message could not be read",
    sendNeedsUpdate: "Sending to this group chat needs an updated Desktop; restart Hermes Desktop",
    attachmentTooLarge: name => `${name}: too large (max 15MB).`,
    attachmentFallback: "attachment",
    pastedImage: "pasted image"
  },
  'ja': {
    sendRouteRequired: "このグループチャットへの送信には接続、プロファイル、グループチャットIDが必要です",
    savedMessagesUnreadable: "保存されたグループチャットのメッセージを読み込めませんでした",
    savedMessageUnreadable: "保存されたグループチャットのメッセージを読み込めませんでした",
    sendNeedsUpdate: "グループチャットへの送信にはDesktopの更新が必要です。Hermes Desktopを再起動してください",
    attachmentTooLarge: name => `${name}：サイズが大きすぎます（最大15MB）。`,
    attachmentFallback: "添付ファイル",
    pastedImage: "貼り付けた画像"
  },
  'zh': {
    sendRouteRequired: "向此群聊发送消息需要其连接、配置档和群聊标识",
    savedMessagesUnreadable: "无法读取已保存的群聊消息",
    savedMessageUnreadable: "无法读取一条已保存的群聊消息",
    sendNeedsUpdate: "群聊发送需要新版 Desktop，请重启 Hermes Desktop",
    attachmentTooLarge: name => `${name}：文件过大（最大 15MB）。`,
    attachmentFallback: "附件",
    pastedImage: "粘贴的图片"
  },
  'zh-hant': {
    sendRouteRequired: "向此群組聊天傳送訊息需要其連線、設定檔和群組聊天識別碼",
    savedMessagesUnreadable: "無法讀取已儲存的群組聊天訊息",
    savedMessageUnreadable: "無法讀取一則已儲存的群組聊天訊息",
    sendNeedsUpdate: "群組聊天傳送需要新版 Desktop，請重新啟動 Hermes Desktop",
    attachmentTooLarge: name => `${name}：檔案過大（最大 15MB）。`,
    attachmentFallback: "附件",
    pastedImage: "貼上的圖片"
  },
  'fr': {
    sendRouteRequired: "L’envoi dans cette discussion de groupe nécessite sa connexion, son profil et son identifiant",
    savedMessagesUnreadable: "Impossible de lire les messages enregistrés de la discussion de groupe",
    savedMessageUnreadable: "Impossible de lire un message enregistré de la discussion de groupe",
    sendNeedsUpdate: "L’envoi dans un groupe nécessite Desktop à jour ; redémarrez Hermes Desktop",
    attachmentTooLarge: name => `${name} : fichier trop volumineux (15 Mo maximum).`,
    attachmentFallback: "pièce jointe",
    pastedImage: "image collée"
  },
  'de': {
    sendRouteRequired: "Zum Senden an diesen Gruppenchat werden Verbindung, Profil und Gruppenchat-ID benötigt",
    savedMessagesUnreadable: "Gespeicherte Gruppenchat-Nachrichten konnten nicht gelesen werden",
    savedMessageUnreadable: "Eine gespeicherte Gruppenchat-Nachricht konnte nicht gelesen werden",
    sendNeedsUpdate: "Für das Senden an Gruppen wird ein aktuelles Desktop benötigt; starte Hermes Desktop neu",
    attachmentTooLarge: name => `${name}: zu groß (max. 15 MB).`,
    attachmentFallback: "Anhang",
    pastedImage: "eingefügtes Bild"
  },
  'es': {
    sendRouteRequired: "Para enviar a este chat de grupo se necesitan su conexión, perfil e identificador",
    savedMessagesUnreadable: "No se pudieron leer los mensajes guardados del chat de grupo",
    savedMessageUnreadable: "No se pudo leer un mensaje guardado del chat de grupo",
    sendNeedsUpdate: "El envío al grupo necesita una versión actualizada de Desktop; reinicia Hermes Desktop",
    attachmentTooLarge: name => `${name}: demasiado grande (máximo 15 MB).`,
    attachmentFallback: "adjunto",
    pastedImage: "imagen pegada"
  },
  'ru': {
    sendRouteRequired: "Для отправки нужны подключение, профиль и идентификатор группового чата",
    savedMessagesUnreadable: "Не удалось прочитать сохранённые сообщения группового чата",
    savedMessageUnreadable: "Не удалось прочитать сохранённое сообщение группового чата",
    sendNeedsUpdate: "Для отправки в группу нужна обновлённая версия Desktop; перезапустите Hermes Desktop",
    attachmentTooLarge: name => `${name}: слишком большой файл (максимум 15 МБ).`,
    attachmentFallback: "вложение",
    pastedImage: "вставленное изображение"
  },
  'ar': {
    sendRouteRequired: "يتطلب الإرسال الاتصال والملف الشخصي ومعرّف محادثة المجموعة",
    savedMessagesUnreadable: "تعذّر قراءة رسائل محادثة المجموعة المحفوظة",
    savedMessageUnreadable: "تعذّر قراءة رسالة محفوظة في محادثة المجموعة",
    sendNeedsUpdate: "يتطلب الإرسال للمجموعة إصدارًا محدّثًا من Desktop؛ أعد تشغيل Hermes Desktop",
    attachmentTooLarge: name => `${name}: حجمه كبير جدًا (الحد الأقصى 15 ميغابايت).`,
    attachmentFallback: "مرفق",
    pastedImage: "صورة ملصقة"
  },

  cs: {
    sendRouteRequired: 'Odeslání do tohoto skupinového chatu vyžaduje jeho připojení, profil a ID skupinového chatu',
    savedMessagesUnreadable: 'Uložené zprávy skupinového chatu se nepodařilo načíst',
    savedMessageUnreadable: 'Uloženou zprávu skupinového chatu se nepodařilo načíst',
    sendNeedsUpdate: 'Odesílání do tohoto skupinového chatu vyžaduje novější Desktop; restartujte Hermes Desktop',
    attachmentTooLarge: (name) => `${name}: příliš velké (max. 15 MB).`,
    attachmentFallback: 'příloha',
    pastedImage: 'vložený obrázek',
  }
} satisfies Record<keyof typeof CANONICAL_GROUP_LOCALES, GroupChatErrorMessages>
