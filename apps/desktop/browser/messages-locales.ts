import type { BundledLocale } from '../src/i18n/types'

export interface BrowserMessages {
  attachmentSize: string
  attachmentCount: string
  attachmentExpired: string
  desktopOnly: string
  backendTab: string
  invalidRoute: string
  httpLinks: string
  terminalTimeout: string
  terminalFailed: string
  terminalClosed: string
  oauthCancelled: string
  oauthMissingFlow: string
  oauthTimeout: string
  oauthFailed: string
}

export const browserMessages = {
  "en": {
    "attachmentSize": "Attachments must be 16 MB or smaller.",
    "attachmentCount": "Too many pending attachments. Reload this tab to clear them.",
    "attachmentExpired": "This attachment expired. Select the file again.",
    "desktopOnly": "This feature is available only in the desktop application.",
    "backendTab": "Open that backend in its own browser tab.",
    "invalidRoute": "Invalid session route.",
    "httpLinks": "Only HTTP(S) links can be opened.",
    "terminalTimeout": "Terminal startup timed out.",
    "terminalFailed": "Terminal connection failed.",
    "terminalClosed": "Terminal connection closed.",
    "oauthCancelled": "OAuth cancelled by user.",
    "oauthMissingFlow": "OAuth server did not provide a flow identifier.",
    "oauthTimeout": "Timed out waiting for MCP OAuth authorization.",
    "oauthFailed": "MCP OAuth authorization failed."
  },
  "de": {
    "attachmentSize": "Anhänge dürfen höchstens 16 MB groß sein.",
    "attachmentCount": "Zu viele ausstehende Anhänge. Laden Sie diesen Tab neu, um sie zu löschen.",
    "attachmentExpired": "Dieser Anhang ist abgelaufen. Wählen Sie die Datei erneut aus.",
    "desktopOnly": "Diese Funktion ist nur in der Desktop-App verfügbar.",
    "backendTab": "Öffnen Sie dieses Backend in einem eigenen Browser-Tab.",
    "invalidRoute": "Ungültiger Sitzungspfad.",
    "httpLinks": "Es können nur HTTP(S)-Links geöffnet werden.",
    "terminalTimeout": "Zeitüberschreitung beim Terminalstart.",
    "terminalFailed": "Terminalverbindung fehlgeschlagen.",
    "terminalClosed": "Terminalverbindung geschlossen.",
    "oauthCancelled": "OAuth vom Benutzer abgebrochen.",
    "oauthMissingFlow": "Der OAuth-Server hat keine Ablaufkennung bereitgestellt.",
    "oauthTimeout": "Zeitüberschreitung beim Warten auf die MCP-OAuth-Autorisierung.",
    "oauthFailed": "MCP-OAuth-Autorisierung fehlgeschlagen."
  },
  "fr": {
    "attachmentSize": "Les pièces jointes ne doivent pas dépasser 16 Mo.",
    "attachmentCount": "Trop de pièces jointes en attente. Rechargez cet onglet pour les effacer.",
    "attachmentExpired": "Cette pièce jointe a expiré. Sélectionnez à nouveau le fichier.",
    "desktopOnly": "Cette fonction est disponible uniquement dans l’application de bureau.",
    "backendTab": "Ouvrez ce serveur dans un autre onglet du navigateur.",
    "invalidRoute": "Chemin de session invalide.",
    "httpLinks": "Seuls les liens HTTP(S) peuvent être ouverts.",
    "terminalTimeout": "Le démarrage du terminal a expiré.",
    "terminalFailed": "Échec de connexion au terminal.",
    "terminalClosed": "Connexion au terminal fermée.",
    "oauthCancelled": "OAuth annulé par l’utilisateur.",
    "oauthMissingFlow": "Le serveur OAuth n’a pas fourni d’identifiant de procédure.",
    "oauthTimeout": "Délai d’attente de l’autorisation OAuth MCP dépassé.",
    "oauthFailed": "Échec de l’autorisation OAuth MCP."
  },
  "es": {
    "attachmentSize": "Los archivos adjuntos deben tener 16 MB o menos.",
    "attachmentCount": "Demasiados archivos adjuntos pendientes. Recarga esta pestaña para borrarlos.",
    "attachmentExpired": "Este archivo adjunto ha caducado. Selecciona el archivo de nuevo.",
    "desktopOnly": "Esta función solo está disponible en la aplicación de escritorio.",
    "backendTab": "Abre ese servidor en otra pestaña del navegador.",
    "invalidRoute": "Ruta de sesión no válida.",
    "httpLinks": "Solo se pueden abrir enlaces HTTP(S).",
    "terminalTimeout": "Se agotó el tiempo de inicio del terminal.",
    "terminalFailed": "Falló la conexión con el terminal.",
    "terminalClosed": "Se cerró la conexión con el terminal.",
    "oauthCancelled": "OAuth cancelado por el usuario.",
    "oauthMissingFlow": "El servidor OAuth no proporcionó un identificador de flujo.",
    "oauthTimeout": "Se agotó el tiempo de espera de autorización OAuth de MCP.",
    "oauthFailed": "Falló la autorización OAuth de MCP."
  },
  "ru": {
    "attachmentSize": "Размер вложения не должен превышать 16 МБ.",
    "attachmentCount": "Слишком много ожидающих вложений. Перезагрузите вкладку, чтобы очистить их.",
    "attachmentExpired": "Срок действия вложения истёк. Выберите файл заново.",
    "desktopOnly": "Эта функция доступна только в настольном приложении.",
    "backendTab": "Откройте этот сервер в отдельной вкладке браузера.",
    "invalidRoute": "Недопустимый маршрут сеанса.",
    "httpLinks": "Можно открывать только ссылки HTTP(S).",
    "terminalTimeout": "Время ожидания запуска терминала истекло.",
    "terminalFailed": "Не удалось подключиться к терминалу.",
    "terminalClosed": "Соединение с терминалом закрыто.",
    "oauthCancelled": "OAuth отменён пользователем.",
    "oauthMissingFlow": "Сервер OAuth не предоставил идентификатор процесса.",
    "oauthTimeout": "Время ожидания авторизации MCP OAuth истекло.",
    "oauthFailed": "Не удалось выполнить авторизацию MCP OAuth."
  },
  "ja": {
    "attachmentSize": "添付ファイルは16 MB以下にしてください。",
    "attachmentCount": "保留中の添付ファイルが多すぎます。このタブを再読み込みして消去してください。",
    "attachmentExpired": "添付ファイルの有効期限が切れました。ファイルを選択し直してください。",
    "desktopOnly": "この機能はデスクトップアプリでのみ利用できます。",
    "backendTab": "そのバックエンドを別のブラウザータブで開いてください。",
    "invalidRoute": "セッションの経路が無効です。",
    "httpLinks": "HTTP(S)リンクのみ開けます。",
    "terminalTimeout": "ターミナルの起動がタイムアウトしました。",
    "terminalFailed": "ターミナルへの接続に失敗しました。",
    "terminalClosed": "ターミナルへの接続が閉じられました。",
    "oauthCancelled": "ユーザーがOAuthをキャンセルしました。",
    "oauthMissingFlow": "OAuthサーバーがフロー識別子を返しませんでした。",
    "oauthTimeout": "MCP OAuth認証の待機がタイムアウトしました。",
    "oauthFailed": "MCP OAuth認証に失敗しました。"
  },
  "zh": {
    "attachmentSize": "附件不得超过16 MB。",
    "attachmentCount": "待处理附件过多。请重新加载此标签页以清除。",
    "attachmentExpired": "此附件已过期。请重新选择文件。",
    "desktopOnly": "此功能仅在桌面应用中可用。",
    "backendTab": "请在单独的浏览器标签页中打开该后端。",
    "invalidRoute": "会话路径无效。",
    "httpLinks": "只能打开HTTP(S)链接。",
    "terminalTimeout": "终端启动超时。",
    "terminalFailed": "终端连接失败。",
    "terminalClosed": "终端连接已关闭。",
    "oauthCancelled": "用户取消了OAuth。",
    "oauthMissingFlow": "OAuth服务器未提供流程标识符。",
    "oauthTimeout": "等待MCP OAuth授权超时。",
    "oauthFailed": "MCP OAuth授权失败。"
  },
  "zh-hant": {
    "attachmentSize": "附件不得超過16 MB。",
    "attachmentCount": "待處理附件過多。請重新載入此分頁以清除。",
    "attachmentExpired": "此附件已過期。請重新選擇檔案。",
    "desktopOnly": "此功能僅在桌面應用程式中可用。",
    "backendTab": "請在獨立的瀏覽器分頁中開啟該後端。",
    "invalidRoute": "工作階段路徑無效。",
    "httpLinks": "只能開啟HTTP(S)連結。",
    "terminalTimeout": "終端機啟動逾時。",
    "terminalFailed": "終端機連線失敗。",
    "terminalClosed": "終端機連線已關閉。",
    "oauthCancelled": "使用者取消了OAuth。",
    "oauthMissingFlow": "OAuth伺服器未提供流程識別碼。",
    "oauthTimeout": "等待MCP OAuth授權逾時。",
    "oauthFailed": "MCP OAuth授權失敗。"
  },
  "ar": {
    "attachmentSize": "يجب ألا يتجاوز حجم المرفق 16 ميغابايت.",
    "attachmentCount": "هناك مرفقات معلقة كثيرة. أعد تحميل علامة التبويب لمسحها.",
    "attachmentExpired": "انتهت صلاحية هذا المرفق. اختر الملف مجددًا.",
    "desktopOnly": "هذه الميزة متاحة فقط في تطبيق سطح المكتب.",
    "backendTab": "افتح هذا الخادم في علامة تبويب منفصلة.",
    "invalidRoute": "مسار الجلسة غير صالح.",
    "httpLinks": "يمكن فتح روابط HTTP(S) فقط.",
    "terminalTimeout": "انتهت مهلة بدء الطرفية.",
    "terminalFailed": "تعذر الاتصال بالطرفية.",
    "terminalClosed": "أُغلق اتصال الطرفية.",
    "oauthCancelled": "ألغى المستخدم OAuth.",
    "oauthMissingFlow": "لم يقدم خادم OAuth معرّفًا للعملية.",
    "oauthTimeout": "انتهت مهلة انتظار تفويض MCP OAuth.",
    "oauthFailed": "فشل تفويض MCP OAuth."
  }
} satisfies Record<BundledLocale, BrowserMessages>
