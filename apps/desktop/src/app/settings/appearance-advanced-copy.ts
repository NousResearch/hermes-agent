import { type BundledLocale, isBundledLocale } from '@/i18n'

type Row = readonly [title: string, description: string]

/** Redesigned Appearance › Advanced (Figma 829:2712): three groups of rows, plus this device's extras. */
export interface AppearanceAdvancedCopy {
  groups: { text: string; chat: string; layout: string; device: string }
  interfaceScale: Row
  chatFont: Row
  textDirection: Row
  toolCalls: Row
  hideCodeDiffs: Row
  linkPreviews: Row
  reactions: Row
  vibeHearts: Row
  bubble: Row
  collapseThinking: Row
  density: Row
  timeline: Row
  pricing: Row
  welcome: Row
}

const COPY: Record<BundledLocale, AppearanceAdvancedCopy> = {
  en: {
    groups: { text: 'Text', chat: 'Chat', layout: 'Layout', device: 'This device' },
    interfaceScale: ['Interface scale', 'Sizes menus and controls. Ctrl/Cmd + and − work too.'],
    chatFont: ['Chat font', 'Swap in a readability face such as OpenDyslexic.'],
    textDirection: ['Text direction', 'Auto follows each paragraph’s first letter. Code stays left-to-right.'],
    toolCalls: ['Tool calls', 'Simple hides raw payloads; Technical shows full input and output.'],
    hideCodeDiffs: ['Hide code diffs', 'Show file edits as a one-line summary of lines added and removed.'],
    linkPreviews: ['Link previews', 'Rich embeds from sites like YouTube and X.'],
    reactions: ['Message reactions', 'Emoji tapbacks on messages, both ways.'],
    vibeHearts: ['Vibe hearts', 'Floating hearts when you say thanks.'],
    bubble: ['Message bubble transparency', 'Solid at 0%; outline only at 100%.'],
    collapseThinking: ['Collapse thinking', 'Keep Hermes’s reasoning tucked away until you open it.'],
    density: ['Sidebar density', 'How much detail shows under each chat title.'],
    timeline: ['Thread timeline', 'Navigation bars along the right edge of a conversation.'],
    pricing: ['Model pricing', 'Show per-million-token prices in the model picker.'],
    welcome: ['Welcome screen', 'Wordmark and prompt on an empty chat.']
  },
  fr: {
    groups: { text: 'Texte', chat: 'Discussion', layout: 'Disposition', device: 'Cet appareil' },
    interfaceScale: [
      'Échelle de l’interface',
      'Ajuste les menus et les commandes. Ctrl/Cmd + et − fonctionnent aussi.'
    ],
    chatFont: ['Police des discussions', 'Choisissez une police lisible comme OpenDyslexic.'],
    textDirection: [
      'Sens du texte',
      'Auto suit la première lettre de chaque paragraphe. Le code reste de gauche à droite.'
    ],
    toolCalls: [
      'Appels d’outils',
      'Simple masque les données brutes ; Technique montre l’entrée et la sortie complètes.'
    ],
    hideCodeDiffs: [
      'Masquer les diffs de code',
      'Affiche les modifications de fichiers en une ligne de lignes ajoutées et supprimées.'
    ],
    linkPreviews: ['Aperçus de liens', 'Intégrations enrichies de sites comme YouTube et X.'],
    reactions: ['Réactions aux messages', 'Réactions emoji sur les messages, dans les deux sens.'],
    vibeHearts: ['Cœurs flottants', 'Des cœurs s’envolent quand vous dites merci.'],
    bubble: ['Transparence des bulles', 'Pleine à 0 % ; contour seul à 100 %.'],
    collapseThinking: ['Replier la réflexion', 'Garde le raisonnement de Hermes replié jusqu’à ce que vous l’ouvriez.'],
    density: ['Densité de la barre latérale', 'Quantité de détails sous chaque titre de discussion.'],
    timeline: ['Chronologie du fil', 'Barres de navigation le long du bord droit d’une conversation.'],
    pricing: ['Tarifs des modèles', 'Affiche les prix par million de jetons dans le sélecteur de modèle.'],
    welcome: ['Écran d’accueil', 'Logo et invite sur une discussion vide.']
  },
  de: {
    groups: { text: 'Text', chat: 'Chat', layout: 'Layout', device: 'Dieses Gerät' },
    interfaceScale: [
      'Oberflächenskalierung',
      'Skaliert Menüs und Bedienelemente. Strg/Cmd + und − funktionieren auch.'
    ],
    chatFont: ['Chatschrift', 'Eine gut lesbare Schrift wie OpenDyslexic verwenden.'],
    textDirection: ['Textrichtung', 'Auto folgt dem ersten Buchstaben jedes Absatzes. Code bleibt links nach rechts.'],
    toolCalls: ['Tool-Aufrufe', 'Einfach blendet Rohdaten aus; Technisch zeigt Ein- und Ausgabe vollständig.'],
    hideCodeDiffs: [
      'Code-Diffs ausblenden',
      'Dateiänderungen als einzeilige Zusammenfassung hinzugefügter und entfernter Zeilen.'
    ],
    linkPreviews: ['Linkvorschauen', 'Eingebettete Inhalte von Seiten wie YouTube und X.'],
    reactions: ['Nachrichtenreaktionen', 'Emoji-Reaktionen auf Nachrichten, in beide Richtungen.'],
    vibeHearts: ['Vibe-Herzen', 'Schwebende Herzen, wenn du Danke sagst.'],
    bubble: ['Transparenz der Nachrichtenblasen', 'Deckend bei 0 %; nur Umriss bei 100 %.'],
    collapseThinking: ['Denken einklappen', 'Hält Hermes’ Überlegungen eingeklappt, bis du sie öffnest.'],
    density: ['Seitenleistendichte', 'Wie viele Details unter jedem Chattitel stehen.'],
    timeline: ['Verlaufsleiste', 'Navigationsbalken am rechten Rand einer Unterhaltung.'],
    pricing: ['Modellpreise', 'Preise pro Million Token in der Modellauswahl anzeigen.'],
    welcome: ['Begrüßungsbildschirm', 'Wortmarke und Eingabe in einem leeren Chat.']
  },
  es: {
    groups: { text: 'Texto', chat: 'Chat', layout: 'Diseño', device: 'Este dispositivo' },
    interfaceScale: ['Escala de la interfaz', 'Ajusta menús y controles. Ctrl/Cmd + y − también funcionan.'],
    chatFont: ['Fuente del chat', 'Usa una fuente legible como OpenDyslexic.'],
    textDirection: [
      'Dirección del texto',
      'Auto sigue la primera letra de cada párrafo. El código va de izquierda a derecha.'
    ],
    toolCalls: [
      'Llamadas a herramientas',
      'Simple oculta los datos sin procesar; Técnico muestra entrada y salida completas.'
    ],
    hideCodeDiffs: [
      'Ocultar diffs de código',
      'Muestra los cambios de archivos como un resumen de líneas añadidas y eliminadas.'
    ],
    linkPreviews: ['Vistas previas de enlaces', 'Contenido incrustado de sitios como YouTube y X.'],
    reactions: ['Reacciones a mensajes', 'Reacciones con emoji en los mensajes, en ambos sentidos.'],
    vibeHearts: ['Corazones flotantes', 'Corazones que flotan cuando das las gracias.'],
    bubble: ['Transparencia de las burbujas', 'Opaca al 0 %; solo contorno al 100 %.'],
    collapseThinking: ['Contraer razonamiento', 'Mantiene el razonamiento de Hermes plegado hasta que lo abras.'],
    density: ['Densidad de la barra lateral', 'Cuánto detalle aparece bajo cada título de chat.'],
    timeline: ['Línea de tiempo del hilo', 'Barras de navegación en el borde derecho de una conversación.'],
    pricing: ['Precios de modelos', 'Muestra precios por millón de tokens en el selector de modelo.'],
    welcome: ['Pantalla de bienvenida', 'Logotipo e indicación en un chat vacío.']
  },
  ja: {
    groups: { text: 'テキスト', chat: 'チャット', layout: 'レイアウト', device: 'このデバイス' },
    interfaceScale: ['画面の拡大率', 'メニューと操作部品のサイズです。Ctrl/Cmd と + / − でも変更できます。'],
    chatFont: ['チャットのフォント', 'OpenDyslexic など読みやすいフォントに切り替えます。'],
    textDirection: ['文字の方向', '自動では段落の最初の文字に従います。コードは常に左から右です。'],
    toolCalls: ['ツール呼び出し', 'シンプルは生データを隠し、テクニカルは入出力をすべて表示します。'],
    hideCodeDiffs: ['コードの差分を隠す', 'ファイル編集を追加・削除行数の1行の要約で表示します。'],
    linkPreviews: ['リンクのプレビュー', 'YouTube や X などのサイトの埋め込み表示です。'],
    reactions: ['メッセージのリアクション', 'メッセージへの絵文字リアクション（双方向）。'],
    vibeHearts: ['ハートの演出', 'お礼を言うとハートが浮かびます。'],
    bubble: ['メッセージの吹き出しの透明度', '0% で不透明、100% で枠線のみ。'],
    collapseThinking: ['思考を折りたたむ', 'エージェントの思考を開くまで折りたたんでおきます。'],
    density: ['サイドバーの密度', '各チャットのタイトルの下に表示する情報量です。'],
    timeline: ['スレッドのタイムライン', '会話の右端に表示されるナビゲーションバーです。'],
    pricing: ['モデルの料金', 'モデル選択に100万トークンあたりの料金を表示します。'],
    welcome: ['ようこそ画面', '空のチャットに表示するロゴと入力の案内です。']
  },
  zh: {
    groups: { text: '文字', chat: '对话', layout: '布局', device: '此设备' },
    interfaceScale: ['界面缩放', '调整菜单和控件的大小。也可使用 Ctrl/Cmd 加 + 和 −。'],
    chatFont: ['对话字体', '换用 OpenDyslexic 等易读字体。'],
    textDirection: ['文字方向', '自动会依据每段的第一个字母。代码始终从左到右。'],
    toolCalls: ['工具调用', '简洁模式隐藏原始数据；技术模式显示完整的输入和输出。'],
    hideCodeDiffs: ['隐藏代码差异', '将文件编辑显示为新增和删除行数的一行摘要。'],
    linkPreviews: ['链接预览', '来自 YouTube、X 等网站的富媒体嵌入。'],
    reactions: ['消息回应', '双向的消息表情回应。'],
    vibeHearts: ['爱心特效', '道谢时飘出爱心。'],
    bubble: ['消息气泡透明度', '0% 为实心；100% 仅保留轮廓。'],
    collapseThinking: ['折叠思考过程', '在你展开之前，Hermes 的推理保持折叠。'],
    density: ['侧边栏密度', '每个对话标题下显示多少细节。'],
    timeline: ['对话时间线', '对话右侧边缘的导航条。'],
    pricing: ['模型价格', '在模型选择器中显示每百万 token 的价格。'],
    welcome: ['欢迎界面', '空白对话中的标志和提示。']
  },
  'zh-hant': {
    groups: { text: '文字', chat: '對話', layout: '版面配置', device: '此裝置' },
    interfaceScale: ['介面縮放', '調整選單與控制項的大小。也可使用 Ctrl/Cmd 加 + 與 −。'],
    chatFont: ['對話字型', '換用 OpenDyslexic 等易讀字型。'],
    textDirection: ['文字方向', '自動會依據每段的第一個字母。程式碼一律由左至右。'],
    toolCalls: ['工具呼叫', '簡潔模式隱藏原始資料；技術模式顯示完整的輸入與輸出。'],
    hideCodeDiffs: ['隱藏程式碼差異', '將檔案編輯顯示為新增與刪除行數的一行摘要。'],
    linkPreviews: ['連結預覽', '來自 YouTube、X 等網站的豐富內嵌內容。'],
    reactions: ['訊息回應', '雙向的訊息表情回應。'],
    vibeHearts: ['愛心特效', '道謝時飄出愛心。'],
    bubble: ['訊息泡泡透明度', '0% 為實心；100% 只保留外框。'],
    collapseThinking: ['摺疊思考過程', '在你展開之前，Hermes 的推理保持摺疊。'],
    density: ['側邊欄密度', '每個對話標題下顯示多少細節。'],
    timeline: ['對話時間軸', '對話右側邊緣的導覽列。'],
    pricing: ['模型價格', '在模型選擇器中顯示每百萬 token 的價格。'],
    welcome: ['歡迎畫面', '空白對話中的標誌與提示。']
  },
  ru: {
    groups: { text: 'Текст', chat: 'Чат', layout: 'Расположение', device: 'Это устройство' },
    interfaceScale: ['Масштаб интерфейса', 'Размер меню и элементов управления. Ctrl/Cmd + и − тоже работают.'],
    chatFont: ['Шрифт чата', 'Удобочитаемый шрифт, например OpenDyslexic.'],
    textDirection: ['Направление текста', 'Авто следует первой букве абзаца. Код всегда слева направо.'],
    toolCalls: ['Вызовы инструментов', 'Простой скрывает сырые данные; технический показывает ввод и вывод полностью.'],
    hideCodeDiffs: ['Скрывать диффы кода', 'Правки файлов — одной строкой с числом добавленных и удалённых строк.'],
    linkPreviews: ['Превью ссылок', 'Встраивания с сайтов вроде YouTube и X.'],
    reactions: ['Реакции на сообщения', 'Эмодзи-реакции на сообщения в обе стороны.'],
    vibeHearts: ['Сердечки', 'Летящие сердечки, когда вы благодарите.'],
    bubble: ['Прозрачность пузырей сообщений', 'Сплошной при 0 %; только контур при 100 %.'],
    collapseThinking: ['Сворачивать размышления', 'Рассуждения Hermes свёрнуты, пока вы их не откроете.'],
    density: ['Плотность боковой панели', 'Сколько деталей показывать под названием чата.'],
    timeline: ['Шкала беседы', 'Панели навигации у правого края беседы.'],
    pricing: ['Цены моделей', 'Показывать цены за миллион токенов в выборе модели.'],
    welcome: ['Экран приветствия', 'Логотип и подсказка в пустом чате.']
  },
  ar: {
    groups: { text: 'النص', chat: 'المحادثة', layout: 'التخطيط', device: 'هذا الجهاز' },
    interfaceScale: ['حجم الواجهة', 'يضبط حجم القوائم وعناصر التحكم. يعمل أيضًا Ctrl/Cmd مع + و −.'],
    chatFont: ['خط المحادثة', 'استخدم خطًا سهل القراءة مثل OpenDyslexic.'],
    textDirection: ['اتجاه النص', 'تلقائي يتبع أول حرف في كل فقرة. تبقى الشيفرة من اليسار إلى اليمين.'],
    toolCalls: ['استدعاءات الأدوات', 'البسيط يخفي البيانات الخام؛ التقني يعرض المدخلات والمخرجات كاملة.'],
    hideCodeDiffs: ['إخفاء فروق الشيفرة', 'يعرض تعديلات الملفات كملخص من سطر واحد للأسطر المضافة والمحذوفة.'],
    linkPreviews: ['معاينات الروابط', 'تضمينات غنية من مواقع مثل YouTube وX.'],
    reactions: ['تفاعلات الرسائل', 'تفاعلات بالرموز التعبيرية على الرسائل في الاتجاهين.'],
    vibeHearts: ['قلوب عائمة', 'قلوب تطفو عندما تقول شكرًا.'],
    bubble: ['شفافية فقاعات الرسائل', 'مصمتة عند 0٪؛ إطار فقط عند 100٪.'],
    collapseThinking: ['طي التفكير', 'يبقي تفكير Hermes مطويًا حتى تفتحه.'],
    density: ['كثافة الشريط الجانبي', 'مقدار التفاصيل أسفل عنوان كل محادثة.'],
    timeline: ['الخط الزمني للمحادثة', 'أشرطة التنقل على الحافة اليمنى للمحادثة.'],
    pricing: ['أسعار النماذج', 'يعرض الأسعار لكل مليون رمز في منتقي النموذج.'],
    welcome: ['شاشة الترحيب', 'الشعار والتلميح في محادثة فارغة.']
  }
}

export function appearanceAdvancedCopy(locale: string): AppearanceAdvancedCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}
