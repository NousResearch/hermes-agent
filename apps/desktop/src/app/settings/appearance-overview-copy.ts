import { type BundledLocale, isBundledLocale } from '@/i18n'

interface AppearanceOverviewCopy {
  intro: string
  reading: string
  size: string
  sizeDescription: string
  small: string
  default: string
  large: string
  languageDescription: string
  tips: string
  tipsDescription: string
  artwork: string
  artworkDescription: string
  advancedDescription: string
}

const COPY: Record<BundledLocale, AppearanceOverviewCopy> = {
  en: {
    intro: 'Choose how Hermes looks. The basics are below; everything else lives in Advanced.',
    reading: 'Reading & display',
    size: 'Text size',
    sizeDescription: 'Scales chat text and the composer.',
    small: 'Small', default: 'Default', large: 'Large',
    languageDescription: 'Interface language. Chats stay in whatever language you write.',
    tips: 'Tips & guided tours',
    tipsDescription: 'Occasional hints while you learn the app. Turns off after 30 days.',
    artwork: 'Background artwork',
    artworkDescription: 'Paints an artwork behind the app. Off leaves the plain background and its grain.',
    advancedDescription: 'Themes, scale, fonts, chat details and window layout'
  },
  zh: {
    intro: '选择 Hermes 的外观。下方是基本选项，其他设置都在「高级」中。',
    reading: '阅读与显示', size: '文字大小',
    sizeDescription: '调整对话文字和消息输入框的大小。',
    small: '小', default: '默认', large: '大',
    languageDescription: '界面语言。对话仍使用你输入的语言。',
    tips: '提示与引导导览', tipsDescription: '在熟悉应用时偶尔显示提示。30 天后自动关闭。',
    artwork: '背景图案',
    artworkDescription: '在应用背景中显示图案。关闭后只保留纯色背景和颗粒质感。',
    advancedDescription: '主题、缩放、字体、对话细节和窗口布局'
  },
  'zh-hant': {
    intro: '選擇 Hermes 的外觀。下方是基本選項，其他設定都在「進階」中。',
    reading: '閱讀與顯示', size: '文字大小',
    sizeDescription: '調整對話文字和訊息輸入框的大小。',
    small: '小', default: '預設', large: '大',
    languageDescription: '介面語言。對話仍使用你輸入的語言。',
    tips: '提示與導覽', tipsDescription: '熟悉應用程式時偶爾顯示提示。30 天後自動關閉。',
    artwork: '背景圖案',
    artworkDescription: '在應用程式背景中顯示圖案。關閉後只保留純色背景和顆粒質感。',
    advancedDescription: '主題、縮放、字型、對話細節和視窗版面配置'
  },
  ja: {
    intro: 'Hermes の外観を選びます。基本設定は以下に、その他の設定は「詳細」にあります。',
    reading: '文字と表示', size: '文字サイズ',
    sizeDescription: 'チャットと入力欄の文字サイズを変更します。',
    small: '小', default: '標準', large: '大',
    languageDescription: '画面の表示言語です。チャットは入力した言語のままです。',
    tips: 'ヒントとガイドツアー', tipsDescription: 'アプリに慣れるまで、ときどきヒントを表示します。30日後にオフになります。',
    artwork: '背景アートワーク',
    artworkDescription: 'アプリの背景にアートワークを表示します。オフにすると、無地の背景と粒状感だけになります。',
    advancedDescription: 'テーマ、拡大率、フォント、チャットの詳細、ウィンドウの配置'
  },
  ar: {
    intro: 'اختر مظهر Hermes. الإعدادات الأساسية أدناه، والباقي ضمن الخيارات المتقدمة.',
    reading: 'القراءة والعرض', size: 'حجم النص',
    sizeDescription: 'يضبط حجم نص المحادثة ومحرر الرسائل.',
    small: 'صغير', default: 'افتراضي', large: 'كبير',
    languageDescription: 'لغة الواجهة. تبقى المحادثات باللغة التي تكتب بها.',
    tips: 'نصائح وجولات إرشادية', tipsDescription: 'تلميحات من حين لآخر أثناء تعلّم التطبيق. تتوقف بعد 30 يومًا.',
    artwork: 'صورة الخلفية',
    artworkDescription: 'تعرض صورة فنية خلف التطبيق. عند الإيقاف تبقى الخلفية السادة وحبيباتها فقط.',
    advancedDescription: 'السمات والتكبير والخطوط وتفاصيل المحادثة وتخطيط النافذة'
  },
  ru: {
    intro: 'Выберите внешний вид Hermes. Основные настройки ниже, остальные — в разделе «Дополнительно».',
    reading: 'Чтение и отображение', size: 'Размер текста',
    sizeDescription: 'Изменяет размер текста чата и поля ввода.',
    small: 'Маленький', default: 'Стандартный', large: 'Большой',
    languageDescription: 'Язык интерфейса. Чаты остаются на языке, на котором вы пишете.',
    tips: 'Подсказки и обучающие туры', tipsDescription: 'Редкие подсказки при знакомстве с приложением. Отключаются через 30 дней.',
    artwork: 'Фоновое изображение',
    artworkDescription: 'Показывает изображение за интерфейсом. Если выключить, останутся ровный фон и его зернистость.',
    advancedDescription: 'Темы, масштаб, шрифты, детали чата и расположение окон'
  },
  fr: {
    intro: 'Choisissez l’apparence de Hermes. Les réglages essentiels sont ci-dessous, les autres dans Avancé.',
    reading: 'Lecture et affichage', size: 'Taille du texte',
    sizeDescription: 'Ajuste le texte des conversations et de la saisie.',
    small: 'Petite', default: 'Par défaut', large: 'Grande',
    languageDescription: 'Langue de l’interface. Les conversations restent dans la langue que vous utilisez.',
    tips: 'Astuces et visites guidées', tipsDescription: 'Des conseils occasionnels pour découvrir l’application. Désactivés après 30 jours.',
    artwork: 'Illustration d’arrière-plan',
    artworkDescription: 'Affiche une illustration derrière l’application. Désactivée, il reste le fond uni et son grain.',
    advancedDescription: 'Thèmes, échelle, polices, détails des discussions et disposition des fenêtres'
  },
  de: {
    intro: 'Wähle das Aussehen von Hermes. Die Grundlagen stehen unten, alles Weitere unter Erweitert.',
    reading: 'Lesen und Anzeige', size: 'Textgröße',
    sizeDescription: 'Skaliert den Chattext und die Nachrichteneingabe.',
    small: 'Klein', default: 'Standard', large: 'Groß',
    languageDescription: 'Sprache der Oberfläche. Chats bleiben in der Sprache, in der du schreibst.',
    tips: 'Tipps und geführte Touren', tipsDescription: 'Gelegentliche Hinweise zum Kennenlernen der App. Nach 30 Tagen deaktiviert.',
    artwork: 'Hintergrundbild',
    artworkDescription: 'Zeigt ein Bild hinter der App. Ausgeschaltet bleiben der einfarbige Hintergrund und seine Körnung.',
    advancedDescription: 'Themes, Skalierung, Schriften, Chatdetails und Fensterlayout'
  },
  es: {
    intro: 'Elige el aspecto de Hermes. Los ajustes básicos están abajo; el resto, en Avanzado.',
    reading: 'Lectura y visualización', size: 'Tamaño del texto',
    sizeDescription: 'Ajusta el texto del chat y del editor de mensajes.',
    small: 'Pequeño', default: 'Predeterminado', large: 'Grande',
    languageDescription: 'Idioma de la interfaz. Los chats siguen en el idioma en el que escribas.',
    tips: 'Consejos y recorridos guiados', tipsDescription: 'Consejos ocasionales mientras conoces la app. Se desactivan tras 30 días.',
    artwork: 'Imagen de fondo',
    artworkDescription: 'Muestra una imagen detrás de la app. Desactivada, queda el fondo liso con su grano.',
    advancedDescription: 'Temas, escala, fuentes, detalles del chat y disposición de ventanas'
  }
}

export function appearanceOverviewCopy(locale: string): AppearanceOverviewCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}
