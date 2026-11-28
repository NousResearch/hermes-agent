import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

import type { GalleryGroup } from './registry/types'

/** The design system page's chrome. What it documents (token names, values, the source comments, the
 *  primitives' one-line uses, DESIGN.md's table) is the code's own text and stays as written. */
export interface DesignSystemCopy {
  title: string
  description: string
  /** The page panel's title. */
  contents: string
  preview: string
  previewNote: string
  mode: string
  light: string
  dark: string
  accent: string
  tokens: string
  tokensIntro: string
  tokenCount: (count: number) => string
  /** Token families, by token file (`raw` for raw.css). A file without a name shows its file name. */
  families: Record<string, string>
  copyName: (name: string) => string
  components: string
  componentsIntro: string
  groups: Record<GalleryGroup, string>
  notShown: string
  notShownIntro: string
  stillToUnify: string
  stillToUnifyIntro: string
}

const en: DesignSystemCopy = {
  title: 'Design system',
  description:
    'The tokens and primitives the redesign is built from, drawn from the code that ships them. Figma sets the direction; this page, the tokens and DESIGN.md are the reference.',
  contents: 'Contents',
  preview: 'Preview',
  previewNote: 'These change your appearance setting.',
  mode: 'Mode',
  light: 'Light',
  dark: 'Dark',
  accent: 'Accent',
  tokens: 'Tokens',
  tokensIntro:
    'Every token in src/app/redesign/tokens, read from the CSS. Values are resolved in the current theme; select a name to copy it.',
  tokenCount: count => `${count} ${count === 1 ? 'token' : 'tokens'}`,
  families: {
    raw: 'Primitives',
    color: 'Colour',
    type: 'Type',
    space: 'Space and layout',
    shape: 'Shape',
    motion: 'Motion',
    layers: 'Layers',
    bridge: 'Bridge',
    sidebar: 'Chat sidebar',
    tailwind: 'Utilities'
  },
  copyName: name => `Copy ${name}`,
  components: 'Components',
  componentsIntro:
    'The primitives in src/components/ui and the page frame, rendered live in their variants and states.',
  groups: {
    actions: 'Actions',
    inputs: 'Inputs',
    overlays: 'Overlays',
    feedback: 'Feedback',
    layout: 'Layout',
    text: 'Text',
    icons: 'Icons'
  },
  notShown: 'Not shown',
  notShownIntro: 'Files in src/components/ui this page does not render, and why.',
  stillToUnify: 'Still to unify',
  stillToUnifyIntro: 'Where the implementation still differs from the language, from DESIGN.md, which owns the list.'
}

const fr: DesignSystemCopy = {
  title: 'Système de design',
  description:
    'Les jetons et primitives qui composent la refonte, tirés du code qui les livre. Figma donne la direction ; cette page, les jetons et DESIGN.md font référence.',
  contents: 'Sommaire',
  preview: 'Aperçu',
  previewNote: 'Ces choix modifient votre réglage d’apparence.',
  mode: 'Mode',
  light: 'Clair',
  dark: 'Sombre',
  accent: 'Accent',
  tokens: 'Jetons',
  tokensIntro:
    'Tous les jetons de src/app/redesign/tokens, lus dans le CSS. Les valeurs sont résolues dans le thème actuel ; sélectionnez un nom pour le copier.',
  tokenCount: count => `${count} ${count === 1 ? 'jeton' : 'jetons'}`,
  families: {
    raw: 'Primitives',
    color: 'Couleur',
    type: 'Typographie',
    space: 'Espace et mise en page',
    shape: 'Forme',
    motion: 'Mouvement',
    layers: 'Calques',
    bridge: 'Pont',
    sidebar: 'Barre latérale du chat',
    tailwind: 'Utilitaires'
  },
  copyName: name => `Copier ${name}`,
  components: 'Composants',
  componentsIntro:
    'Les primitives de src/components/ui et le cadre de page, rendus en direct dans leurs variantes et leurs états.',
  groups: {
    actions: 'Actions',
    inputs: 'Saisie',
    overlays: 'Superpositions',
    feedback: 'Retour',
    layout: 'Mise en page',
    text: 'Texte',
    icons: 'Icônes'
  },
  notShown: 'Non affichés',
  notShownIntro: 'Les fichiers de src/components/ui que cette page n’affiche pas, et pourquoi.',
  stillToUnify: 'Reste à unifier',
  stillToUnifyIntro: 'Là où l’implémentation diffère encore du langage, d’après DESIGN.md, qui tient la liste.'
}

const de: DesignSystemCopy = {
  title: 'Designsystem',
  description:
    'Die Tokens und Grundbausteine des Redesigns, direkt aus dem Code, der sie ausliefert. Figma gibt die Richtung vor; diese Seite, die Tokens und DESIGN.md sind die Referenz.',
  contents: 'Inhalt',
  preview: 'Vorschau',
  previewNote: 'Diese Auswahl ändert deine Darstellungseinstellung.',
  mode: 'Modus',
  light: 'Hell',
  dark: 'Dunkel',
  accent: 'Akzent',
  tokens: 'Tokens',
  tokensIntro:
    'Alle Tokens aus src/app/redesign/tokens, aus dem CSS gelesen. Werte sind im aktuellen Design aufgelöst; wähle einen Namen, um ihn zu kopieren.',
  tokenCount: count => `${count} ${count === 1 ? 'Token' : 'Tokens'}`,
  families: {
    raw: 'Grundwerte',
    color: 'Farbe',
    type: 'Schrift',
    space: 'Abstand und Layout',
    shape: 'Form',
    motion: 'Bewegung',
    layers: 'Ebenen',
    bridge: 'Brücke',
    sidebar: 'Chat-Seitenleiste',
    tailwind: 'Hilfsklassen'
  },
  copyName: name => `${name} kopieren`,
  components: 'Komponenten',
  componentsIntro:
    'Die Grundbausteine aus src/components/ui und der Seitenrahmen, live in ihren Varianten und Zuständen.',
  groups: {
    actions: 'Aktionen',
    inputs: 'Eingaben',
    overlays: 'Überlagerungen',
    feedback: 'Rückmeldung',
    layout: 'Layout',
    text: 'Text',
    icons: 'Symbole'
  },
  notShown: 'Nicht gezeigt',
  notShownIntro: 'Dateien in src/components/ui, die diese Seite nicht zeigt, und warum.',
  stillToUnify: 'Noch zu vereinheitlichen',
  stillToUnifyIntro: 'Wo die Umsetzung noch von der Sprache abweicht, laut DESIGN.md, das die Liste führt.'
}

const es: DesignSystemCopy = {
  title: 'Sistema de diseño',
  description:
    'Los tokens y primitivas con los que se construye el rediseño, tomados del código que los publica. Figma marca la dirección; esta página, los tokens y DESIGN.md son la referencia.',
  contents: 'Contenido',
  preview: 'Vista previa',
  previewNote: 'Esto cambia tu ajuste de apariencia.',
  mode: 'Modo',
  light: 'Claro',
  dark: 'Oscuro',
  accent: 'Acento',
  tokens: 'Tokens',
  tokensIntro:
    'Todos los tokens de src/app/redesign/tokens, leídos del CSS. Los valores se resuelven en el tema actual; selecciona un nombre para copiarlo.',
  tokenCount: count => `${count} ${count === 1 ? 'token' : 'tokens'}`,
  families: {
    raw: 'Primitivas',
    color: 'Color',
    type: 'Tipografía',
    space: 'Espacio y diseño',
    shape: 'Forma',
    motion: 'Movimiento',
    layers: 'Capas',
    bridge: 'Puente',
    sidebar: 'Barra lateral del chat',
    tailwind: 'Utilidades'
  },
  copyName: name => `Copiar ${name}`,
  components: 'Componentes',
  componentsIntro:
    'Las primitivas de src/components/ui y el marco de página, mostradas en vivo en sus variantes y estados.',
  groups: {
    actions: 'Acciones',
    inputs: 'Entradas',
    overlays: 'Superposiciones',
    feedback: 'Respuesta',
    layout: 'Diseño',
    text: 'Texto',
    icons: 'Iconos'
  },
  notShown: 'No mostrados',
  notShownIntro: 'Archivos de src/components/ui que esta página no muestra, y por qué.',
  stillToUnify: 'Pendiente de unificar',
  stillToUnifyIntro: 'Donde la implementación aún difiere del lenguaje, según DESIGN.md, que mantiene la lista.'
}

const ja: DesignSystemCopy = {
  title: 'デザインシステム',
  description:
    'リデザインを構成するトークンとプリミティブを、それを出荷するコードから描画しています。Figma は方向性を示し、このページ・トークン・DESIGN.md が基準です。',
  contents: '目次',
  preview: 'プレビュー',
  previewNote: 'ここでの選択は外観の設定を変更します。',
  mode: 'モード',
  light: 'ライト',
  dark: 'ダーク',
  accent: 'アクセント',
  tokens: 'トークン',
  tokensIntro:
    'src/app/redesign/tokens のすべてのトークンを CSS から読み込んでいます。値は現在のテーマで解決されます。名前を選ぶとコピーされます。',
  tokenCount: count => `${count} 件のトークン`,
  families: {
    raw: 'プリミティブ',
    color: 'カラー',
    type: 'タイポグラフィ',
    space: 'スペースとレイアウト',
    shape: 'シェイプ',
    motion: 'モーション',
    layers: 'レイヤー',
    bridge: 'ブリッジ',
    sidebar: 'チャットサイドバー',
    tailwind: 'ユーティリティ'
  },
  copyName: name => `${name} をコピー`,
  components: 'コンポーネント',
  componentsIntro: 'src/components/ui のプリミティブとページフレームを、バリアントと状態ごとにライブで描画しています。',
  groups: {
    actions: 'アクション',
    inputs: '入力',
    overlays: 'オーバーレイ',
    feedback: 'フィードバック',
    layout: 'レイアウト',
    text: 'テキスト',
    icons: 'アイコン'
  },
  notShown: '表示していないもの',
  notShownIntro: 'このページで描画していない src/components/ui のファイルとその理由。',
  stillToUnify: '統一が残っている箇所',
  stillToUnifyIntro: '実装がまだ言語と異なる箇所です。一覧は DESIGN.md が管理しています。'
}

const zh: DesignSystemCopy = {
  title: '设计系统',
  description:
    '重新设计所用的令牌和基础组件，直接取自发布它们的代码。Figma 指明方向；本页、令牌和 DESIGN.md 是参考标准。',
  contents: '目录',
  preview: '预览',
  previewNote: '这些选项会更改你的外观设置。',
  mode: '模式',
  light: '浅色',
  dark: '深色',
  accent: '强调色',
  tokens: '令牌',
  tokensIntro: 'src/app/redesign/tokens 中的全部令牌，从 CSS 读取。数值按当前主题解析；选择名称即可复制。',
  tokenCount: count => `${count} 个令牌`,
  families: {
    raw: '原始值',
    color: '颜色',
    type: '字体',
    space: '间距与布局',
    shape: '形状',
    motion: '动效',
    layers: '层级',
    bridge: '桥接',
    sidebar: '聊天侧边栏',
    tailwind: '工具类'
  },
  copyName: name => `复制 ${name}`,
  components: '组件',
  componentsIntro: 'src/components/ui 中的基础组件和页面框架，按变体和状态实时渲染。',
  groups: {
    actions: '操作',
    inputs: '输入',
    overlays: '浮层',
    feedback: '反馈',
    layout: '布局',
    text: '文本',
    icons: '图标'
  },
  notShown: '未展示',
  notShownIntro: '本页未渲染的 src/components/ui 文件及原因。',
  stillToUnify: '待统一',
  stillToUnifyIntro: '实现与设计语言仍有差异之处，清单由 DESIGN.md 维护。'
}

const zhHant: DesignSystemCopy = {
  title: '設計系統',
  description:
    '重新設計所用的權杖與基礎元件，直接取自發佈它們的程式碼。Figma 指出方向；本頁、權杖與 DESIGN.md 是參考標準。',
  contents: '目錄',
  preview: '預覽',
  previewNote: '這些選項會變更你的外觀設定。',
  mode: '模式',
  light: '淺色',
  dark: '深色',
  accent: '強調色',
  tokens: '權杖',
  tokensIntro: 'src/app/redesign/tokens 中的全部權杖，從 CSS 讀取。數值依目前主題解析；選取名稱即可複製。',
  tokenCount: count => `${count} 個權杖`,
  families: {
    raw: '原始值',
    color: '顏色',
    type: '字體',
    space: '間距與版面',
    shape: '形狀',
    motion: '動態',
    layers: '圖層',
    bridge: '橋接',
    sidebar: '聊天側邊欄',
    tailwind: '工具類別'
  },
  copyName: name => `複製 ${name}`,
  components: '元件',
  componentsIntro: 'src/components/ui 中的基礎元件與頁面框架，依變體與狀態即時呈現。',
  groups: {
    actions: '動作',
    inputs: '輸入',
    overlays: '浮層',
    feedback: '回饋',
    layout: '版面',
    text: '文字',
    icons: '圖示'
  },
  notShown: '未展示',
  notShownIntro: '本頁未呈現的 src/components/ui 檔案及原因。',
  stillToUnify: '待統一',
  stillToUnifyIntro: '實作與設計語言仍有差異之處，清單由 DESIGN.md 維護。'
}

const ru: DesignSystemCopy = {
  title: 'Дизайн-система',
  description:
    'Токены и примитивы, из которых собран редизайн, взятые из кода, который их поставляет. Figma задаёт направление; эталон — эта страница, токены и DESIGN.md.',
  contents: 'Содержание',
  preview: 'Предпросмотр',
  previewNote: 'Эти переключатели меняют настройку оформления.',
  mode: 'Режим',
  light: 'Светлый',
  dark: 'Тёмный',
  accent: 'Акцент',
  tokens: 'Токены',
  tokensIntro:
    'Все токены из src/app/redesign/tokens, прочитанные из CSS. Значения вычислены в текущей теме; выберите имя, чтобы скопировать его.',
  tokenCount: count => `Токенов: ${count}`,
  families: {
    raw: 'Примитивы',
    color: 'Цвет',
    type: 'Шрифт',
    space: 'Отступы и сетка',
    shape: 'Форма',
    motion: 'Движение',
    layers: 'Слои',
    bridge: 'Мост',
    sidebar: 'Боковая панель чата',
    tailwind: 'Утилиты'
  },
  copyName: name => `Скопировать ${name}`,
  components: 'Компоненты',
  componentsIntro: 'Примитивы из src/components/ui и рамка страницы, вживую во всех вариантах и состояниях.',
  groups: {
    actions: 'Действия',
    inputs: 'Ввод',
    overlays: 'Оверлеи',
    feedback: 'Обратная связь',
    layout: 'Компоновка',
    text: 'Текст',
    icons: 'Значки'
  },
  notShown: 'Не показаны',
  notShownIntro: 'Файлы из src/components/ui, которых нет на этой странице, и причины.',
  stillToUnify: 'Ещё не унифицировано',
  stillToUnifyIntro: 'Где реализация пока расходится с языком; список ведёт DESIGN.md.'
}

const ar: DesignSystemCopy = {
  title: 'نظام التصميم',
  description:
    'الرموز والعناصر الأساسية التي بُني منها التصميم الجديد، مأخوذة من الشيفرة التي تشحنها. يحدّد Figma الاتجاه؛ وهذه الصفحة والرموز وDESIGN.md هي المرجع.',
  contents: 'المحتويات',
  preview: 'معاينة',
  previewNote: 'تغيّر هذه الخيارات إعداد المظهر لديك.',
  mode: 'الوضع',
  light: 'فاتح',
  dark: 'داكن',
  accent: 'لون التمييز',
  tokens: 'الرموز',
  tokensIntro: 'كل الرموز في src/app/redesign/tokens، مقروءة من CSS. القيم محسوبة في السمة الحالية؛ اختر اسمًا لنسخه.',
  tokenCount: count => `${count} رمز`,
  families: {
    raw: 'القيم الأساسية',
    color: 'الألوان',
    type: 'الخطوط',
    space: 'المسافات والتخطيط',
    shape: 'الأشكال',
    motion: 'الحركة',
    layers: 'الطبقات',
    bridge: 'الجسر',
    sidebar: 'الشريط الجانبي للمحادثة',
    tailwind: 'الأدوات المساعدة'
  },
  copyName: name => `نسخ ${name}`,
  components: 'المكوّنات',
  componentsIntro: 'العناصر الأساسية في src/components/ui وإطار الصفحة، معروضة مباشرةً بأشكالها وحالاتها.',
  groups: {
    actions: 'الإجراءات',
    inputs: 'الإدخال',
    overlays: 'الطبقات العائمة',
    feedback: 'الاستجابة',
    layout: 'التخطيط',
    text: 'النص',
    icons: 'الأيقونات'
  },
  notShown: 'غير معروضة',
  notShownIntro: 'ملفات src/components/ui التي لا تعرضها هذه الصفحة، وسبب ذلك.',
  stillToUnify: 'ما زال بحاجة إلى توحيد',
  stillToUnifyIntro: 'حيث ما زال التنفيذ يختلف عن اللغة، وفق DESIGN.md الذي يحتفظ بالقائمة.'
}

const COPY: Record<BundledLocale, DesignSystemCopy> = { en, fr, de, es, ja, zh, 'zh-hant': zhHant, ru, ar }

export function designSystemCopy(locale: string): DesignSystemCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useDesignSystemCopy(): DesignSystemCopy {
  return designSystemCopy(useI18n().locale)
}
