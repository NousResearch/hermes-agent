import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

export type ChatItemsTab = 'all' | 'artifacts' | 'files' | 'web'

/** The chat items pane's words (Figma 1083:8348 / 1095:9308). Redesign only, like the Library's. */
export interface ChatItemsCopy {
  pane: string
  tabs: Record<ChatItemsTab, string>
  tabsLabel: string
  seeAll: (section: string) => string
  seeAllShort: string
  artifactsIntro: string
  madeForYou: string
  allItems: string
  itemsButton: (count: number) => string
  emptyTitle: string
  emptyDescription: string
  noChat: string
  version: (version: number) => string
  close: string
}

const en: ChatItemsCopy = {
  pane: 'Items',
  tabs: { all: 'All', artifacts: 'Artifacts', files: 'Files', web: 'Web' },
  tabsLabel: 'Chat items',
  seeAll: section => `See all ${section}`,
  seeAllShort: 'See all',
  artifactsIntro: 'Pages, graphics and long code made in this chat. A new version of the same piece stays in one card.',
  madeForYou: 'Made for you',
  allItems: 'All items',
  itemsButton: count => `${count} ${count === 1 ? 'item' : 'items'}`,
  emptyTitle: 'Nothing here yet',
  emptyDescription: 'Pages, files and links the agent makes or finds in this chat show up here.',
  noChat: 'Open a chat to see what it made.',
  version: version => `v${version}`,
  close: 'Close'
}

const fr: ChatItemsCopy = {
  pane: 'Éléments',
  tabs: { all: 'Tout', artifacts: 'Artefacts', files: 'Fichiers', web: 'Web' },
  tabsLabel: 'Éléments de la discussion',
  seeAll: section => `Tout voir : ${section}`,
  seeAllShort: 'Tout voir',
  artifactsIntro:
    'Pages, graphiques et code long créés dans cette discussion. Une nouvelle version du même élément reste dans la même carte.',
  madeForYou: 'Créé pour vous',
  allItems: 'Tous les éléments',
  itemsButton: count => `${count} ${count === 1 ? 'élément' : 'éléments'}`,
  emptyTitle: 'Rien pour l’instant',
  emptyDescription: 'Les pages, fichiers et liens que l’agent crée ou trouve dans cette discussion apparaissent ici.',
  noChat: 'Ouvrez une discussion pour voir ce qu’elle a produit.',
  version: version => `v${version}`,
  close: 'Fermer'
}

const de: ChatItemsCopy = {
  pane: 'Elemente',
  tabs: { all: 'Alle', artifacts: 'Artefakte', files: 'Dateien', web: 'Web' },
  tabsLabel: 'Elemente des Chats',
  seeAll: section => `Alle ${section} anzeigen`,
  seeAllShort: 'Alle anzeigen',
  artifactsIntro:
    'Seiten, Grafiken und langer Code aus diesem Chat. Eine neue Version desselben Stücks bleibt in einer Karte.',
  madeForYou: 'Für dich erstellt',
  allItems: 'Alle Elemente',
  itemsButton: count => `${count} ${count === 1 ? 'Element' : 'Elemente'}`,
  emptyTitle: 'Noch nichts hier',
  emptyDescription: 'Seiten, Dateien und Links, die der Agent in diesem Chat erstellt oder findet, erscheinen hier.',
  noChat: 'Öffne einen Chat, um zu sehen, was er erstellt hat.',
  version: version => `v${version}`,
  close: 'Schließen'
}

const es: ChatItemsCopy = {
  pane: 'Elementos',
  tabs: { all: 'Todo', artifacts: 'Artefactos', files: 'Archivos', web: 'Web' },
  tabsLabel: 'Elementos del chat',
  seeAll: section => `Ver todo: ${section}`,
  seeAllShort: 'Ver todo',
  artifactsIntro:
    'Páginas, gráficos y código largo creados en este chat. Una nueva versión de la misma pieza se queda en una tarjeta.',
  madeForYou: 'Hecho para ti',
  allItems: 'Todos los elementos',
  itemsButton: count => `${count} ${count === 1 ? 'elemento' : 'elementos'}`,
  emptyTitle: 'Todavía no hay nada',
  emptyDescription: 'Las páginas, archivos y enlaces que el agente crea o encuentra en este chat aparecen aquí.',
  noChat: 'Abre un chat para ver lo que creó.',
  version: version => `v${version}`,
  close: 'Cerrar'
}

const ja: ChatItemsCopy = {
  pane: 'アイテム',
  tabs: { all: 'すべて', artifacts: 'アーティファクト', files: 'ファイル', web: 'Web' },
  tabsLabel: 'チャットのアイテム',
  seeAll: section => `${section}をすべて表示`,
  seeAllShort: 'すべて表示',
  artifactsIntro:
    'このチャットで作られたページ、グラフィック、長いコード。同じものの新しいバージョンは 1 枚のカードにまとまります。',
  madeForYou: 'あなたのために作成',
  allItems: 'すべてのアイテム',
  itemsButton: count => `${count} 件`,
  emptyTitle: 'まだ何もありません',
  emptyDescription: 'このチャットでエージェントが作成または見つけたページ、ファイル、リンクがここに表示されます。',
  noChat: 'チャットを開くと、作成されたものが表示されます。',
  version: version => `v${version}`,
  close: '閉じる'
}

const zh: ChatItemsCopy = {
  pane: '项目',
  tabs: { all: '全部', artifacts: '作品', files: '文件', web: '网页' },
  tabsLabel: '对话项目',
  seeAll: section => `查看全部${section}`,
  seeAllShort: '查看全部',
  artifactsIntro: '此对话中生成的页面、图形和长代码。同一内容的新版本会保留在同一张卡片中。',
  madeForYou: '为你制作',
  allItems: '全部项目',
  itemsButton: count => `${count} 项`,
  emptyTitle: '这里还没有内容',
  emptyDescription: '智能体在此对话中创建或找到的页面、文件和链接会显示在这里。',
  noChat: '打开一个对话即可查看它生成的内容。',
  version: version => `v${version}`,
  close: '关闭'
}

const zhHant: ChatItemsCopy = {
  pane: '項目',
  tabs: { all: '全部', artifacts: '作品', files: '檔案', web: '網頁' },
  tabsLabel: '對話項目',
  seeAll: section => `查看全部${section}`,
  seeAllShort: '查看全部',
  artifactsIntro: '此對話中產生的頁面、圖形和長程式碼。同一內容的新版本會保留在同一張卡片中。',
  madeForYou: '為你製作',
  allItems: '全部項目',
  itemsButton: count => `${count} 項`,
  emptyTitle: '這裡還沒有內容',
  emptyDescription: '代理在此對話中建立或找到的頁面、檔案和連結會顯示在這裡。',
  noChat: '開啟一個對話即可查看它產生的內容。',
  version: version => `v${version}`,
  close: '關閉'
}

const ru: ChatItemsCopy = {
  pane: 'Элементы',
  tabs: { all: 'Все', artifacts: 'Артефакты', files: 'Файлы', web: 'Веб' },
  tabsLabel: 'Элементы чата',
  seeAll: section => `Показать все: ${section}`,
  seeAllShort: 'Показать все',
  artifactsIntro:
    'Страницы, графика и длинный код из этого чата. Новая версия того же элемента остаётся в одной карточке.',
  madeForYou: 'Создано для вас',
  allItems: 'Все элементы',
  itemsButton: count => `Элементов: ${count}`,
  emptyTitle: 'Здесь пока пусто',
  emptyDescription: 'Здесь появятся страницы, файлы и ссылки, которые агент создаёт или находит в этом чате.',
  noChat: 'Откройте чат, чтобы увидеть, что в нём создано.',
  version: version => `v${version}`,
  close: 'Закрыть'
}

const ar: ChatItemsCopy = {
  pane: 'العناصر',
  tabs: { all: 'الكل', artifacts: 'المخرجات', files: 'الملفات', web: 'الويب' },
  tabsLabel: 'عناصر المحادثة',
  seeAll: section => `عرض كل ${section}`,
  seeAllShort: 'عرض الكل',
  artifactsIntro:
    'الصفحات والرسومات والشيفرات الطويلة التي أُنشئت في هذه المحادثة. تبقى النسخة الجديدة من العنصر نفسه في بطاقة واحدة.',
  madeForYou: 'صُنع لك',
  allItems: 'كل العناصر',
  itemsButton: count => `العناصر: ${count}`,
  emptyTitle: 'لا شيء هنا بعد',
  emptyDescription: 'تظهر هنا الصفحات والملفات والروابط التي ينشئها الوكيل أو يجدها في هذه المحادثة.',
  noChat: 'افتح محادثة لترى ما أنشأته.',
  version: version => `v${version}`,
  close: 'إغلاق'
}

const COPY: Record<BundledLocale, ChatItemsCopy> = { en, fr, de, es, ja, zh, 'zh-hant': zhHant, ru, ar }

export function chatItemsCopy(locale: string): ChatItemsCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useChatItemsCopy(): ChatItemsCopy {
  return chatItemsCopy(useI18n().locale)
}
