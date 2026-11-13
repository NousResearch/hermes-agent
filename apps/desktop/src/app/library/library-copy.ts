import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

import type { ArtifactFilter } from '../artifacts/artifact-utils'

/** The redesigned Library's own words (Figma 1163:17584 "Library — redesign"). Only the redesign
 *  renders them; classic keeps the catalog's `artifacts` strings. The page title is the redesign
 *  overlay's `simplifiedAgents.library`. */
export interface LibraryCopy {
  summary: (items: number, time: null | string) => string
  tabs: Record<ArtifactFilter, string>
  search: Record<ArtifactFilter, string>
  refresh: string
  refreshing: string
  sectionFilesLinks: string
  viewAll: string
  columns: { name: string; location: string; path: string; url: string; chat: string }
  range: (start: number, end: number, total: number, tab: ArtifactFilter) => string
  pages: string
  page: (page: number) => string
  previousPage: string
  nextPage: string
  position: (index: number, total: number) => string
  previousImage: string
  nextImage: string
  meta: { chat: string; created: string; path: string }
  copyPath: string
  download: string
  openChat: string
  close: string
  zoomIn: string
  zoomOut: string
  fit: string
  enlarge: string
  imageSource: (type: string, chat: string, when: string) => string
  thumbnails: string
  emptyTitle: string
  emptyDescription: string
  noMatchesTitle: string
  noMatchesDescription: (query: string) => string
  errorTitle: string
  errorDescription: string
  retry: string
  loading: string
  imageUnavailable: string
}

const en: LibraryCopy = {
  summary: (items, time) => `${items} ${items === 1 ? 'item' : 'items'}${time ? ` · Updated ${time}` : ''}`,
  tabs: { all: 'All', image: 'Images', file: 'Files', link: 'Links' },
  search: { all: 'Search library', image: 'Search images', file: 'Search files', link: 'Search links' },
  refresh: 'Refresh',
  refreshing: 'Refreshing',
  sectionFilesLinks: 'Files & links',
  viewAll: 'View all',
  columns: { name: 'Name', location: 'Location', path: 'Path', url: 'URL', chat: 'Chat' },
  range: (start, end, total, tab) =>
    `${start}–${end} of ${total} ${{ all: 'items', image: 'images', file: 'files', link: 'links' }[tab]}`,
  pages: 'Pages',
  page: page => `Page ${page}`,
  previousPage: 'Previous page',
  nextPage: 'Next page',
  position: (index, total) => `${index} of ${total}`,
  previousImage: 'Previous image',
  nextImage: 'Next image',
  meta: { chat: 'Chat', created: 'Created', path: 'Path' },
  copyPath: 'Copy path',
  download: 'Download',
  openChat: 'Open chat',
  close: 'Close',
  zoomIn: 'Zoom in',
  zoomOut: 'Zoom out',
  fit: 'Fit',
  enlarge: 'Open full size',
  imageSource: (type, chat, when) => `${type} · From “${chat}” · ${when}`,
  thumbnails: 'Images',
  emptyTitle: 'Nothing here yet',
  emptyDescription: 'Images, files and links from your recent chats show up here.',
  noMatchesTitle: 'No matches',
  noMatchesDescription: query => `Nothing here matches “${query}”.`,
  errorTitle: 'The library could not load',
  errorDescription: 'Your chats could not be checked. What was found before is still here.',
  retry: 'Try again',
  loading: 'Loading the library',
  imageUnavailable: 'Image unavailable'
}

const fr: LibraryCopy = {
  summary: (items, time) => `${items} ${items === 1 ? 'élément' : 'éléments'}${time ? ` · Mis à jour à ${time}` : ''}`,
  tabs: { all: 'Tout', image: 'Images', file: 'Fichiers', link: 'Liens' },
  search: {
    all: 'Rechercher dans la bibliothèque',
    image: 'Rechercher des images',
    file: 'Rechercher des fichiers',
    link: 'Rechercher des liens'
  },
  refresh: 'Actualiser',
  refreshing: 'Actualisation',
  sectionFilesLinks: 'Fichiers et liens',
  viewAll: 'Tout voir',
  columns: { name: 'Nom', location: 'Emplacement', path: 'Chemin', url: 'URL', chat: 'Discussion' },
  range: (start, end, total, tab) =>
    `${start}–${end} sur ${total} ${{ all: 'éléments', image: 'images', file: 'fichiers', link: 'liens' }[tab]}`,
  pages: 'Pages',
  page: page => `Page ${page}`,
  previousPage: 'Page précédente',
  nextPage: 'Page suivante',
  position: (index, total) => `${index} sur ${total}`,
  previousImage: 'Image précédente',
  nextImage: 'Image suivante',
  meta: { chat: 'Discussion', created: 'Créé', path: 'Chemin' },
  copyPath: 'Copier le chemin',
  download: 'Télécharger',
  openChat: 'Ouvrir la discussion',
  close: 'Fermer',
  zoomIn: 'Zoom avant',
  zoomOut: 'Zoom arrière',
  fit: 'Ajuster',
  enlarge: 'Afficher en taille réelle',
  imageSource: (type, chat, when) => `${type} · De « ${chat} » · ${when}`,
  thumbnails: 'Images',
  emptyTitle: 'Rien pour l’instant',
  emptyDescription: 'Les images, fichiers et liens de vos discussions récentes apparaissent ici.',
  noMatchesTitle: 'Aucun résultat',
  noMatchesDescription: query => `Rien ne correspond à « ${query} ».`,
  errorTitle: 'Impossible de charger la bibliothèque',
  errorDescription: 'Vos discussions n’ont pas pu être vérifiées. Ce qui avait été trouvé reste affiché.',
  retry: 'Réessayer',
  loading: 'Chargement de la bibliothèque',
  imageUnavailable: 'Image indisponible'
}

const de: LibraryCopy = {
  summary: (items, time) => `${items} ${items === 1 ? 'Eintrag' : 'Einträge'}${time ? ` · Aktualisiert ${time}` : ''}`,
  tabs: { all: 'Alle', image: 'Bilder', file: 'Dateien', link: 'Links' },
  search: {
    all: 'Bibliothek durchsuchen',
    image: 'Bilder durchsuchen',
    file: 'Dateien durchsuchen',
    link: 'Links durchsuchen'
  },
  refresh: 'Aktualisieren',
  refreshing: 'Wird aktualisiert',
  sectionFilesLinks: 'Dateien & Links',
  viewAll: 'Alle anzeigen',
  columns: { name: 'Name', location: 'Ort', path: 'Pfad', url: 'URL', chat: 'Chat' },
  range: (start, end, total, tab) =>
    `${start}–${end} von ${total} ${{ all: 'Einträgen', image: 'Bildern', file: 'Dateien', link: 'Links' }[tab]}`,
  pages: 'Seiten',
  page: page => `Seite ${page}`,
  previousPage: 'Vorherige Seite',
  nextPage: 'Nächste Seite',
  position: (index, total) => `${index} von ${total}`,
  previousImage: 'Vorheriges Bild',
  nextImage: 'Nächstes Bild',
  meta: { chat: 'Chat', created: 'Erstellt', path: 'Pfad' },
  copyPath: 'Pfad kopieren',
  download: 'Herunterladen',
  openChat: 'Chat öffnen',
  close: 'Schließen',
  zoomIn: 'Vergrößern',
  zoomOut: 'Verkleinern',
  fit: 'Einpassen',
  enlarge: 'In voller Größe öffnen',
  imageSource: (type, chat, when) => `${type} · Aus „${chat}“ · ${when}`,
  thumbnails: 'Bilder',
  emptyTitle: 'Noch nichts hier',
  emptyDescription: 'Bilder, Dateien und Links aus deinen letzten Chats erscheinen hier.',
  noMatchesTitle: 'Keine Treffer',
  noMatchesDescription: query => `Nichts passt zu „${query}“.`,
  errorTitle: 'Die Bibliothek konnte nicht geladen werden',
  errorDescription: 'Deine Chats konnten nicht geprüft werden. Bisher Gefundenes bleibt sichtbar.',
  retry: 'Erneut versuchen',
  loading: 'Bibliothek wird geladen',
  imageUnavailable: 'Bild nicht verfügbar'
}

const es: LibraryCopy = {
  summary: (items, time) => `${items} ${items === 1 ? 'elemento' : 'elementos'}${time ? ` · Actualizado ${time}` : ''}`,
  tabs: { all: 'Todo', image: 'Imágenes', file: 'Archivos', link: 'Enlaces' },
  search: {
    all: 'Buscar en la biblioteca',
    image: 'Buscar imágenes',
    file: 'Buscar archivos',
    link: 'Buscar enlaces'
  },
  refresh: 'Actualizar',
  refreshing: 'Actualizando',
  sectionFilesLinks: 'Archivos y enlaces',
  viewAll: 'Ver todo',
  columns: { name: 'Nombre', location: 'Ubicación', path: 'Ruta', url: 'URL', chat: 'Chat' },
  range: (start, end, total, tab) =>
    `${start}–${end} de ${total} ${{ all: 'elementos', image: 'imágenes', file: 'archivos', link: 'enlaces' }[tab]}`,
  pages: 'Páginas',
  page: page => `Página ${page}`,
  previousPage: 'Página anterior',
  nextPage: 'Página siguiente',
  position: (index, total) => `${index} de ${total}`,
  previousImage: 'Imagen anterior',
  nextImage: 'Imagen siguiente',
  meta: { chat: 'Chat', created: 'Creado', path: 'Ruta' },
  copyPath: 'Copiar ruta',
  download: 'Descargar',
  openChat: 'Abrir chat',
  close: 'Cerrar',
  zoomIn: 'Acercar',
  zoomOut: 'Alejar',
  fit: 'Ajustar',
  enlarge: 'Abrir a tamaño completo',
  imageSource: (type, chat, when) => `${type} · De «${chat}» · ${when}`,
  thumbnails: 'Imágenes',
  emptyTitle: 'Todavía no hay nada',
  emptyDescription: 'Las imágenes, archivos y enlaces de tus chats recientes aparecen aquí.',
  noMatchesTitle: 'Sin resultados',
  noMatchesDescription: query => `Nada coincide con «${query}».`,
  errorTitle: 'No se pudo cargar la biblioteca',
  errorDescription: 'No se pudieron revisar tus chats. Lo encontrado antes sigue aquí.',
  retry: 'Reintentar',
  loading: 'Cargando la biblioteca',
  imageUnavailable: 'Imagen no disponible'
}

const ja: LibraryCopy = {
  summary: (items, time) => `${items} 件${time ? ` · ${time} に更新` : ''}`,
  tabs: { all: 'すべて', image: '画像', file: 'ファイル', link: 'リンク' },
  search: { all: 'ライブラリを検索', image: '画像を検索', file: 'ファイルを検索', link: 'リンクを検索' },
  refresh: '更新',
  refreshing: '更新中',
  sectionFilesLinks: 'ファイルとリンク',
  viewAll: 'すべて表示',
  columns: { name: '名前', location: '場所', path: 'パス', url: 'URL', chat: 'チャット' },
  range: (start, end, total) => `${total} 件中 ${start}–${end} 件`,
  pages: 'ページ',
  page: page => `${page} ページ`,
  previousPage: '前のページ',
  nextPage: '次のページ',
  position: (index, total) => `${index} / ${total}`,
  previousImage: '前の画像',
  nextImage: '次の画像',
  meta: { chat: 'チャット', created: '作成日時', path: 'パス' },
  copyPath: 'パスをコピー',
  download: 'ダウンロード',
  openChat: 'チャットを開く',
  close: '閉じる',
  zoomIn: '拡大',
  zoomOut: '縮小',
  fit: '全体表示',
  enlarge: '原寸で開く',
  imageSource: (type, chat, when) => `${type} · 「${chat}」から · ${when}`,
  thumbnails: '画像',
  emptyTitle: 'まだ何もありません',
  emptyDescription: '最近のチャットの画像、ファイル、リンクがここに表示されます。',
  noMatchesTitle: '一致なし',
  noMatchesDescription: query => `「${query}」に一致するものはありません。`,
  errorTitle: 'ライブラリを読み込めませんでした',
  errorDescription: 'チャットを確認できませんでした。以前に見つかった項目は表示されたままです。',
  retry: '再試行',
  loading: 'ライブラリを読み込み中',
  imageUnavailable: '画像を表示できません'
}

const zh: LibraryCopy = {
  summary: (items, time) => `${items} 项${time ? ` · 更新于 ${time}` : ''}`,
  tabs: { all: '全部', image: '图片', file: '文件', link: '链接' },
  search: { all: '搜索资料库', image: '搜索图片', file: '搜索文件', link: '搜索链接' },
  refresh: '刷新',
  refreshing: '正在刷新',
  sectionFilesLinks: '文件和链接',
  viewAll: '查看全部',
  columns: { name: '名称', location: '位置', path: '路径', url: 'URL', chat: '对话' },
  range: (start, end, total) => `第 ${start}–${end} 项，共 ${total} 项`,
  pages: '页码',
  page: page => `第 ${page} 页`,
  previousPage: '上一页',
  nextPage: '下一页',
  position: (index, total) => `${index} / ${total}`,
  previousImage: '上一张',
  nextImage: '下一张',
  meta: { chat: '对话', created: '创建时间', path: '路径' },
  copyPath: '复制路径',
  download: '下载',
  openChat: '打开对话',
  close: '关闭',
  zoomIn: '放大',
  zoomOut: '缩小',
  fit: '适应',
  enlarge: '查看原图',
  imageSource: (type, chat, when) => `${type} · 来自“${chat}” · ${when}`,
  thumbnails: '图片',
  emptyTitle: '这里还没有内容',
  emptyDescription: '最近对话中的图片、文件和链接会显示在这里。',
  noMatchesTitle: '没有匹配项',
  noMatchesDescription: query => `没有与“${query}”匹配的内容。`,
  errorTitle: '无法加载资料库',
  errorDescription: '无法检查你的对话。之前找到的内容仍会保留。',
  retry: '重试',
  loading: '正在加载资料库',
  imageUnavailable: '图片不可用'
}

const zhHant: LibraryCopy = {
  summary: (items, time) => `${items} 項${time ? ` · 更新於 ${time}` : ''}`,
  tabs: { all: '全部', image: '圖片', file: '檔案', link: '連結' },
  search: { all: '搜尋資料庫', image: '搜尋圖片', file: '搜尋檔案', link: '搜尋連結' },
  refresh: '重新整理',
  refreshing: '正在重新整理',
  sectionFilesLinks: '檔案和連結',
  viewAll: '查看全部',
  columns: { name: '名稱', location: '位置', path: '路徑', url: 'URL', chat: '對話' },
  range: (start, end, total) => `第 ${start}–${end} 項，共 ${total} 項`,
  pages: '頁碼',
  page: page => `第 ${page} 頁`,
  previousPage: '上一頁',
  nextPage: '下一頁',
  position: (index, total) => `${index} / ${total}`,
  previousImage: '上一張',
  nextImage: '下一張',
  meta: { chat: '對話', created: '建立時間', path: '路徑' },
  copyPath: '複製路徑',
  download: '下載',
  openChat: '開啟對話',
  close: '關閉',
  zoomIn: '放大',
  zoomOut: '縮小',
  fit: '符合',
  enlarge: '查看原圖',
  imageSource: (type, chat, when) => `${type} · 來自「${chat}」 · ${when}`,
  thumbnails: '圖片',
  emptyTitle: '這裡還沒有內容',
  emptyDescription: '最近對話中的圖片、檔案和連結會顯示在這裡。',
  noMatchesTitle: '沒有相符項目',
  noMatchesDescription: query => `沒有與「${query}」相符的內容。`,
  errorTitle: '無法載入資料庫',
  errorDescription: '無法檢查你的對話。先前找到的內容仍會保留。',
  retry: '重試',
  loading: '正在載入資料庫',
  imageUnavailable: '圖片無法使用'
}

const ru: LibraryCopy = {
  summary: (items, time) => `Элементов: ${items}${time ? ` · Обновлено в ${time}` : ''}`,
  tabs: { all: 'Все', image: 'Изображения', file: 'Файлы', link: 'Ссылки' },
  search: {
    all: 'Поиск в библиотеке',
    image: 'Поиск изображений',
    file: 'Поиск файлов',
    link: 'Поиск ссылок'
  },
  refresh: 'Обновить',
  refreshing: 'Обновление',
  sectionFilesLinks: 'Файлы и ссылки',
  viewAll: 'Показать все',
  columns: { name: 'Название', location: 'Расположение', path: 'Путь', url: 'URL', chat: 'Чат' },
  range: (start, end, total) => `${start}–${end} из ${total}`,
  pages: 'Страницы',
  page: page => `Страница ${page}`,
  previousPage: 'Предыдущая страница',
  nextPage: 'Следующая страница',
  position: (index, total) => `${index} из ${total}`,
  previousImage: 'Предыдущее изображение',
  nextImage: 'Следующее изображение',
  meta: { chat: 'Чат', created: 'Создано', path: 'Путь' },
  copyPath: 'Копировать путь',
  download: 'Скачать',
  openChat: 'Открыть чат',
  close: 'Закрыть',
  zoomIn: 'Увеличить',
  zoomOut: 'Уменьшить',
  fit: 'По размеру',
  enlarge: 'Открыть в полном размере',
  imageSource: (type, chat, when) => `${type} · Из «${chat}» · ${when}`,
  thumbnails: 'Изображения',
  emptyTitle: 'Здесь пока пусто',
  emptyDescription: 'Здесь появятся изображения, файлы и ссылки из ваших недавних чатов.',
  noMatchesTitle: 'Ничего не найдено',
  noMatchesDescription: query => `Нет совпадений с «${query}».`,
  errorTitle: 'Не удалось загрузить библиотеку',
  errorDescription: 'Не удалось проверить чаты. Найденное ранее остаётся на месте.',
  retry: 'Повторить',
  loading: 'Загрузка библиотеки',
  imageUnavailable: 'Изображение недоступно'
}

const ar: LibraryCopy = {
  summary: (items, time) => `العناصر: ${items}${time ? ` · آخر تحديث ${time}` : ''}`,
  tabs: { all: 'الكل', image: 'الصور', file: 'الملفات', link: 'الروابط' },
  search: { all: 'البحث في المكتبة', image: 'البحث في الصور', file: 'البحث في الملفات', link: 'البحث في الروابط' },
  refresh: 'تحديث',
  refreshing: 'جارٍ التحديث',
  sectionFilesLinks: 'الملفات والروابط',
  viewAll: 'عرض الكل',
  columns: { name: 'الاسم', location: 'الموقع', path: 'المسار', url: 'الرابط', chat: 'المحادثة' },
  range: (start, end, total) => `${start}–${end} من ${total}`,
  pages: 'الصفحات',
  page: page => `الصفحة ${page}`,
  previousPage: 'الصفحة السابقة',
  nextPage: 'الصفحة التالية',
  position: (index, total) => `${index} من ${total}`,
  previousImage: 'الصورة السابقة',
  nextImage: 'الصورة التالية',
  meta: { chat: 'المحادثة', created: 'تاريخ الإنشاء', path: 'المسار' },
  copyPath: 'نسخ المسار',
  download: 'تنزيل',
  openChat: 'فتح المحادثة',
  close: 'إغلاق',
  zoomIn: 'تكبير',
  zoomOut: 'تصغير',
  fit: 'ملاءمة',
  enlarge: 'فتح بالحجم الكامل',
  imageSource: (type, chat, when) => `${type} · من «${chat}» · ${when}`,
  thumbnails: 'الصور',
  emptyTitle: 'لا شيء هنا بعد',
  emptyDescription: 'تظهر هنا الصور والملفات والروابط من محادثاتك الأخيرة.',
  noMatchesTitle: 'لا توجد نتائج',
  noMatchesDescription: query => `لا شيء يطابق «${query}».`,
  errorTitle: 'تعذّر تحميل المكتبة',
  errorDescription: 'تعذّر فحص محادثاتك. ما عُثر عليه سابقًا لا يزال هنا.',
  retry: 'إعادة المحاولة',
  loading: 'جارٍ تحميل المكتبة',
  imageUnavailable: 'الصورة غير متاحة'
}

const COPY: Record<BundledLocale, LibraryCopy> = { en, fr, de, es, ja, zh, 'zh-hant': zhHant, ru, ar }

export function libraryCopy(locale: string): LibraryCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useLibraryCopy(): LibraryCopy {
  return libraryCopy(useI18n().locale)
}
