/**
 * The redesign's icon vocabulary: every legacy glyph the app draws (Tabler alias or codicon name)
 * resolved to its Pixelarticons twin, the brand set the Figma frames are drawn with. The classic
 * shell never reads this module; `src/lib/icons.ts` and `Codicon` consult it only under the portal
 * shell.
 *
 * Where a Figma frame draws a control, the entry uses that frame's icon (node ids in comments);
 * elsewhere it is the closest semantic match in the free set. `KEEP_CLASSIC` is a deliberate
 * decision — a language/brand mark or a glyph the set has no honest equivalent for — so the classic
 * glyph renders, never a blank.
 */
import type { PixelIconName } from './pixel-icon-registry'

export const KEEP_CLASSIC = 'classic'

export type PixelMapping = PixelIconName | typeof KEEP_CLASSIC

/** Tabler aliases exported by `src/lib/icons.ts`. Typed against that module's export list in
 *  `icons.ts` itself (`satisfies Record<…>`), so an alias without an entry fails to compile. */
export const TABLER_PIXEL = {
  Activity: 'chart-line',
  AlertCircle: 'square-alert',
  AlertTriangle: 'warning-diamond',
  AppWindow: 'window-frame',
  Archive: 'archive', // settings nav "Sessions & History" (856:2522)
  ArchiveOff: 'archive',
  ArrowBarToDown: 'arrow-bar-down',
  ArrowBarToLeft: 'arrow-bar-left', // sidebar collapse (906 brand block, shell.md §3.1)
  ArrowBarToUp: 'arrow-bar-up',
  ArrowUp: 'arrow-up',
  ArrowUpRight: 'external-link',
  AtSign: 'at-sign',
  AudioLines: 'audio-waveform', // composer voice (906:19866)
  BarChart3: 'coins', // settings nav "Billing & Usage" (747:3868)
  Bell: 'bell', // settings nav "Notifications" (856:2443)
  Book: 'book-open',
  Bookmark: 'bookmark',
  BookmarkFilled: 'bookmark',
  Box: 'box',
  Brain: 'lightbulb',
  Bug: 'bug',
  Category: 'icon-category',
  Check: 'check',
  CheckCircle2: 'checkbox-on',
  CheckIcon: 'check',
  ChevronDown: 'chevron-down',
  ChevronDownIcon: 'chevron-down',
  ChevronLeft: 'chevron-left',
  ChevronLeftIcon: 'chevron-left',
  ChevronRight: 'chevron-right',
  ChevronRightIcon: 'chevron-right',
  ChevronUp: 'chevron-up',
  CircleIcon: 'circle',
  CircleLetterA: 'letter-a-circle',
  Clipboard: 'clipboard',
  Clock: 'clock',
  Cloud: 'cloud',
  Command: KEEP_CLASSIC, // the ⌘ key cap; the set has no command glyph
  Copy: 'copy',
  CopyIcon: 'copy',
  CornerDownLeft: 'corner-down-left',
  Cpu: 'cpu',
  CreditCard: 'credit-card',
  Database: 'database',
  Download: 'download',
  Ear: 'headphone',
  EarOff: 'volume-x',
  Egg: KEEP_CLASSIC, // pet hatching; no egg in the set
  Eject: 'arrow-up-box',
  ExternalLink: 'external-link',
  Eye: 'eye',
  EyeOff: 'eye-off',
  FileImage: 'image',
  FileText: 'file-text', // composer file chip (906:19836)
  Filter2: 'filter',
  FolderOpen: 'folder',
  Gateway: 'server', // settings nav "Gateways & Machines" (856:2501); classic keeps the globe
  GitBranch: 'git-branch',
  GitBranchIcon: 'git-branch',
  GitFork: 'git-branch',
  GitForkIcon: 'git-branch',
  Globe: 'globe',
  Hash: 'hash',
  HelpCircle: 'circle-question',
  ImageIcon: 'image',
  Info: 'circle-info', // settings nav "About & Updates" (747:3889)
  Keyboard: 'keyboard', // settings nav "Keyboard Shortcuts" (856:2471)
  KeyRound: 'key', // settings nav "Account & Plan" (747:3865)
  Layers3: 'section-copy',
  LayoutDashboard: 'grid-2x2-2',
  Link: 'link',
  Link2: 'link',
  LinkIcon: 'link',
  Loader2: 'loader',
  Loader2Icon: 'loader',
  Lock: 'lock', // vault "Unlock" (895:20997)
  LockOpen: 'unlock',
  LogIn: 'login',
  Mail: 'mail',
  Maximize: 'scale',
  Message: 'message', // sidebar "New Session" (906:19710)
  MessageCircle: 'comment',
  MessageCode: 'message-text',
  MessageQuestion: 'comment-text', // "Help & Feedback" (747:3892)
  MessageSquareText: 'message-text',
  Mic: 'mic', // composer (906:19863)
  MicOff: 'mic-off',
  Monitor: 'monitor',
  MonitorPlay: 'tv',
  Moon: 'moon',
  MoreHorizontal: 'more-horizontal', // `more-vertical` rotated 90° in the sidebar labels (906:19723)
  MoreHorizontalIcon: 'more-horizontal',
  MoreVertical: 'more-vertical',
  Network: 'modem',
  NotebookTabs: 'notebook',
  Package: 'package',
  Palette: 'colors-swatch',
  PanelBottom: 'dock',
  PanelLeftIcon: 'arrow-bar-left',
  PanelTop: 'layout',
  Pause: 'pause',
  PawPrint: 'dog',
  Pencil: 'pencil', // history row "Rename" (878:6057)
  PencilIcon: 'pencil',
  PencilLine: 'pencil',
  Pin: 'map-pin',
  Play: 'play',
  Plug: 'plug',
  Plus: 'plus', // band / page "New" buttons (906:19483, 747:3829)
  Power: 'power', // approval mode "Off" (895:20151)
  Puzzle: 'blocks',
  QrCode: 'qr-code',
  RefreshCw: 'refresh',
  RefreshCwIcon: 'refresh',
  RosetteDiscountCheck: 'checkbox-on',
  Save: 'save',
  Search: 'search', // history toolbar search (877:5952)
  SearchIcon: 'search',
  Send: 'send',
  Settings: 'settings-2', // band settings (906:19488)
  Settings2: 'settings-cog',
  ShieldLock: 'shield',
  SlidersHorizontal: 'sliders-horizontal',
  SmilePlusIcon: 'smile',
  Square: 'stop-solid', // only drawn filled, as the stop control
  Starmap: 'algorithm',
  SteeringWheel: 'joystick',
  StopFilled: 'stop-solid',
  Sun: 'sun',
  Terminal: 'terminal',
  Trash2: 'trash', // history "Delete…" (878:4860)
  Upload: 'upload',
  Users: 'users',
  Volume2: 'volume-2',
  Volume2Icon: 'volume-2',
  VolumeX: 'volume-x',
  VolumeXIcon: 'volume-x',
  Wrench: 'tools',
  X: 'close', // dialog / window close (906 Setup header, 878:7026)
  XIcon: 'close',
  Zap: 'zap', // composer (906:19853)
  ZapFilled: 'zap',
  ZoomIn: 'zoom-in',
  ZoomOut: 'zoom-out'
} as const satisfies Record<string, PixelMapping>

export type TablerAlias = keyof typeof TABLER_PIXEL

/** Codicon names the app renders (literals, tables, `codiconIcon()`, language/file icons). The
 *  completeness test derives the in-use set from the source and fails on any name missing here.
 *  Plugin-supplied names outside the table fall back to the classic glyph at runtime. */
export const CODICON_PIXEL: Readonly<Record<string, PixelMapping>> = {
  account: 'user', // rail avatar (906:19461)
  add: 'plus', // new tab (906:19483), composer attach (906:19860)
  agent: 'robot',
  alert: 'warning-diamond',
  archive: 'archive',
  'arrow-down': 'arrow-down',
  'arrow-left': 'arrow-left', // band back (906:19470)
  'arrow-right': 'arrow-right', // band forward (906:19472, the mirrored arrow-left)
  'arrow-small-right': 'arrow-right',
  'arrow-swap': 'sort-horizontal',
  'arrow-up': 'arrow-up',
  ask: 'circle-question',
  attach: 'attachment',
  beaker: 'test-tube',
  bell: 'bell', // rail bottom (906:19459)
  'bell-dot': 'bell-ring',
  'bell-slash': 'bell-off',
  blank: KEEP_CLASSIC, // an intentionally empty spacer glyph
  book: 'book-open',
  bookmark: 'bookmark',
  brain: 'lightbulb',
  broadcast: 'radio',
  browser: 'window-frame',
  bug: 'bug',
  build: 'tools',
  calendar: 'calendar',
  check: 'check',
  'check-all': 'check-double',
  checklist: 'checkbox-on',
  'chevron-down': 'chevron-down',
  'chevron-left': 'chevron-left',
  'chevron-right': 'chevron-right', // list rows (906:19726)
  'chevron-up': 'chevron-up',
  chip: 'cpu',
  'chrome-close': 'close', // window controls (906 Setup header)
  'chrome-maximize': 'square',
  'chrome-minimize': 'minus',
  'chrome-restore': 'copy',
  circle: 'circle',
  'circle-large-outline': 'circle',
  'circle-outline': 'circle',
  'circle-slash': 'cancel',
  claude: KEEP_CLASSIC, // brand mark
  'clear-all': 'delete',
  clippy: 'clipboard',
  clock: 'clock',
  close: 'close',
  'close-all': 'copy-x',
  cloud: 'cloud', // right rail (906 rail bottom)
  'cloud-download': 'download',
  'cloud-upload': 'upload',
  code: 'code',
  'color-mode': 'invert',
  comment: 'message', // composer model row (906:19845)
  'comment-discussion': 'message', // rail "Chats" (906:19421)
  compass: 'compass',
  console: 'terminal',
  copy: 'copy', // "Copy path" (877:6919)
  'credit-card': 'credit-card',
  dash: 'minus',
  dashboard: 'grid-2x2-2',
  database: 'database',
  debug: 'debug',
  'debug-breakpoint-data-unverified': 'circle',
  'debug-continue': 'play',
  'debug-disconnect': 'unlink',
  'debug-pause': 'pause',
  'debug-restart': 'refresh',
  'debug-step-over': 'redo',
  'debug-stop': 'stop',
  'device-camera': 'camera',
  'device-desktop': 'monitor',
  'device-mobile': 'smartphone',
  diff: 'git-commit',
  'diff-added': 'plus-box',
  'diff-modified': 'stop',
  'diff-removed': 'minus-box',
  'diff-renamed': 'arrow-right-box',
  discard: 'undo',
  download: 'download',
  edit: 'pencil', // "Rename" (878:6057)
  ellipsis: 'more-horizontal', // row ⋯ menus (906:19723)
  'empty-window': 'window-frame',
  error: 'square-alert', // error card (426:944)
  exclude: 'cancel',
  export: 'upload',
  extensions: 'blocks',
  eye: 'eye',
  'eye-closed': 'eye-off',
  file: 'file',
  'file-media': 'image',
  'file-pdf': 'file',
  files: 'file', // rail "Library" (1040:17979)
  filter: 'filter', // history "Filter" (877:5958)
  flag: 'flag',
  flame: 'fire',
  fold: 'collapse',
  folder: 'folder', // project rows (692:22423)
  'folder-library': 'library',
  'folder-opened': 'folder',
  forward: 'forward',
  gear: 'gear',
  gift: 'gift',
  'git-branch': 'git-branch',
  'git-compare': 'arrows-horizontal',
  'git-merge': 'git-merge',
  'git-pull-request': 'git-pull-request',
  'git-pull-request-closed': 'git-pull-request',
  'git-pull-request-draft': 'git-pull-request',
  github: 'github',
  globe: 'globe', // right rail (906 rail bottom)
  grabber: 'menu',
  graph: 'chart-line',
  gripper: 'more-vertical',
  heart: 'heart',
  history: 'clock',
  home: 'home',
  hubot: 'robot',
  inbox: 'inbox',
  info: 'circle-info',
  inspect: 'square-cursor',
  italic: KEEP_CLASSIC, // text formatting; the set has none
  json: 'braces',
  'kebab-vertical': 'more-vertical',
  key: 'key',
  keyboard: 'keyboard',
  layers: 'section-copy',
  layout: 'layout',
  'layout-menubar': 'layout',
  'layout-sidebar-left': 'arrow-bar-left',
  'layout-sidebar-right': 'album', // right-panel toggle (906 right rail, shell.md §9)
  'layout-statusbar': 'dock',
  lightbulb: 'lightbulb',
  link: 'link',
  'link-external': 'external-link',
  'list-filter': 'filter',
  'list-flat': 'bulletlist',
  'list-ordered': 'list-box',
  'list-selection': 'list-box',
  'list-tree': 'algorithm',
  'list-unordered': 'bulletlist',
  loading: 'loader',
  location: 'map-pin',
  lock: 'lock',
  mail: 'mail',
  'mail-read': 'mail-open',
  markdown: 'article',
  mcp: 'plug',
  mention: 'at-sign',
  menu: 'menu',
  mic: 'mic',
  'mic-off': 'mic-off',
  more: 'more-horizontal',
  move: 'move',
  'multiple-windows': 'app-windows',
  'new-folder': 'folder-plus',
  'new-session': 'pen-square', // agent row "New session" (622:1827)
  newline: 'corner-down-left', // composer send return glyph (906:19871)
  note: 'note',
  openai: KEEP_CLASSIC, // brand mark
  organization: 'users',
  output: 'script',
  package: 'package',
  paintcan: 'colors-swatch',
  pass: 'checkbox-on',
  'pass-filled': 'checkbox-on',
  pin: 'map-pin',
  pinned: 'map-pin',
  play: 'play',
  'play-circle': 'play',
  plug: 'plug',
  preview: 'eye',
  project: 'text-colums',
  pulse: 'chart-line',
  python: KEEP_CLASSIC, // language mark
  question: 'circle-question',
  'record-keys': 'keyboard',
  references: 'link',
  refresh: 'refresh',
  regex: 'code',
  remote: 'cloud-server',
  remove: 'close',
  rename: 'pencil',
  replace: 'sort-horizontal',
  reply: 'corner-up-left',
  repo: 'notebook',
  'repo-forked': 'git-branch',
  robot: 'robot', // rail "Agents" (906:19430)
  rocket: 'send',
  'root-folder': 'folder',
  ruby: KEEP_CLASSIC, // language mark
  run: 'play',
  'run-all': 'play',
  save: 'save',
  'screen-normal': 'collapse',
  search: 'search',
  selection: 'section',
  send: 'send',
  server: 'server',
  'server-process': 'server',
  settings: 'settings-2',
  'settings-gear': 'settings-2', // band settings (906:19488)
  share: 'share',
  shield: 'shield',
  'sign-in': 'login',
  smiley: 'smile',
  sparkle: 'sparkle', // capability banners (895:14662)
  'split-horizontal': 'proportions',
  'star-empty': 'star',
  'star-full': 'star',
  stop: 'stop',
  'stop-circle': 'stop',
  'symbol-color': 'colors-swatch',
  'symbol-misc': 'circle-square', // rail "Capabilities" (906:19436)
  'symbol-numeric': 'hash',
  sync: 'refresh',
  table: 'grid-3x3',
  target: 'target',
  telescope: 'search',
  terminal: 'terminal',
  'text-size': 'a-arrow-up',
  thinking: 'lightbulb',
  'three-bars': 'menu',
  tools: 'tools',
  trash: 'trash',
  'triangle-right': 'play',
  twitter: KEEP_CLASSIC, // brand mark
  unlock: 'unlock',
  verified: 'checkbox-on',
  'verified-filled': 'checkbox-on',
  vm: 'computer',
  vscode: KEEP_CLASSIC, // brand mark
  warning: 'warning-diamond',
  watch: 'date-time', // rail "Routines" (1040:17988)
  window: 'window-frame',
  worktree: 'git-branch',
  x: 'close',
  zap: 'zap',
  'zoom-out': 'zoom-out'
}

/** Glyphs whose meaning is a direction along the reading axis; they mirror in RTL. Up/down and
 *  rotation-driven disclosure chevrons are not here: their owners already handle RTL. */
export const RTL_MIRRORED: ReadonlySet<PixelIconName> = new Set<PixelIconName>([
  'arrow-left',
  'arrow-right',
  'arrow-bar-left',
  'arrow-right-box',
  'chevron-left',
  'chevron-right',
  'corner-down-left',
  'corner-up-left',
  'forward',
  'redo',
  'undo'
])

/** The portal glyph for a codicon name, or null when the classic glyph should render. */
export function pixelForCodicon(name: string): PixelIconName | null {
  const mapped = CODICON_PIXEL[name]

  return mapped && mapped !== KEEP_CLASSIC ? mapped : null
}
