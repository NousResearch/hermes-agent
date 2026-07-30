/**
 * The composer's reference vocabulary: one place that decides what a `@file:`,
 * a `@folder:`, a picked `/skill`, or any other reference LOOKS like — its
 * icon, its accent, and the word for its kind.
 *
 * Both surfaces that show a reference read from this table:
 *
 *   - the trigger popover row (browsing for one)
 *   - the chip (having picked one)
 *
 * so a thing is the same color with the same glyph wherever you meet it. They
 * used to be two hand-maintained icon maps and two unrelated row layouts, which
 * is why `@` and `/` looked like features from different apps.
 */

/** Every kind of thing the composer can reference. */
export type ReferenceKind =
  | 'file'
  | 'folder'
  | 'url'
  | 'image'
  | 'tool'
  | 'line'
  | 'terminal'
  | 'session'
  | 'git'
  | 'diff'
  | 'staged'
  | 'command'
  | 'skill'
  | 'theme'
  | 'emoji'
  | 'other'

interface ReferenceStyle {
  /** Codicon name — the popover row's leading glyph. */
  codicon: string
  /** Tabler outline path data — the chip's inline SVG. */
  paths: string[]
  /** Chip text color. Chips are text-only now, so this carries the whole
   *  signal for "what kind of thing is this". */
  color: string
  /** Section label when the popover groups by this kind. */
  label: string
}

// Accents map to the theme's own tokens rather than raw hues, so a reference
// keeps its meaning across every skin. Files/folders/paths share the neutral
// secondary text color — they're the common case and shouldn't shout; the
// things that ACT (commands, skills) take the accent colors.
const NEUTRAL = 'text-(--ui-text-secondary)'
const ACCENT = 'text-[color-mix(in_srgb,var(--ui-accent)_82%,var(--foreground))]'
const WARM = 'text-[color-mix(in_srgb,var(--ui-warm)_82%,var(--foreground))]'
const SECONDARY = 'text-[color-mix(in_srgb,var(--ui-accent-secondary)_82%,var(--foreground))]'

const FILE_PATHS = [
  'M14 3v4a1 1 0 0 0 1 1h4',
  'M17 21h-10a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2h7l5 5v11a2 2 0 0 1 -2 2',
  'M9 9l1 0',
  'M9 13l6 0',
  'M9 17l6 0'
]

const TERMINAL_PATHS = ['M5 7l5 5l-5 5', 'M12 19l7 0']

export const REFERENCE_STYLES: Record<ReferenceKind, ReferenceStyle> = {
  file: { codicon: 'file', paths: FILE_PATHS, color: NEUTRAL, label: 'Files' },
  folder: {
    codicon: 'folder',
    paths: [
      'M5 19l2.757 -7.351a1 1 0 0 1 .936 -.649h12.307a1 1 0 0 1 .986 1.164l-.996 5.211a2 2 0 0 1 -1.964 1.625h-14.026a2 2 0 0 1 -2 -2v-11a2 2 0 0 1 2 -2h4l3 3h7a2 2 0 0 1 2 2v2'
    ],
    color: NEUTRAL,
    label: 'Folders'
  },
  url: {
    codicon: 'globe',
    paths: [
      'M9 15l6 -6',
      'M11 6l.463 -.536a5 5 0 0 1 7.071 7.072l-.534 .464',
      'M13 18l-.397 .534a5.068 5.068 0 0 1 -7.127 0a4.972 4.972 0 0 1 0 -7.071l.524 -.463'
    ],
    color: SECONDARY,
    label: 'Links'
  },
  image: {
    codicon: 'file-media',
    paths: [
      'M15 8h.01',
      'M3 6a3 3 0 0 1 3 -3h12a3 3 0 0 1 3 3v12a3 3 0 0 1 -3 3h-12a3 3 0 0 1 -3 -3v-12',
      'M3 16l5 -5c.928 -.893 2.072 -.893 3 0l5 5',
      'M14 14l1 -1c.928 -.893 2.072 -.893 3 0l3 3'
    ],
    color: SECONDARY,
    label: 'Images'
  },
  tool: {
    codicon: 'tools',
    paths: ['M7 10h3v-3l-3.5 -3.5a6 6 0 0 1 8 8l6 6a2 2 0 0 1 -3 3l-6 -6a6 6 0 0 1 -8 -8l3.5 3.5'],
    color: ACCENT,
    label: 'Tools'
  },
  line: {
    codicon: 'list-selection',
    paths: ['M5 9l14 0', 'M5 15l14 0', 'M11 4l-4 16', 'M17 4l-4 16'],
    color: NEUTRAL,
    label: 'Lines'
  },
  terminal: { codicon: 'terminal', paths: TERMINAL_PATHS, color: NEUTRAL, label: 'Terminal' },
  session: {
    codicon: 'comment-discussion',
    paths: ['M4 4h16v2.172a2 2 0 0 1 -.586 1.414l-4.414 4.414v7l-6 2v-8.5l-4.48 -4.928a2 2 0 0 1 -.52 -1.345v-2.227'],
    color: SECONDARY,
    label: 'Sessions'
  },
  git: { codicon: 'git-branch', paths: ['M7 18l0 -12', 'M7 8a2 2 0 1 0 0 -4a2 2 0 0 0 0 4'], color: WARM, label: 'Git' },
  diff: { codicon: 'diff', paths: ['M12 5l0 14', 'M5 12l14 0'], color: WARM, label: 'Changes' },
  staged: { codicon: 'diff-added', paths: ['M12 5l0 14', 'M5 12l14 0'], color: WARM, label: 'Staged' },
  command: { codicon: 'terminal', paths: TERMINAL_PATHS, color: ACCENT, label: 'Commands' },
  skill: { codicon: 'zap', paths: ['M13 3l0 7l6 0l-8 11l0 -7l-6 0l8 -11'], color: WARM, label: 'Skills' },
  theme: {
    codicon: 'symbol-color',
    paths: [
      'M3 21v-4a4 4 0 1 1 4 4h-4',
      'M21 3a16 16 0 0 0 -12.8 10.2',
      'M21 3a16 16 0 0 1 -10.2 12.8',
      'M10.6 9a9 9 0 0 1 4.4 4.4'
    ],
    color: SECONDARY,
    label: 'Themes'
  },
  emoji: { codicon: 'smiley', paths: [], color: NEUTRAL, label: 'Emoji' },
  other: { codicon: 'symbol-misc', paths: FILE_PATHS, color: NEUTRAL, label: 'Other' }
}

const KNOWN = new Set(Object.keys(REFERENCE_STYLES))

/** Coerce any incoming type string to a kind we have a style for. */
export function referenceKind(type: string | undefined): ReferenceKind {
  return type && KNOWN.has(type) ? (type as ReferenceKind) : 'other'
}

export function referenceStyle(type: string | undefined): ReferenceStyle {
  return REFERENCE_STYLES[referenceKind(type)]
}
