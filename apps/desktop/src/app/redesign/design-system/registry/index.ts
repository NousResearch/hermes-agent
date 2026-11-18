/**
 * The design system page's component registry: one entry per primitive (actions.tsx … icons.tsx, one
 * file per group), and the files in `components/ui/` the page does not render, each with why.
 * `registry.test.ts` fails when a file there is in neither.
 */
import { ACTIONS } from './actions'
import { FEEDBACK } from './feedback'
import { ICONS } from './icons'
import { INPUTS } from './inputs'
import { LAYOUT } from './layout'
import { OVERLAYS } from './overlays'
import { TEXT } from './text'
import type { GalleryEntry } from './types'

export { GALLERY_GROUPS, type GalleryEntry, type GalleryGroup } from './types'

export const GALLERY: readonly GalleryEntry[] = [
  ...ACTIONS,
  ...INPUTS,
  ...OVERLAYS,
  ...FEEDBACK,
  ...LAYOUT,
  ...TEXT,
  ...ICONS
]

/** `components/ui/` files (by stem) the page does not render, and why. */
export const NOT_SHOWN: Readonly<Record<string, string>> = {
  'card-stack': 'Stacks, swipes and promotes a live queue of pending approvals; it needs that queue.',
  command:
    'The list behind ⌘K and the searchable pickers. It scrolls its selected row into view as it mounts, which would move this page; press ⌘K to see it.',
  'connector-card': 'Renders a live connector record (status, accounts, actions); see Capabilities › Connectors.',
  'connector-logo': "Loads a service's icon from its website; this page makes no network requests.",
  'dialog-portal-context': 'Internal: tells popovers which container an open dialog portals into.',
  'drop-affordance': 'Class names for the file-drop sheet, painted only while a file is dragged over a drop target.',
  'fan-menu': "Opens from the composer's attach button with its own gesture; see it there.",
  favicon: "Loads a site's favicon over the network; this page makes no network requests.",
  'keyboard-first': 'Hooks that tell keyboard focus from pointer focus; no UI of its own.',
  'pane-tab': "The pane tree's tab chrome; selection, drag and close need a live zone.",
  'pane-tab-keyboard': 'Hook: arrow-key movement across pane tabs; no UI of its own.',
  'preview-document-frame': 'Hosts a previewed document in a sandboxed frame; it needs a real document.',
  'row-button':
    'An unstyled full-row click target: its look is the row that uses it (PageSidebarRow, SettingsLinkRow).',
  'sandboxed-frame': 'Hosts untrusted content in a sandboxed iframe; it needs content to host.',
  'setup-form-dialog':
    'The setup dialog for a connector or channel, driven by its live schema and save path; its fields are SetupFieldList.',
  sidebar: "The classic shell's sidebar frame; it needs the shell's provider and layout.",
  'tooltip-placement': 'The placement table Tip reads; no UI of its own.',
  'use-zoom-pan': 'The hook behind Zoomable; no UI of its own.',
  zoomable: 'Zoom and pan over live media (an image, a diagram) in a preview pane.'
}
