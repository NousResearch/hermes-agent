// The SDK's design-language primitives, re-exported from `./index`.

export { Badge } from '@/components/ui/badge'
export { Button } from '@/components/ui/button'
export { Checkbox } from '@/components/ui/checkbox'
export { Codicon } from '@/components/ui/codicon'
/** THE color picker — swatch grid plus a clear row that means "back to the
 *  deterministic color". Feed it `PROFILE_SWATCHES` so a hand-picked color
 *  shares the generated palette's saturation and lightness; a bespoke grid of
 *  literal hex drifts off-theme the moment the palette moves. */
export { ColorSwatches } from '@/components/ui/color-swatches'
/** Searchable picker list (cmdk). Pair `variant="menu"` with
 *  `<PopoverContent variant="menu">` for a combobox that reads like a Select. */
export {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandItemCheck,
  CommandList
} from '@/components/ui/command'
export { ConfirmDialog } from '@/components/ui/confirm-dialog'
export {
  ContextMenu,
  ContextMenuCheckboxItem,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  // Submenus: Bot Mode files a bot into a user section from its row menu, and
  // a flat list of every folder would swamp the items already there.
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
/** The shared field look (Input / SelectTrigger), for a custom trigger button. */
export { controlVariants } from '@/components/ui/control'
export { CopyButton } from '@/components/ui/copy-button'
export { DecodeText } from '@/components/ui/decode-text'
export {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
/** The caret every collapsible section in core uses — points right when closed
 *  and rotates down when open, so the motion matches the rest of the app. Swap
 *  a hand-written `chevron-down`/`chevron-right` ternary for this. */
export { DisclosureCaret } from '@/components/ui/disclosure-caret'
export {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
export { EmptyState } from '@/components/ui/empty-state'
export { ErrorState } from '@/components/ui/error-state'
export { FadeScroll } from '@/components/ui/fade-scroll'
export { GlyphSpinner } from '@/components/ui/glyph-spinner'
export { Input } from '@/components/ui/input'
export { Kbd, KbdGroup } from '@/components/ui/kbd'
/** The app's canonical loader (animated curves; `lemniscate-bloom` for long
 *  page loads) — the same one every core page uses. */
export { Loader, type LoaderType } from '@/components/ui/loader'
export { LogView } from '@/components/ui/log-view'
export { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
/** Full-row / region click target. Imposes NO styling — the caller keeps its own
 *  layout classes — it just bakes in `type="button"` and a stable `data-slot`.
 *  Use it for rows and regions; `Button` is for ordinary compact actions. */
export { RowButton } from '@/components/ui/row-button'
/** The sanctioned embed primitive for external web content: a sandboxed
 *  iframe (opaque origin, `allow-scripts` by default) — never a raw
 *  `<webview>`, which would land on the app's own preview partition. */
export { SandboxedFrame, type SandboxedFrameProps } from '@/components/ui/sandboxed-frame'
export { ScrollArea } from '@/components/ui/scroll-area'
export { SearchField } from '@/components/ui/search-field'
export { SegmentedControl } from '@/components/ui/segmented-control'
export { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
export { Separator } from '@/components/ui/separator'
export { Skeleton } from '@/components/ui/skeleton'
export { Switch } from '@/components/ui/switch'
export { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
export { Textarea } from '@/components/ui/textarea'
export { Tip, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
