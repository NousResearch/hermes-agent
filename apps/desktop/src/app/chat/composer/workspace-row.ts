/**
 * Chrome shared by the composer's "where am I working" strip in both of its
 * states — the draft project/checkout pickers and the bound workspace summary.
 * It is the header row of the composer SURFACE (not a floating dock sibling),
 * so it inherits the input's fill, width and clipped top radius, and closes
 * with the same hairline the rest of the surface uses. Hover stays flat: the
 * strip is chrome, not a list row.
 */
export const workspaceRowClassName =
  'coding-status-bar min-h-7 rounded-t-[inherit] rounded-b-none border-b border-(--ui-stroke-tertiary) px-3.5 py-1.5 hover:bg-transparent'
