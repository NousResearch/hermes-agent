/** Whether a short viewport collapses THIS composer's control row to its Send button.
 *  Touch keeps a trigger so the model stays reachable once the keyboard is up. A fine pointer (a desktop window
 *  squeezed short) keeps the pill only while its OWN menu is open, so a search field's keyboard cannot unmount the
 *  menu mid-pick. Both branches answer for one composer — never for the pane next door. */
export function shortHeightCollapse(shortViewport: boolean, touchPointer: boolean, modelMenuOpen: boolean): boolean {
  return shortViewport && (touchPointer || !modelMenuOpen)
}
