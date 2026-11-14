import { atom } from 'nanostores'

/** Whether the shell shows Bots as a sidebar pane. The app sets it (the redesign's Agents page
 *  hosts Bots instead); it is this bundled plugin's private seam, not plugin SDK surface. */
export const $botsPaneHosted = atom(true)
