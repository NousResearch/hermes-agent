import { atom } from 'nanostores'

/** Whether the project switcher dialog is open. */
export const $projectSwitcherOpen = atom(false)

export const openProjectSwitcher = (): void => $projectSwitcherOpen.set(true)
export const setProjectSwitcherOpen = (open: boolean): void => $projectSwitcherOpen.set(open)
