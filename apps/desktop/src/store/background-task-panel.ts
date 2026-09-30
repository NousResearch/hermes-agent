import { atom } from 'nanostores'

export type TaskTab = 'tasks' | 'subagents' | 'terminals'

export const $backgroundPanelOpen = atom(false)
export const $backgroundPanelHeight = atom(280)
export const $backgroundPanelActiveTab = atom<TaskTab>('subagents')

export const toggleBackgroundPanel = () => {
  $backgroundPanelOpen.set(!$backgroundPanelOpen.get())
}

export const setBackgroundPanelHeight = (height: number) => {
  $backgroundPanelHeight.set(Math.max(120, Math.min(600, height)))
}

export const setBackgroundPanelTab = (tab: TaskTab) => {
  $backgroundPanelActiveTab.set(tab)
}
