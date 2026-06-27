export interface ImeKeyboardEventLike {
  isComposing?: boolean
  key?: string
  keyCode?: number
  which?: number
}

export function shouldLetImeHandleKeyDown(event: ImeKeyboardEventLike): boolean {
  return Boolean(event.isComposing || event.key === 'Process' || event.keyCode === 229 || event.which === 229)
}
