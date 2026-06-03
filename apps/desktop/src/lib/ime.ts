interface ImeKeyboardEventLike {
  key: string
  nativeEvent?: {
    isComposing?: boolean
    keyCode?: number
  }
}

export function isImeComposing(event: ImeKeyboardEventLike): boolean {
  return event.nativeEvent?.isComposing === true || event.nativeEvent?.keyCode === 229 || event.key === 'Process'
}
