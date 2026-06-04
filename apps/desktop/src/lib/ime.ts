type ImeKeyboardLikeEvent = {
  isComposing?: boolean
  keyCode?: number
  nativeEvent?: {
    isComposing?: boolean
    keyCode?: number
  }
}

export const isImeComposing = (event: ImeKeyboardLikeEvent): boolean => {
  const nativeEvent = event.nativeEvent

  return Boolean(nativeEvent?.isComposing || event.isComposing || nativeEvent?.keyCode === 229 || event.keyCode === 229)
}
