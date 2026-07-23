export function windowTitleBarStyle(isWsl: boolean): 'default' | 'hidden' {
  return isWsl ? 'default' : 'hidden'
}
