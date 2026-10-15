import { browserMessage } from './messages'

export function unsupported(_name: string): never {
  throw new Error(browserMessage('desktopOnly'))
}

export function unsupportedAsync(_name: string): Promise<never> {
  return Promise.reject(new Error(browserMessage('desktopOnly')))
}

export const noEvent = () => () => {}
