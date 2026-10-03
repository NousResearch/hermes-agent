export function unsupported(name: string): never {
  throw new Error(`${name} is available only in the desktop application`)
}

export function unsupportedAsync(name: string): Promise<never> {
  return Promise.reject(new Error(`${name} is available only in the desktop application`))
}

export const noEvent = () => () => {}
