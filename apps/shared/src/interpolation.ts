export const INTERPOLATION_RE = /\{!(.+?)\}/g

export const hasInterpolation = (s: string) => /\{!.+?\}/.test(s)
