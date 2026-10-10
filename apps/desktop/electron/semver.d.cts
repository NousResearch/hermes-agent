declare module 'semver' {
  export function compare(left: string, right: string): number

  export class SemVer {
    constructor(version: string)
    readonly version: string
  }
}
