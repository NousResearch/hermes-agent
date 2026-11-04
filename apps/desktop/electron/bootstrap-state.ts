import { INSTALL_STAMP, installShape, type InstallStamp } from './install-stamp'

export function bootstrapSnapshot<State>(
  state: State,
  stamp: Readonly<InstallStamp> | null = INSTALL_STAMP
): State & { bundled: boolean } {
  return { ...state, bundled: installShape(stamp) === 'bundled' }
}
