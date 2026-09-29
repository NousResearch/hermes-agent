import { atom } from 'nanostores'

export interface BrandSession {
  signedIn: boolean
  email: string
  appIds: string[]
  activeAppId: string
  isSuper: boolean
}

export const signedOutBrand = (): BrandSession => ({
  signedIn: false,
  email: '',
  appIds: [],
  activeAppId: '',
  isSuper: false
})

export const $brandSession = atom<BrandSession>(signedOutBrand())

/** Fields session.create and session.resume forward. Empty until the portal login is loaded. */
export function brandCreateFields(): Record<string, unknown> {
  const brand = $brandSession.get()

  if (!brand.signedIn) {
    return {}
  }

  return {
    brand_app_id: brand.activeAppId,
    brand_app_ids: brand.appIds,
    brand_email: brand.email,
    brand_is_super: brand.isSuper
  }
}
