import type { AuthMeResponse } from '@/lib/api'

export function shouldHideAuthWidget(identity: AuthMeResponse): boolean {
  return identity.provider === 'loopback'
}
