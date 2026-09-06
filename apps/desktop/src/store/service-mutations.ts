import { serviceMutationRequest, type ServiceMutationRequest } from '@hermes/shared'

import { translateNow } from '@/i18n'
import { confirm } from '@/store/confirm'

export async function confirmServiceMutation(
  action: 'restart' | 'update',
  title?: string
): Promise<ServiceMutationRequest | null> {
  const confirmation = action === 'restart' ? 'RESTART' : 'UPDATE'

  const accepted = await confirm({
    title: title ?? translateNow(action === 'restart' ? 'commandCenter.restartGateway' : 'commandCenter.updateHermes'),
    typedConfirmation: confirmation
  })

  return accepted ? serviceMutationRequest(confirmation) : null
}
