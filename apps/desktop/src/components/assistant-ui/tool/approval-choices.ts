import type { ApprovalRequest } from '@/store/prompts'

export type ApprovalChoice = 'once' | 'session' | 'always' | 'deny'

// Running order, safest-to-widest then refuse. Every surface that offers the
// choices offers them in this order.
const ORDER: readonly ApprovalChoice[] = ['once', 'session', 'always', 'deny']

/**
 * Which answers this request will take.
 *
 * The backend names them when it wants to narrow them, so the list is its word
 * where it gave one. Two cases it leaves to us: a smart-denied command offers
 * only once-or-refuse, and a permanent allow the backend will not honour (a
 * credential-shaped command Tirith flagged) drops "always".
 */
export function approvalChoices(request: ApprovalRequest): ApprovalChoice[] {
  const named = request.choices ?? (request.smartDenied ? ['once', 'deny'] : null)

  if (named) {
    return ORDER.filter(choice => named.includes(choice))
  }

  return request.allowPermanent === false ? ORDER.filter(choice => choice !== 'always') : [...ORDER]
}
