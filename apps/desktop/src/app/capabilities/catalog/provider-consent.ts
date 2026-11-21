import type { CatalogReview } from '@/types/skill-catalog'

/** Presentation guard, not mutation authority. The server revalidates every binding. */
export function reviewConsent(review: CatalogReview) {
  const required = review.scan.findings.filter(finding => finding.acknowledgement_required).map(finding => finding.id)
  const exactSet = new Set(review.acknowledgements_required).size === review.acknowledgements_required.length && new Set(required).size === required.length && required.length === review.acknowledgements_required.length && required.every(id => review.acknowledgements_required.includes(id))
  const complete = [review.confirmable, review.preview_complete, review.scan.complete, exactSet, review.review_token, review.version, review.content_digest, review.policy_revision, review.expires_at_ms].every(Boolean)
  const policyAllows = (review.scan.policy === 'allow' && !required.length) || (review.scan.policy === 'acknowledge' && required.length > 0)

  return { required, confirmable: complete && policyAllows }
}
