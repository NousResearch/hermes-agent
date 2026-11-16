import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import { type ButterbarItem, useButterbar } from '@/store/butterbar'
import { $freeTierStatus } from '@/store/free-tier'

const TERMS_URL = 'https://portal.nousresearch.com/terms'
const PRIVACY_URL = 'https://portal.nousresearch.com/privacy'

function TermsNotice() {
  const copy = useI18n().t.butterbar.legal

  return (
    <>
      {copy.before}
      <ExternalLink href={TERMS_URL} native>
        {copy.terms}
      </ExternalLink>
      {copy.between}
      <ExternalLink href={PRIVACY_URL} native>
        {copy.privacy}
      </ExternalLink>
      {copy.after}
    </>
  )
}

// Module-level so the item keeps one identity across renders.
const TERMS_BUTTERBAR: ButterbarItem = { id: 'portal-terms', node: <TermsNotice />, priority: -1, tone: 'neutral' }

/** Accountless free-tier users get a persistent link to the Terms and Privacy
 *  Policy. Not closeable: it stands for as long as they use the free tier. */
export function TermsButterbar() {
  const freeTier = useStore($freeTierStatus)

  useButterbar(freeTier?.available ? TERMS_BUTTERBAR : null)

  return null
}
