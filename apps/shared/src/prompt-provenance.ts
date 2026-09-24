/** Intent evidence, not permission. Only a deliberate composer submit may produce
 * this field, before sanitizing text or expanding references/attachments. Omit on
 * generated, hidden, delegated, queued, retry, rewind and resume submissions.
 * The authenticated client is trusted like the approval-response client; this
 * is not physical-human attestation or protection from compromised clients. */
export interface ComposerInputProvenance {
  kind: 'desktop_composer'
  raw_text: string
}
