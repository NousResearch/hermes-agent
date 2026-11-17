/** The webhook detail's test block: a copy-paste signed curl and a Send test button. */
export interface WebhookTestCopy {
  title: string
  hint: string
  placeholderHint: string
  copyCommand: string
  send: string
  sending: string
  accepted: (status: number, deliveryId: string) => string
  rejected: (status: number, detail: string) => string
  failed: string
}
