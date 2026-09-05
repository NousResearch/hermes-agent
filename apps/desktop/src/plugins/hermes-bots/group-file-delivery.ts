/** A required byte delivery failed before the member saw the input. */
export class GroupFileDeliveryError extends Error {
  constructor(message = 'Required file delivery failed. Retry this member after the source is available.') {
    super(message)
    this.name = 'GroupFileDeliveryError'
  }
}
