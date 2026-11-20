export interface ApprovalCopy {
  gatewayDisconnected: string
  sendFailed: string
  reconnect: string
  timedOutSystemLine: string
  openSafetySettings: string
  run: string
  command: string
  commandDetails: string
  moreOptions: string
  allowSession: string
  alwaysAllowMenu: string
  jumpToApproval: string
  reject: string
  alwaysTitle: string
  alwaysDescription: (pattern: string) => string
  alwaysAllow: string
  deny: string
  allow: string
  allowHint: string
  denyReason: string
  denyReasonPlaceholder: string
  access: {
    title: string
    readOnly: string
    readWrite: string
    alwaysTitle: string
    alwaysDescription: string
  }
  network: {
    title: string
    alwaysTitle: string
    alwaysDescription: string
  }
  schedule: {
    title: string
    allowSession: string
    alwaysAllowMenu: string
    alwaysTitle: string
    alwaysDescription: string
    timedOutSystemLine: string
  }
}
