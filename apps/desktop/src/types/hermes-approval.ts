// `ApprovalRequestParams` (tui_gateway/contracts/server_requests.py): the fields that make an approval a
// sandbox's folder (`access`) or site (`network`) card. Absent, or `kind: 'command'`, is a command approval.
export interface ApprovalKindFields {
  kind?: 'command' | 'access' | 'network'
  path?: string
  access?: 'read' | 'read_write'
  host?: string
  label?: string
  risk?: string
}
