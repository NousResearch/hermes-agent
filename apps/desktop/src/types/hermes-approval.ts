// `ApprovalRequestParams` (tui_gateway/contracts/server_requests.py): the fields that make an approval a
// sandbox's folder (`access`) or site (`network`) card, or a new scheduled job (`schedule`). Absent, or
// `kind: 'command'`, is a command approval.
export interface ApprovalKindFields {
  kind?: 'command' | 'access' | 'network' | 'schedule' | 'schedule_edit'
  path?: string
  access?: 'read' | 'read_write'
  host?: string
  label?: string
  risk?: string
}
