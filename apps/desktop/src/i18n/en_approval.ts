import type { ApprovalCopy } from './types_approval'

export const enApproval: ApprovalCopy = {
  gatewayDisconnected:
    'Hermes is offline right now. The command is still waiting for your answer (until the approval timeout). Reconnect, then send it again.',
  sendFailed: 'Could not send your answer',
  reconnect: 'Reconnect',
  timedOutSystemLine:
    'Approval timed out — the command was not run. Ask Hermes to try again, or raise the limit in Settings → Safety → Approval timeout.',
  openSafetySettings: 'Open Safety settings',
  run: 'Run',
  command: 'Command',
  commandDetails: 'Command details',
  moreOptions: 'More approval options',
  allowSession: 'Allow this session',
  alwaysAllowMenu: 'Always allow…',
  jumpToApproval: 'Approval needed',
  reject: 'Reject',
  alwaysTitle: 'Always allow this command?',
  alwaysDescription: pattern =>
    `This adds the “${pattern}” pattern to your permanent allowlist (~/.hermes/config.yaml). Hermes won’t ask again for commands like this — in this session or any future one.`,
  alwaysAllow: 'Always allow',
  deny: 'Deny',
  allow: 'Allow',
  allowHint: 'Until the sandbox is switched off',
  denyReason: 'Reason for denying',
  denyReasonPlaceholder: 'Tell Hermes why (optional)',
  access: {
    title: 'Folder access',
    readOnly: 'Read only',
    readWrite: 'Read and write',
    alwaysTitle: 'Always allow this folder?',
    alwaysDescription: 'The folder is added to this profile’s sandbox. Hermes won’t ask again for it in any chat.'
  },
  network: {
    title: 'Website access',
    alwaysTitle: 'Always allow this site?',
    alwaysDescription: 'The site is added to this profile’s allowed sites. Hermes won’t ask again for it in any chat.'
  },
  schedule: {
    title: 'Schedule this job?',
    allowSession: 'Allow for this chat',
    alwaysAllowMenu: 'Always allow new jobs…',
    alwaysTitle: 'Always allow new scheduled jobs?',
    alwaysDescription:
      'Hermes will create scheduled jobs without asking first. You can turn this back on by removing cron_create from command_allowlist in config.yaml.',
    timedOutSystemLine:
      'Approval timed out — the job was not created. Ask Hermes to try again, or raise the limit in Settings → Safety → Approval timeout.',
    editTitle: 'Change this job?',
    editTimedOutSystemLine:
      'Approval timed out — the job was not changed. Ask Hermes to try again, or raise the limit in Settings → Safety → Approval timeout.'
  }
}
