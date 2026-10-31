import type { DesktopCommandNotices } from './types_desktop_commands'

// Desktop-side /rollback /agents /config /history /save /export /import /timestamps lines.
export const enDesktopCommands: DesktopCommandNotices = {
  error: message => `error: ${message}`,
  rollbackUsage: 'usage: /rollback [list] | diff <n|hash> | <n|hash> [file] [--all]',
  checkpointsDisabled: 'Checkpoints are not enabled for this session (set checkpoints.enabled in config.yaml).',
  checkpointsEmpty: 'No checkpoints yet for this working directory.',
  checkpointsHeader: 'Checkpoints (newest first):',
  checkpointsHint:
    'Restore with /rollback <n> [file] (add --all to also overwrite your own edits); preview with /rollback diff <n>.',
  noDiff: 'No changes since that checkpoint.',
  rollbackFailed: error => `Rollback failed: ${error}`,
  unknownError: 'unknown error',
  restored: hash => (hash ? `Restored checkpoint ${hash}.` : 'Restored checkpoint.'),
  rewound: count => `Rewound ${count} message(s) from this conversation.`,
  reopenForRewind: 'Reopen this chat to see the rewound conversation.',
  agentsEmpty: 'No active subagents or background processes in this chat.',
  subagentsHeader: count => `Subagents (${count}):`,
  subagentFallback: 'subagent',
  statusRunning: 'running',
  toolCalls: count => `${count} tool calls`,
  processesHeader: count => `Background processes (${count}):`,
  configHeader: 'Configuration (display, approvals, paste and voice settings; everything else is not shown):',
  configDefaults: '(defaults)',
  configFooter: 'Change settings in Settings, or edit config.yaml with `hermes config edit` on the gateway host.',
  historyHeader: count => `Conversation history (${count} messages):`,
  historyEmpty: 'No messages yet.',
  historyYou: 'You',
  historyHermes: 'Hermes',
  saveUsage: 'usage: /save [json|md|html] [filename] [redact]',
  saveLegacyRefused:
    'Not saved: this gateway cannot export a redacted, named or non-JSON copy. Update the gateway, or run /save with no options to write raw JSON on the gateway host.',
  savedOnHost: path => `Saved on the gateway host: ${path}`,
  unknownPath: '(unknown path)',
  saveUnsupported: 'error: this desktop build cannot save files',
  saveDialogTitle: 'Save conversation',
  savedTo: path => `Saved conversation to ${path}`,
  saveCancelled: 'Not saved (cancelled).',
  exportRemote:
    '/export packages a profile from the gateway host, which is not this machine. Run `hermes profile export` on that host.',
  importRemote:
    '/import installs a profile on the gateway host, which is not this machine. Run `hermes profile import <archive>` on that host.',
  timestampsUsage: 'usage: /timestamps [on|off|status]',
  timestampsOn: 'Timestamps: on (this window; set display.timestamps in config.yaml to keep it)',
  timestampsOff: 'Timestamps: off (this window; set display.timestamps in config.yaml to keep it)'
}
