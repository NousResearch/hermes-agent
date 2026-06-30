# Standing Orders

## Proactive Operations Review
- scope: personal assistant operations, Obsidian task review, Hermes health checks, low-risk local drafts
- trigger: schedule: every 60 minutes; event: gateway startup; condition: user asks for follow-up
- allowed_actions: read_obsidian, status_check, write_audit_log, create_local_report, delegate_low_risk
- approval_gates: external_message, production_change, deploy, delete_data, money_movement, secret_change
- escalation_rules: stop and ask KJ before destructive changes, financial actions, production deploys, secret/API-key changes, or external publication
- output_policy: return [SILENT] when no anomaly or due follow-up exists; notify KJ for failures, risks, decisions, blocked commitments, waiting-for-KJ input, or throttled active progress reports

## Commitment Follow-up
- scope: reminders and condition tracking inferred from KJ messages
- trigger: event: user_message; condition: message contains reminder, follow-up, next-week, or if-then notification intent
- allowed_actions: write_audit_log, read_obsidian, status_check
- approval_gates: external_message, production_change, money_movement
- escalation_rules: ask KJ when the inferred commitment is ambiguous or would require a high-risk action
- output_policy: quietly create records; notify when due, blocked, or waiting for KJ to provide missing information

## Missing Information Follow-up
- scope: tasks and commitments blocked on KJ-provided facts, files, photos, credentials, decisions, or approval
- trigger: condition: commitment or delegated task status is waiting_for_kj, blocked, stuck, failed, timeout, awaiting_kj_input, or needs_kj_input
- allowed_actions: read_obsidian, status_check, write_audit_log, write_proactive_state
- approval_gates: external_message, production_change, deploy, money_movement, secret_change
- escalation_rules: ask KJ directly when missing information blocks the next safe action; do not continue by guessing
- output_policy: send a concise throttled reminder asking whether the missing information is ready; remain silent until the reminder interval elapses
