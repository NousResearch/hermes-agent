# Facebook CRM and legacy skill migration

## Procedure

1. Register one exact Facebook `ConnectedAccount` and keep
   `write_actions_enabled=0`.
2. Hash and open the legacy CRM with SQLite URI `mode=ro`.
3. Run `communication migration facebook-import` against a synthetic copy
   first, then the intended source.
4. Compare source counts with stable mappings/reconciliation. Re-running the
   same account/source hash is idempotent.
5. Use Core CLI/skill reads. Keep legacy paths until rollback evidence is
   accepted.

Stable IDs are derived from source system, connected account, entity type, and
legacy ID. Friends map to Person/Identity/Endpoint/Conversation; messages and
events retain mappings/provenance. Birthdays, campaigns, settings, sync state,
legacy approvals, and legacy outbox rows are preserved inertly in
`legacy_records`. They never become active Core approvals/outbox items.

## Legacy inventory and canonical replacement

| Legacy family / examples | Disposition | Canonical replacement | Rollback |
| --- | --- | --- | --- |
| Facebook local lookup: `skills/social-media/facebook/facebook_api.py`, `facebook_db.py`, `facebook_crm_skill.py` | retained, skill consumer migrated | `people search/show`, timeline, Facebook read adapter | use old read-only command only after ID compare |
| Message/history lookup: `search_messages.py`, `get_*_chat*.py`, `inspect_*.py`, `query*.py`, dump/check scripts | retained for forensics | `people search`, `timeline show`, `analyze conversation` | no data mutation; disable Core adapter |
| Sync/inbox/profile/timeline: `facebook_sync.py`, `facebook_inbox_processor.py`, `facebook_profile_deep_scraper.py`, `facebook_enrich*.py`, `facebook_timeline.py` | browser ownership retained; direct skill calls retired | Facebook application service + adapter, `sync run/status/retry` | disable account; retain source CRM/profile |
| Birthday: `check_facebook_birthdays.py`, `birthday_orchestrator.py`, `send_belated_birthday.py`, resend scripts | sender paths retired | `greetings plan/list`, draft/approval evidence | Core migration rollback; legacy rows remain inert |
| Dialogue/outreach: `dialogue_orchestrator.py`, campaign analyzers, `social_outreach_pipeline.py` | retired | explainable analysis/brief and user-reviewed drafts | legacy campaign records remain in `legacy_records` |
| Send/queue: `facebook_messages_send.py`, `send_*.py`, `queue_*_send.py` | prohibited; production worker off | no production replacement in this goal; fake sink tests only | never re-enable as rollback |
| Diagnostics/export monitors: `monitor_facebook_export.py`, `check_*`, `verify_*` | retained | account health, sync status/issues, migration reconciliation | read-only legacy diagnostic allowed if documented |

The Facebook `SKILL.md` now delegates common work to
`$manage-communications`. `dialogue_campaigns/SKILL.md` is a retirement shim.
No legacy file was deleted.
