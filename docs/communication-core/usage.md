# CLI and skill usage

Initialize the Core database, then register an explicit account:

```text
hermes communication init
hermes communication accounts add --provider facebook --namespace personal-1 --label F1 --owner-profile default
hermes communication accounts list
hermes communication accounts status <account-id>
hermes communication accounts capabilities <account-id>
```

Read/sync workflows:

```text
hermes communication sync run <account-id> --mode full
hermes communication sync run <account-id> --mode incremental
hermes communication sync retry <account-id>
hermes communication people search <query>
hermes communication people show <person-id>
hermes communication timeline show <person-id> --endpoint-id <endpoint-id> --start-at <iso> --end-at <iso>
hermes communication analyze conversation <conversation-id>
hermes communication brief daily
```

Routing, groups, and safe drafts:

```text
hermes communication routes allow <source-account> <target-account> --reason <reason>
hermes communication routes dry-run <person> <source-endpoint> <target-endpoint>
hermes communication routes set <person> <source-endpoint> <target-endpoint>
hermes communication groups create <name>
hermes communication groups preview <group-id>
hermes communication drafts create <person> <source-endpoint> --text <draft>
hermes communication approvals approve <draft-id> --ttl-minutes 30
hermes communication greetings plan --date YYYY-MM-DD
```

Migration is explicit and reversible:

```text
hermes communication migration facebook-import <facebook-account-id> <legacy-db>
hermes communication migration facebook-rollback <migration-run-id>
```

The `$manage-communications` skill is the shared orchestrator. Its six
one-level references cover account routing, identity journeys, analysis/CRM,
groups/greetings, approval safety, and adapters. The Facebook skill delegates
to it; the dialogue-campaign skill is a retirement shim. Skills must call the
CLI and never SQLite or a legacy sender.
