# Router output format — schema v4

`dry-run-route.sh` emits human text plus one JSON receipt per mission. Parse from the first `{` to the last `}`.

The JSON receipt is canonical. `repository_authority` is derived compatibility only. Role fields, `write_intents`, `role_evidence`, `ontology_status`, `explicit_target`, and `conflicts` drive interpretation.

Required decision values: `RESOLVED`, `AMBIGUOUS`, `UNRESOLVED`, `BLOCKED`.

Do not merge stderr into stdout. Do not infer missing fields. Do not convert fail-closed decisions.
