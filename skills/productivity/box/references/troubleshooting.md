# Troubleshooting

Capture the actor, object ID and type, exact command, status code, and safe error body before changing approach.

## First checks

```bash
box users:get me --json --fields id,name,login
box configure:environments:list
box files:get <FILE_ID> --json --fields id,name,parent
box folders:get <FOLDER_ID> --json --fields id,name,parent
box hubs --scope all --max-items 1000 --json
box hubs:get <HUB_ID> --json
```

Confirm the current actor, resource type, ID, resource-specific collaboration, app scopes, and selected environment. Do not use folder `0` as an access test: it cannot discover a Hub and may not list every shared file or folder.

## Common failures

| Signal | Likely cause | Next action |
| --- | --- | --- |
| 401 or 403 | expired auth, missing scope, insufficient role | verify identity, reauthorize the app, and check folder role |
| shared file/folder absent from root or 404 | wrong actor, an access-only/shared item, or missing file/folder collaboration | verify `users:get me`, then fetch the known file/folder ID directly; only change collaboration after confirming the target and actor |
| Hub absent from root or 404 | root listing cannot discover Hubs, wrong actor, or missing Hub collaboration | run `box hubs --scope all` and `box hubs:get <HUB_ID>`; verify Hub collaboration separately from underlying-file access |
| 409 | duplicate name, existing collaboration, metadata conflict | list the parent/template and reuse or rename deliberately |
| 429 | rate limit | honor `Retry-After`, retry the same request, and reduce batch rate |
| Box AI access error | feature disabled, plan/unit restriction, unsupported content | explain the limitation and offer metadata/search, a sample, units, or approved fallback |

Do not diagnose missing content until identity and access are verified. Do not silently change actors, broaden sharing, or download confidential source files as a workaround.
