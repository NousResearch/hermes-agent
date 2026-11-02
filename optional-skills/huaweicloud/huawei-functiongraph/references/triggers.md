# Triggers

**Always run `hcloud FunctionGraph CreateFunctionTrigger --help` first** for exact parameter names and requirements.

## KooCLI event_data Format (CRITICAL)

KooCLI uses **dotted key-value format**, NOT JSON strings:

```bash
# CORRECT
--event_data.name=my-api --event_data.auth=IAM --event_data.path=/test

# WRONG — will fail
--event_data='{"name":"my-api","auth":"IAM","path":"/test"}'
```

## Trigger Types

| Type           | `--trigger_type_code` | Notes                                                      |
| -------------- | --------------------- | ---------------------------------------------------------- |
| APIG Dedicated | `DEDICATEDGATEWAY`    | Use this, not `APIG` (deprecated). Requires APIG instance. |
| Timer / Cron   | `TIMER`               | Simplest trigger for testing. No APIG dependency.          |
| OBS            | `OBS`                 | Event when objects created/deleted in bucket               |
| SMN            | `SMN`                 | Message notification trigger                               |

## TIMER Trigger (Simple Testing)

The TIMER trigger is the easiest path for verifying a function works — no APIG instance needed:

```bash
hcloud FunctionGraph CreateFunctionTrigger \
  --function_urn=<urn> \
  --trigger_type_code=TIMER \
  --event_type_code=MessageCreated \
  --trigger_status=ACTIVE \
  --event_data.name=<trigger-name> \
  --event_data.schedule_type=Rate \
  --event_data.schedule="1m"
```

## DEDICATEDGATEWAY Trigger (HTTP Access)

`trigger_type_code=APIG` is **deprecated**. Use `DEDICATEDGATEWAY` for KooCLI 7.x.

### Required Parameters (Hidden Optional)

These are labeled optional by `--help` but are **required** for DEDICATEDGATEWAY:

| Param                      | Note                                                                         |
| -------------------------- | ---------------------------------------------------------------------------- |
| `--event_data.protocol`    | `HTTPS` or `HTTP` or `BOTH`                                                  |
| `--event_data.sl_domain`   | Subdomain from APIG instance (e.g. `xxxx.apic.<region>.huaweicloudapis.com`) |
| `--event_data.env_name`    | API environment name (`RELEASE`)                                             |
| `--event_data.env_id`      | API environment ID                                                           |
| `--event_data.instance_id` | APIG dedicated instance ID                                                   |
| `--event_data.group_id`    | API group ID                                                                 |
| `--event_data.name`        | API name — **hyphens (`-`) are DISALLOWED**. Use underscores (`_`) instead.  |

### Error Mapping

| Error               | Root Cause → Fix                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| FSS.1417            | Missing `instance_id`, `group_id`, `protocol`, `env_name`, or `env_id`. These are labeled optional but REQUIRED. |
| API name regex fail | Rename API — remove hyphens (`-`), use `[a-zA-Z0-9_]+` only.                                                     |

### Example

```bash
hcloud FunctionGraph CreateFunctionTrigger \
  --function_urn=<urn> \
  --trigger_type_code=DEDICATEDGATEWAY \
  --event_type_code=APICreated \
  --trigger_status=ACTIVE \
  --event_data.name=<api-name-no-hyphens> \
  --event_data.auth=IAM \
  --event_data.path=/my-backend \
  --event_data.match_mode=SWA \
  --event_data.type=1 \
  --event_data.protocol=HTTPS \
  --event_data.req_method=ANY \
  --event_data.func_info.timeout=5000 \
  --event_data.instance_id=<apig-instance-id> \
  --event_data.group_id=<api-group-id> \
  --event_data.sl_domain=<sl-domain> \
  --event_data.env_name=RELEASE \
  --event_data.env_id=<env-id>
```

After trigger creation, you must **publish** the API before it's accessible:

```bash
hcloud APIG BatchPublishOrOfflineApiV2 \
  --instance_id=<apig-instance-id> \
  --action=online \
  --env_id=<env-id> \
  --api_ids=<api-id-from-trigger-response>
```

### Alternative: --cli-jsonInput (Recommended for Complex Configs)

For trigger configs with many nested fields, use `--cli-jsonInput` to avoid shell escaping issues:

```bash
cat > trigger.json << 'EOF'
{
  "trigger_type_code": "DEDICATEDGATEWAY",
  "trigger_status": "ACTIVE",
  "event_type_code": "APICreated",
  "event_data": {
    "name": "my-api",
    "auth": "IAM",
    "path": "/my-backend",
    "match_mode": "SWA",
    "type": 1,
    "protocol": "HTTPS",
    "req_method": "ANY",
    "func_info": { "timeout": 5000 },
    "instance_id": "<apig-instance-id>",
    "group_id": "<api-group-id>",
    "sl_domain": "<sl-domain>",
    "env_name": "RELEASE",
    "env_id": "<env-id>"
  }
}
EOF
hcloud FunctionGraph CreateFunctionTrigger \
  --function_urn=<urn> \
  --cli-region=<region> \
  --project_id=<project_id> \
  --cli-jsonInput=trigger.json
```

## List / Delete Triggers

```bash
hcloud FunctionGraph ListFunctionTriggers --function_urn=<urn>
hcloud FunctionGraph DeleteFunctionTrigger --function_urn=<urn> --trigger_type_code=<type> --trigger_id=<id>
```

> Deleting a trigger does NOT cascade-delete the associated APIG API. After `DeleteFunctionTrigger`, also run `hcloud APIG ListApisV2 --instance_id=<id>` and delete orphaned APIs to avoid resource residue.
>
> **APIG event format**: When using DEDICATEDGATEWAY, the event body is Base64 encoded and uses a non-standard structure. See `apig-event-format.md` for handler templates.
