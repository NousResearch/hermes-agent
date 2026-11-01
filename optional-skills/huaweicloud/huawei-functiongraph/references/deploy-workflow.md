# End-to-End Deployment Workflow

```
1. Write code     → Create index.py with handler function
2. Package        → zip -j function.zip index.py (no directory nesting!)
3. Create function → hcloud FunctionGraph CreateFunction (inline for small demos, zip for production)
4. Verify         → hcloud FunctionGraph InvokeFunction
5. Create trigger  → hcloud FunctionGraph CreateFunctionTrigger (TIMER for testing, DEDICATEDGATEWAY for HTTP)
```

## Quick Demo (Inline — No File Packaging)

Use inline for small demo functions (<10KB):

```bash
# 1. Prepare base64-encoded code
python3 -c "
import base64
code = '''import json
def handler(event, context):
    return {'statusCode': 200, 'body': json.dumps({'message': 'Hello FunctionGraph!'}), 'headers': {'Content-Type': 'application/json'}}
'''
print(base64.b64encode(code.encode()).decode())
"

# 2. Create function with inline code (paste base64 output as --func_code.file)
hcloud FunctionGraph CreateFunction --help
hcloud FunctionGraph CreateFunction \
  --func_name=my-backend \
  --runtime=Python3.10 \
  --handler=index.handler \
  --memory_size=256 \
  --package=default \
  --timeout=30 \
  --code_type=inline \
  --func_code.file=<base64-from-step-1> \
  --cli-region=<region> \
  --project_id=<project-id>

# 3. Verify
hcloud FunctionGraph InvokeFunction \
  --function_urn=<urn> \
  --name=test-event \
  --x_cff_request_version=v0
```

## Production Deployment (Zip)

```bash
# 1. Write code
cat > index.py << 'EOF'
import json

def handler(event, context):
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Hello from FunctionGraph!"}),
        "headers": {"Content-Type": "application/json"}
    }
EOF

# 2. Package — CRITICAL: use -j to flatten, no directory nesting
zip -j function.zip index.py

# 3. Create function (discover params with --help first!)
hcloud FunctionGraph CreateFunction --help
# Required: --func_name, --runtime, --handler, --memory_size, --package, --timeout
# Code type: use --code_type=zip --code_filename=function.zip
# IMPORTANT: cd to function.zip directory first! --code_filename is filename only.

# 4. Verify (store URN from step 3 output)
hcloud FunctionGraph InvokeFunction --help
# Requires body param: --name=test-event (becomes event body passed to handler)
hcloud FunctionGraph InvokeFunction \
  --function_urn=<urn> \
  --name=test-event \
  --x_cff_request_version=v0

# 5. Create trigger (see references/triggers.md)
# For quick testing: TIMER trigger (no prerequisites)
# For HTTP access: DEDICATEDGATEWAY (requires APIG instance — see pre-flight checklist below)

# 5a. TIMER (simple — no APIG dependency)
hcloud FunctionGraph CreateFunctionTrigger \
  --function_urn=<urn> \
  --trigger_type_code=TIMER \
  --event_type_code=MessageCreated \
  --trigger_status=ACTIVE \
  --event_data.name=test-timer \
  --event_data.schedule_type=Rate \
  --event_data.schedule="5m"
```

## DEDICATEDGATEWAY Pre-Flight Checklist

Before creating a DEDICATEDGATEWAY (HTTP) trigger, verify these prerequisites exist:

```bash
# 1. Check for APIG dedicated instance
hcloud APIG ListInstancesV2 --cli-region=<r>

# 2. If no instance exists, create one (see huawei-apig skill)
#    Key: --spec_id=PROFESSIONAL --loadbalancer_provider=elb for public access
#    Requires: VPC, subnet, security group, enterprise_project_id

# 3. Once instance is Running with public IP, create API group
hcloud APIG CreateApiGroupV2 --instance_id=<id> --name=<group-name>

# 4. Create DEDICATEDGATEWAY trigger (see references/triggers.md)

# 5. Publish the API (after trigger creation)
hcloud APIG BatchPublishOrOfflineApiV2 \
  --instance_id=<apig-instance-id> \
  --action=online \
  --env_id=<env-id> \
  --api_ids=<api-id-from-trigger-response>
```

If no APIG instance is available, use TIMER trigger for function testing instead.
