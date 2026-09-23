# Runtime operations: SSM, networking, CloudWatch, workers, bundles, profiles

## SSM
Preferred admin channel; the runtime is private. No public SSH, ever, for convenience.
```bash
aws ssm start-session --target <instance-id> --region <region>          # interactive
aws ssm send-command --instance-ids <id> --document-name AWS-RunShellScript \
  --parameters 'commands=["docker ps --format {{.Names}}:{{.Status}}"]' --region <region>
aws ssm get-command-invocation --command-id <cmd> --instance-id <id> --region <region>
```
Remote commands must be minimal, observable, idempotent/reversible where possible, and labelled
read-only or MUTATING. For long operations, submit once and check status when useful — don't poll
in a tight loop.

## Networking
```
Internet → NAT → private subnet → EC2 runtime → AWS APIs (Bedrock, ECR, CloudWatch, SSM, S3, KMS)
```
VPC endpoints may replace NAT for AWS APIs; if present, endpoint policies are another place
AccessDenied can originate. Future public webhooks need an explicit design: HTTPS/TLS, authn,
authz, rate limiting, logging, secret management. Control-plane ports are never public.

## CloudWatch
Log group per environment (test: `/nova/test`). Every event should let an operator answer: runtime
alive? worker alive? task started/finished/crashed? Bedrock failed (category)? which model? which
deployment version (image digest + bundle sha)? which customer/environment?
Never log keys, tokens, passwords, credentials or unnecessary personal data.

## Hermes workers
```
gateway → kanban dispatcher → worker → agent profile → model provider
```
`spawned=1 crashed=1` from the dispatcher is a symptom. Retrieve the worker's own stderr/log and
the gateway log for the same task id before forming a hypothesis. Common causes, in rough order of
likelihood: profile not materialized, model config invalid, Bedrock failure (classify it), missing
env/config, image mismatch.

## Deployment bundles
local bundle → validate → archive → SHA-256 → upload to private S3 → controlled transfer
(presigned URL or equivalent) → verify checksum on instance → unpack → apply.
The runtime role does **not** get broad S3 read just because bundles live in S3. Bundles contain
configuration only — never secrets.

## Profile materialization & --prune
Profiles must exist before workers run them: `python3 -m nova apply <bundle>` materializes them.
`--prune` deletes anything not in the bundle. Before using it:
1. list current profiles on the runtime
2. list desired profiles in the bundle
3. show the diff (what disappears)
4. get confirmation that removal is intended
Never `--prune` against an unknown or unverified production bundle.
