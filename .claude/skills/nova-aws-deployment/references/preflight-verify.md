# Preflight & Verify

## nova preflight — stop before apply if any required check fails
Every check has: id, what it calls (read-only), pass condition, failure category, remediation owner.

| id | check | how (read-only) |
|---|---|---|
| PF01 | credentials valid | `aws sts get-caller-identity` |
| PF02 | account matches config | compare Account to config |
| PF03 | region enabled & matches | `aws ec2 describe-availability-zones --region <r>` |
| PF04 | VPC exists | `aws ec2 describe-vpcs --vpc-ids` |
| PF05 | subnet private & in VPC, route to NAT/endpoints | `describe-subnets`, `describe-route-tables` |
| PF06 | IAM capability (can create/pass roles) | `aws iam simulate-principal-policy` |
| PF07 | EC2 instance type offered in AZ | `describe-instance-type-offerings` |
| PF08 | ECR repos + image digests exist | `ecr describe-images --image-ids imageDigest=` |
| PF09 | S3 bucket reachable, private, encrypted | `get-public-access-block`, `get-bucket-encryption` |
| PF10 | KMS key usable | `kms describe-key` (Enabled, not PendingDeletion) |
| PF11 | SSM available (endpoints or NAT) | `ssm describe-instance-information` (post-boot) |
| PF12 | Bedrock target exists | `get-inference-profile` / `get-foundation-model` |
| PF13 | Bedrock destination regions discovered | profile `models[]` |
| PF14 | **Bedrock invocation authorized** | minimal Converse (`bedrock_probe.sh`) — catches incident class D |
| PF15 | bundle valid + sha matches | local validate + sha256 |
| PF16 | config schema valid | schema validation |
| PF17 | no test-env literals in config | grep against test-environment.md values |

Output: table of PASS/FAIL/WARN + overall verdict. Any required FAIL → `FAILED_PREFLIGHT`, no plan.

## nova verify — progressive; stop at first failure and report the level
| L | check |
|---|---|
| 1 | AWS resources exist (from Terraform outputs) |
| 2 | EC2 running, status checks OK |
| 3 | SSM reachable (`describe-instance-information` PingStatus Online) |
| 4 | containers running + healthy, image digests match config |
| 5 | control-plane health endpoint OK (called via SSM, not publicly) |
| 6 | worker alive (heartbeat in logs within N minutes) |
| 7 | profiles materialized == bundle profiles |
| 8 | Bedrock invocation succeeds **from runtime role** |
| 9 | a real NOVA verification task runs to completion |
| 10 | task result + logs visible in CloudWatch with deployment version and env id |

READY only when all required levels pass and rollback info is recorded.
