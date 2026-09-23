# Current test environment (NOT reusable configuration)

These values describe the NOVA **test** environment only. They exist so you can run
diagnostics against it. Never copy them into Terraform modules, the CLI, the bundle
format, tests' expected values, or any code meant for customer deployments — those
must take account, region, network and key identifiers from deployment config or
discovery. If you find one of these literals in reusable code, flag it as a bug.

| Item | Value |
|---|---|
| AWS account | 369607682697 |
| Region | eu-west-2 |
| VPC | vpc-055687d6fd6ed9bbc |
| Runtime subnet (private) | subnet-03e1e382992c7ee9d |
| Runtime EC2 | i-0c5779ca02bb22298 |
| Runtime role / instance profile | nova-test-runtime |
| State EBS volume (prevent_destroy) | vol-0236ef097d22014fa |
| KMS key | arn:aws:kms:eu-west-2:369607682697:key/305871aa-31ca-4d53-8cf9-bef0e5fe252e |
| CloudWatch log group | /nova/test |
| S3 bundle bucket | nova-test-bundle-369607682697-eu-west-2 |
| ECR repositories | nova-control-plane, nova-runtime |
| Control-plane image digest | sha256:208db6a41b47710e96ed09cf911ec0148a9d50498bfc53771401f42a853a0aa0 |
| Terraform (via Docker) | hashicorp/terraform:1.16.3 |
| AWS provider | 5.60.0 |
| Bedrock profile | eu.anthropic.claude-sonnet-4-6 (system-defined, EU cross-region) |

Values drift. Before relying on one, confirm it with a read-only call (e.g.
`aws ec2 describe-instances --instance-ids ...`) and update this table and the
journal if it changed.

## Open incident (see bedrock.md → "Incident: Operation not allowed")
Bedrock Converse returns `ValidationException: Operation not allowed` from the runtime
role, from local credentials, and the console says the account is not authorized.
Classified as **D — account/model authorization**. Owner: AWS support / account admin.
Do not revert the Terraform IAM fix because of it.
