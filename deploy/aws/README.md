# NOVA on AWS

One tenant, one deployment, inside **the customer's own AWS account**.

NOVA renders the input; your DevOps team runs Terraform with their own credentials. That
split is deliberate and not a limitation to work around: an installer holding credentials
that can create IAM roles in a customer account is precisely the thing their security review
exists to prevent. **We never require, hold, or ask for root or administrator access.**

---

## What this creates

| | |
|---|---|
| `aws_iam_role.runtime` | What the runtime runs as. **No customer-data permissions.** |
| `aws_iam_policy.runtime_boundary` | A permissions ceiling attached to every role here. |
| `aws_iam_role.integration[*]` | One role per declared integration, assumed by the runtime role. |
| `aws_instance` + `aws_ebs_volume` | Single node, encrypted volume, IMDSv2 required. |
| `aws_security_group` | **No ingress rules.** Egress on 443 only. |
| `aws_cloudwatch_log_group` | Operational logs, KMS-encrypted, 365-day default retention. |
| `aws_kms_key` | Volume, logs and secrets, with rotation enabled. |

Reached through SSM Session Manager: no inbound port, no SSH key, no bastion. An operator
getting onto the box is an auditable API call rather than a key somebody still has.

## The permission model

The runtime role can do exactly seven things, and the list is meant to be read:

1. Invoke the Bedrock models named in `bedrock_model_ids` — enumerated, never wildcarded.
2. Pull its own image, from one ECR repository.
3. Write to its own log group.
4. Read secrets under its own `secret_prefix`.
5. Use its own KMS key.
6. Assume the integration roles declared for this tenant — **named one by one**.
7. Talk to Session Manager.

Everything an agent can reach outside the runtime is therefore in `var.integrations`, and
adding to that list is an IAM change the customer's own security team reviews. If the list is
empty, the agents can reach nothing.

Each integration role's trust policy names the runtime role and requires an external id, so
a role that leaks by name is still useless to anyone who is not this deployment.

### What is refused

`variable "integrations"` rejects, at plan time:

- a wildcard in any action (`s3:*`);
- `"*"` as a resource;
- any `iam`, `sts`, `organizations`, `account` or `kms` action — those grant the ability to
  grant, which turns a scoped integration into an unscoped one.

`nova deploy render` applies the same refusals, plus a few it can make better error messages
for, before it writes anything. **Both, on purpose:** a check that lives only in the
generator is one an operator skips by hand-editing the tfvars, and the operator who does
that is the one under time pressure.

## Using it

```bash
# 1. Derive the integrations from the tenant bundle.
nova deploy show   ./bundles/acme      # what the agents would be able to reach
nova deploy render ./bundles/acme      # writes deploy/aws/nova.auto.tfvars.json

# 2. Fill in the infrastructure the customer owns.
cp terraform.tfvars.example terraform.tfvars && $EDITOR terraform.tfvars

# 3. Their DevOps team, their credentials.
terraform init
terraform plan
terraform apply
```

`nova.auto.tfvars.json` is generated and git-ignored. The bundle is the thing to review and
to version; a committed copy of the rendered file is a second source of truth that will
disagree with the first one exactly when it matters.

### What the operator applying this needs

Permission to create the resources listed above — IAM roles and policies, EC2, EBS, KMS,
CloudWatch Logs — in one account. Not `AdministratorAccess`, and not a permanent one: a
role their pipeline assumes for the duration of the apply is the intended shape.

## Deliberate constraints

**Single node.** State is SQLite on an attached volume. SQLite's guarantees hold across
processes on one host and not across hosts, and SQLite on EFS or NFS risks corruption. This
is a documented limit, not an oversight — see `docs/platform/ARCHITECTURE_BOUNDARIES.md` §6.

**Bring your own network.** No VPC is created. A customer with a landing zone should not be
handed a second one.

**Bring your own image.** This module grants the instance permission to pull the image and
does not build it. Building and pushing is a separate, deliberate step.

**`prevent_destroy` on the state volume.** It holds the audit log. Removing that lifecycle
block to let a `terraform destroy` through is a decision someone should have to make on
purpose.

## What has and has not been verified

Verified, offline, against the real AWS provider:

- `terraform validate` passes, and `terraform fmt -check` is clean.
- Every refusal above fires at plan time — proven with deliberately over-broad inputs.
- Applied against a mock AWS API, the rendered IAM is what is claimed: the runtime role's
  policy contains no customer-data permissions, the integration trust policy names only the
  runtime role plus the external id, and the boundary denies `iam:*`.

**Not verified:** this has never been applied to a real AWS account. `aws_instance` and the
Session Manager policy attachment were not exercised (the mock has neither a real AMI nor the
AWS-managed policy catalogue), and `user_data.sh.tftpl` has never run on a booting host.
Treat the first real deployment as a first real deployment.
