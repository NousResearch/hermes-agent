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

## Two processes, and what happens with only one

This module deploys **two** units, and the second one is optional because it needs a
second image:

| Unit | Image | Job |
|---|---|---|
| `nova.service` | `image_uri` — the NOVA control plane | Governs, observes, and **submits** work to the board. |
| `nova-worker.service` | `worker_image_uri` — the full Hermes runtime | **Claims and runs** that work. |

They are different images on purpose. The control plane carries the standard library and
PyYAML; it holds no model credential and cannot run an agent, which is most of why its
attack surface is worth having. The dispatcher claims `ready` tasks and shells out to
`hermes -p <assignee> chat -q`, so it needs the whole runtime.

**Deploy with `worker_image_uri` empty and the platform accepts objectives, creates
durable tasks, answers 200 — and nothing ever runs them.** That is not a failure mode you
discover from the health check, because both processes are independently healthy; the
work simply sits in `ready`. NOVA now reports it directly: `GET /platform/v1/tasks`
carries an `execution` block, and submitting onto a board that has been proven stalled
returns a warning saying so. An intentionally control-plane-only deployment is a
legitimate configuration — planning and routing are useful on their own — it just has to
be a choice somebody made rather than one they inherited.

**The worker container runs `sleep infinity`, and that is deliberate.** The runtime image
supervises its own gateway in an s6 slot (`gateway-default`), and the kanban dispatcher is
embedded in that gateway. Passing `gateway run` as the container command starts a *second*
gateway — `main-wrapper.sh` routes it to `hermes gateway run`, and the boot reconciler
separately reads that same argv as a pre-s6 container and starts the slot too. They race
for the PID file, the loser exits, and when the loser is the main program the container
exits with it. `HERMES_GATEWAY_BOOTSTRAP_STATE=running` is what brings the slot up on a
blank volume; without it the reconciler registers it DOWN and waits.

Both units share `/var/lib/nova`, and the worker runs with `HERMES_UID=10001` so it
matches the owner the bootstrap chowns the volume to, and `HERMES_HOME` pointed at the
same home the control plane serves. Its agents reach Bedrock through the instance role —
no credential is placed on the host or in the tenant bundle.

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

## Changing the bootstrap on a deployment that already exists

`user_data.sh.tftpl` runs once, at first boot. cloud-init records a `PER_INSTANCE` semaphore
under `/var/lib/cloud/instance/sem`, so on a host that has already booted the script does not
run again — not on a reboot, and not on the stop/start the AWS provider would perform to write
a new one. **A changed bootstrap reaches a running instance only by replacing that instance.**

So `aws_instance.runtime` ignores `user_data` changes. A plan that corrects an IAM grant,
adds an integration, or bumps a model id touches the IAM policy and leaves the instance alone:

```
aws_iam_role_policy.runtime   will be updated in-place
aws_instance.runtime          no change
aws_volume_attachment.state   no change
```

The cost is that `image_uri` and `worker_image_uri` are interpolated into the systemd units
this script writes, so changing either no longer reaches a running host on its own. Two ways
to roll one forward, and they are not interchangeable:

| | What it does | When |
|---|---|---|
| `terraform apply -replace=aws_instance.runtime` | New host, current bootstrap, every unit rewritten. The state volume is **detached, not destroyed** (`prevent_destroy`, and the replacement only replaces the attachment); `NOVA_APPLY_ON_START` reconciles the bundle on the way up. | The bootstrap itself changed — a new unit, a new mount, a new env var. |
| `nova bundle unpack --replace` over SSM, then restart the unit | Leaves the host alone. | The tenant bundle changed and the units did not. |

`terraform output bootstrap_sha256` is how the difference is seen, since Terraform no longer
reports it. Compare it against the host:

```bash
aws ssm send-command --instance-ids "$(terraform output -raw instance_id)" \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["sha256sum /var/lib/cloud/instance/user-data.txt"]'
```

A mismatch means the running instance predates this module's bootstrap. That is information,
not an emergency — but it is owed a `-replace` before anything that depends on the new script.
Note that the first apply after this lifecycle block was introduced **adopts** whatever hash
the configuration then rendered; an instance older than that will not be flagged by it, so
establish the baseline with the command above once.

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
