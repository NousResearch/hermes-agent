# Phase 7 — The AWS deployment

Turning `ARCHITECTURE_BOUNDARIES.md` §6 from a design into a thing that runs. **Core patches:
still 1.**

This closes the last blocking finding in
[`PRODUCTION_READINESS_AUDIT.md`](PRODUCTION_READINESS_AUDIT.md) (finding 5). That finding's
sentence was the whole problem: *"the IAM design is sound and unimplemented."* A promise whose
violation is silent is not a promise — it is a trap with good intentions, and the first rushed
deployment falls into it.

---

## The one rule

> **Agents never receive broad access to the customer's AWS account.**

Everything below is machinery for making that rule hold against the pressure that actually
breaks it: not malice, but somebody resolving an `AccessDenied` at five o'clock on a Friday by
widening the thing that denied them.

---

## The shape

Two pieces, deliberately separate.

| | Where | What it is |
|---|---|---|
| The module | `deploy/aws/` | A Terraform root module. Static, reviewable, hand-editable. |
| The derivation | `nova/deploy/` | Renders the per-integration IAM **from the tenant bundle**. |

The derivation is the interesting half. An integration's grants could have been maintained
beside the declaration, in a tfvars file somebody edits — and it would have drifted, because
that kind of pairing always drifts, and only ever in one direction. So the grants live in
`deployment.yaml`, go through the same parser and the same refusals as everything else in a
bundle, and land in the bundle digest. Widening a permission moves the provenance; a
permission that could change without the digest moving would be a permission nobody could
prove the age of.

## What NOVA does and does not do

NOVA writes a file. That is all.

```
nova deploy show   ./bundles/acme    # what the agents would be able to reach
nova deploy render ./bundles/acme    # writes deploy/aws/nova.auto.tfvars.json
```

Running Terraform is the customer's DevOps team with the customer's credentials. This is not
a limitation waiting to be automated away: **an installer holding credentials that can create
IAM roles in a customer account is exactly what their security review exists to prevent.** We
would rather not be able to. It also means there is no answer NOVA can give to "what are your
AWS access requirements" other than *none*, which is a good answer to be able to give.

## The permission model, as code

The runtime role can do seven things, and the list is meant to be read aloud in a review:

1. Invoke the Bedrock models named in `bedrock_model_ids` — enumerated, never wildcarded.
2. Pull its own image, from one ECR repository.
3. Write to its own log group.
4. Read secrets under its own prefix.
5. Use its own KMS key.
6. Assume the declared integration roles — **named one by one**, not by pattern.
7. Talk to Session Manager.

Not one of those touches customer data. Every customer-data grant lives on an integration
role, whose trust policy names the runtime role and requires an external id, so a role that
leaks by name is still useless to anyone who is not this deployment. A permissions boundary
sits over all of them and denies `iam:*` outright: a role that cannot change IAM cannot grant
itself anything, whatever gets attached to it in two years by someone who never read this.

## The refusals, and why there are two copies

Refused, at parse time and again at plan time:

| Shape | Why it is refused |
|---|---|
| `s3:*` | The action list is how a security team answers *"what can the agents do?"* A wildcard makes the question unanswerable. |
| `"*"` as a resource | Grants the action on every resource in the account, including every other tenant's. |
| `iam:`, `sts:`, `organizations:`, `account:`, `kms:` | Grants the ability to grant. An integration holding one is not scoped; it is a route to something unscoped, and the scoping around it stops meaning anything. |
| `s3:ListAllMyBuckets` and friends | Enumerates the account rather than reaching one system. |
| An ARN that wildcards its service | `"*"` written longer. |

Every refusal names the narrower thing to write instead. That is not politeness: a refusal
that leaves an operator with nowhere to go gets worked around, and a worked-around control is
worse than no control, because it still reports success.

The same three rules live in `nova/deploy/aws.py` **and** in the module's
`variable "integrations"` validation blocks. Duplication, on purpose — the tfvars file can be
hand-written, and the person who hand-writes it is the person in a hurry. A test
(`test_the_module_restates_every_refusal_the_generator_makes`) is what keeps the two copies
from parting company.

## What is proven, and what is not

Proven offline, against the real AWS provider (5.70.0):

- `terraform validate` passes; `terraform fmt -check` is clean.
- Every refusal fires at plan time, demonstrated with deliberately over-broad input.
- Applied against a mock AWS API, the rendered IAM is what this document claims: the runtime
  role's inline policy contains **no customer-data permission of any kind**; its only
  `sts:AssumeRole` names the one declared integration role; the integration trust policy
  carries the runtime role ARN and the external-id condition; the boundary denies `iam:*`.
- 38 new platform tests, 577 passing in total.

**Not proven: none of this has been applied to a real AWS account.** `aws_instance` and the
Session Manager policy attachment went unexercised — the mock has neither a real AMI nor the
AWS-managed policy catalogue — and `user_data.sh.tftpl` has never run on a booting host. The
first real deployment is a first real deployment, and `deploy/aws/README.md` says so in the
place an operator will actually be standing when it matters.

## Deliberate constraints

**Single node.** SQLite on an attached volume. Its guarantees hold across processes on one
host and not across hosts, and SQLite on EFS or NFS risks corruption. §6 called this a
documented constraint rather than an oversight, and implementing it has not changed that.

**Bring your own network, and your own image.** No VPC is created; a customer with a landing
zone should not be handed a second one. No image is built; that is a separate and deliberate
step, and the module grants only the permission to pull one.

**No inbound port.** Reached through SSM Session Manager. No SSH key to rotate, no bastion to
maintain, and an operator getting onto the box is an auditable API call rather than a key
somebody still has.

**`prevent_destroy` on the state volume.** It holds the audit log. Getting a
`terraform destroy` past it should require someone to decide to.
