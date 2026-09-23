# Terraform

## Contents
1. Execution  2. Change workflow  3. Plan classification  4. The user_data trap
5. Persistent state  6. Version pinning

## 1. Execution
Terraform runs in Docker (not installed in WSL). Use the wrapper so every run is identical:

```bash
scripts/tf.sh fmt -check -recursive
scripts/tf.sh validate
scripts/tf.sh plan            # writes tfplan + tfplan.json, runs plan_guard.py
scripts/tf.sh apply           # applies ONLY the saved tfplan that passed the guard
```

The wrapper is equivalent to:
```bash
docker run --rm -e AWS_PROFILE="${AWS_PROFILE:-default}" \
  -v "$HOME/.aws":/root/.aws:ro -v "$PWD":/workspace -w /workspace \
  hashicorp/terraform:1.16.3 <command>
```
Credentials are mounted read-only and never copied into the workspace.

## 2. Change workflow
fmt → validate → `plan -out=tfplan` → `show -json tfplan > tfplan.json` →
`plan_guard.py` → `nova-safety-reviewer` → human approval if REVIEW/BLOCKED → `apply tfplan`
→ post-apply read-only verification → journal entry with plan SHA-256.

Why saved plans: `terraform apply` without a plan file re-plans against live state. If
anything drifted between review and apply, you apply a diff nobody reviewed.

## 3. Plan classification
`resource_changes[].change.actions` in the JSON plan:

| actions | meaning | default verdict |
|---|---|---|
| `["no-op"]`, `["read"]` | nothing | SAFE |
| `["create"]` | add | SAFE (REVIEW if IAM/network) |
| `["update"]` | in-place change | SAFE (REVIEW if IAM/network/protected) |
| `["delete"]` | destroy | BLOCKED unless allow-listed |
| `["delete","create"]` / `["create","delete"]` | **replace** | BLOCKED unless allow-listed |

The human-readable summary "Plan: 1 to add, 1 to change, 1 to destroy" hides replacements
inside add+destroy. Always read the JSON (or `-/+` / `+/-` markers), never just the summary.

Protected types (blocked on delete/replace): `aws_instance`, `aws_ebs_volume`,
`aws_volume_attachment`, `aws_kms_key`, `aws_kms_alias`, `aws_s3_bucket*`, `aws_ecr_repository`,
`aws_cloudwatch_log_group`, `aws_iam_role` (breaks running instance profile).

When a replacement is unexpected, find the attribute forcing it: in `tfplan.json`, look at
`change.replace_paths` for that resource. Common culprits: AMI data source resolving to a new
image, subnet/AZ change, `user_data` without `ignore_changes`, KMS key ARN changes on EBS.
Fix the cause (pin the AMI ID in config, etc.); don't suppress the symptom.

## 4. The user_data trap
The runtime instance has:
```hcl
lifecycle { ignore_changes = [user_data] }
```
so that editing bootstrap does not destroy a running customer runtime.

Consequences you must keep in mind:
- `user_data` runs via cloud-init **on first boot only**. Changing it on an existing instance
  does nothing to the running system, even if Terraform records an update.
- Therefore do **not** "fix" drift by setting `user_data_replace_on_change = false` or dropping
  `ignore_changes`: Terraform state would then claim the new bootstrap is applied when it never ran.
- `ignore_changes` also hides bootstrap drift. Make it visible: tag the instance with
  `nova:bootstrap-sha256 = sha256(rendered user_data)` via a separate tag not covered by
  `ignore_changes`, and have `nova status` compare it with the current template's hash.
- Runtime software changes go through images + bundles + SSM, not user_data.
- A genuine re-bootstrap is an **explicit replacement** (`-replace=aws_instance.runtime`) with a
  written procedure: snapshot EBS state volume → verify snapshot → plan with `-replace` →
  confirm the volume *attachment* is replaced but the *volume* is not → apply → verify.

## 5. Persistent state
The EBS state volume has `prevent_destroy = true`. Never remove it to make apply succeed.
If an operation needs the volume replaced, stop and report: what is replaced, why, whether data
survives, whether a snapshot exists, whether restore was tested. Proceed only with explicit
human authorization naming that resource.

## 6. Version pinning
Terraform 1.16.3 image, AWS provider 5.60.0, locked in `.terraform.lock.hcl`. Don't upgrade either
during a debugging session — a provider upgrade can change defaults and produce replacements
that look like your bug. Upgrades are their own change with their own plan review.
