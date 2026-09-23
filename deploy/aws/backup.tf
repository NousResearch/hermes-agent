# ---------------------------------------------------------------------------
# Backups: scheduled snapshots of the state volume
# ---------------------------------------------------------------------------
#
# The state volume holds everything a tenant would lose in an incident — the runtime's work
# board, every agent profile, conversation history, the audit log. Until this file it had no
# backup at all: prevent_destroy stops Terraform deleting it, and nothing protected it from
# a failed disk, a lost availability zone or a bad write.
#
# Data Lifecycle Manager snapshots it on a schedule and expires old snapshots itself, so the
# backup keeps happening with no host, cron job or credential of NOVA's involved.
#
# The policy targets the volume by the Name tag the volume already carries. Adding a tag to
# select it would be an update to a prevent_destroy resource for no gain; reading the tag it
# has touches nothing.
#
# Recovery is a new volume created from a snapshot, in any availability zone of the region,
# attached to a runtime instance there — see deploy/aws/README.md, "Recovering from a
# snapshot". A snapshot nobody has restored is a hope, not a backup: the drill in that section
# is part of the deployment, not an optional extra.

resource "aws_iam_role" "snapshots" {
  count = var.backup_enabled ? 1 : 0

  name = "${local.name_prefix}-snapshots"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "dlm.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "snapshots" {
  count = var.backup_enabled ? 1 : 0

  role       = aws_iam_role.snapshots[0].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole"
}

# The volume is encrypted with the tenant's key; a snapshot of it is too, and DLM needs to
# use that key to create and expire them. Scoped to the one key, and CreateGrant only for AWS
# resources — the same condition EBS itself relies on.
resource "aws_iam_role_policy" "snapshots_kms" {
  count = var.backup_enabled ? 1 : 0

  name = "state-key"
  role = aws_iam_role.snapshots[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "UseStateKey"
        Effect   = "Allow"
        Action   = ["kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:DescribeKey"]
        Resource = local.kms_key_arn
      },
      {
        Sid       = "GrantForEbsOnly"
        Effect    = "Allow"
        Action    = "kms:CreateGrant"
        Resource  = local.kms_key_arn
        Condition = { Bool = { "kms:GrantIsForAWSResource" = "true" } }
      },
    ]
  })
}

resource "aws_dlm_lifecycle_policy" "state" {
  count = var.backup_enabled ? 1 : 0

  description        = "${local.name_prefix} state volume"
  execution_role_arn = aws_iam_role.snapshots[0].arn
  state              = "ENABLED"

  policy_details {
    resource_types = ["VOLUME"]
    target_tags    = { Name = "${local.name_prefix}-state" }

    schedule {
      name = "every-${var.snapshot_interval_hours}h"

      create_rule {
        interval      = var.snapshot_interval_hours
        interval_unit = "HOURS"
      }

      retain_rule {
        count = var.snapshot_retain_count
      }

      copy_tags = true
      tags_to_add = {
        "nova:snapshot" = "scheduled"
        "nova:tenant"   = var.tenant_id
      }
    }
  }
}
