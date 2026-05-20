# infra/terraform/dynamodb-skill-locks/main.tf
#
# Phase C: DynamoDB table for distributed skill-write locks (SaaS mode).
#
# APPLY GATE: This module must be reviewed and applied by Blake before
# the distributed lock path in tools/skill_locks.py can acquire real locks.
# Until applied, the lock module degrades gracefully: team writes proceed
# without the distributed lock, which is safe in single-worker dev setups.
#
# What this creates:
#   1. DynamoDB table: hermes-skill-locks
#      - Partition key: skill_key (STRING) — canonical S3 key prefix for the skill
#      - TTL attribute: ttl (NUMBER, Unix epoch seconds)
#      - On-demand (PAY_PER_REQUEST) billing — ~$0/mo at Hermes scale
#   2. IAM policy document granting PutItem/DeleteItem/GetItem on the table
#      (attach to the Fargate task role alongside the S3 skills policy)
#
# What this does NOT create:
#   - The Fargate task IAM role (managed by the ECS/Fargate Terraform in agentic-hub)
#
# Cost estimate:
#   Each skill lock is one conditional PutItem + one DeleteItem.
#   At 1,000 team-skill edits/month → < $0.01/mo.

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region for all resources."
}

variable "table_name" {
  type        = string
  default     = "hermes-skill-locks"
  description = "DynamoDB table name for skill write locks."
}

variable "fargate_task_role_arn" {
  type        = string
  default     = ""
  description = "ARN of the ECS Fargate task IAM role that needs DynamoDB access. Empty = no attachment."
}

variable "environment" {
  type        = string
  default     = "prod"
  description = "Deployment environment tag (prod, staging, dev)."
}

# ---------------------------------------------------------------------------
# DynamoDB lock table
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "skill_locks" {
  name         = var.table_name
  billing_mode = "PAY_PER_REQUEST"   # On-demand — no capacity planning needed.
  hash_key     = "skill_key"

  attribute {
    name = "skill_key"
    type = "S"
  }

  # TTL: locks auto-expire when the worker crashes (30s default in code).
  # DynamoDB deletes expired items within 48h, but for lock purposes the
  # conditional write expression checks the raw `ttl` number, not DynamoDB's
  # TTL deletion — so crashed-worker locks expire the moment their ttl < now.
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = false   # Locks are ephemeral; PITR adds $0.20/GB/mo for no value.
  }

  tags = {
    Name        = "Hermes Skill Write Locks"
    Environment = var.environment
    ManagedBy   = "terraform"
    Plan        = "hermes-001-C"
  }
}

# ---------------------------------------------------------------------------
# IAM policy for Fargate task role
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "skill_locks_rw" {
  statement {
    sid    = "HermesSkillLocksReadWrite"
    effect = "Allow"

    actions = [
      "dynamodb:PutItem",
      "dynamodb:DeleteItem",
      "dynamodb:GetItem",
      # ConditionExpression checks in PutItem require no extra permissions,
      # but explicit GetItem lets the lock code read lock state for diagnostics.
    ]

    resources = [aws_dynamodb_table.skill_locks.arn]
  }
}

resource "aws_iam_policy" "skill_locks_rw" {
  name        = "HermesSkillLocksReadWrite"
  description = "Allows Hermes Fargate task to acquire/release distributed skill write locks."
  policy      = data.aws_iam_policy_document.skill_locks_rw.json

  tags = {
    ManagedBy = "terraform"
    Plan      = "hermes-001-C"
  }
}

# Conditionally attach to Fargate task role if the ARN is provided.
# Phase C does not require the attachment at apply time — it can be deferred
# to the Fargate task module in agentic-hub.
resource "aws_iam_role_policy_attachment" "skill_locks_fargate" {
  count      = var.fargate_task_role_arn != "" ? 1 : 0
  role       = element(split("/", var.fargate_task_role_arn), length(split("/", var.fargate_task_role_arn)) - 1)
  policy_arn = aws_iam_policy.skill_locks_rw.arn
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "table_name" {
  description = "DynamoDB table name — set as HERMES_SKILL_LOCKS_TABLE env var to override."
  value       = aws_dynamodb_table.skill_locks.name
}

output "table_arn" {
  description = "DynamoDB table ARN."
  value       = aws_dynamodb_table.skill_locks.arn
}

output "iam_policy_arn" {
  description = "IAM policy ARN to attach to the Fargate task role."
  value       = aws_iam_policy.skill_locks_rw.arn
}
