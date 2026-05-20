# infra/terraform/s3-skills-bucket/main.tf
#
# Phase D: S3 bucket for Hermes skill artifact storage (SaaS mode).
#
# APPLY GATE: This module must be reviewed and applied by Blake before
# hermes_storage.s3_skills.S3SkillSource can read/write skills in production.
#
# What this creates:
#   1. S3 bucket: hermes-saas-skills (or var.bucket_name)
#   2. Block Public Access configuration (all public access blocked)
#   3. Server-side encryption at rest (SSE-S3)
#   4. Bucket versioning (enabled — protects against accidental skill deletion)
#   5. Lifecycle rule: expire non-current versions after 30 days
#   6. IAM policy document (output only) for Fargate task role to get/put/delete
#      objects under hermes-skills/{tenant_slug}/ prefix
#   7. CORS configuration for potential future web-based skill editor
#
# What this does NOT create:
#   - The IAM role itself (managed by the ECS/Fargate Terraform in agentic-hub)
#   - Per-tenant IAM policies (provisioned at tenant onboarding time, not here)
#   - CloudFront distribution (not needed for backend-to-backend)
#
# Assumptions:
#   - AWS provider is configured with region = us-east-1 (same as Neon endpoint)
#   - The Fargate task role ARN is passed in via var.fargate_task_role_arn
#   - S3 bucket name must be globally unique — default uses account ID suffix

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

variable "bucket_name" {
  type        = string
  default     = "hermes-saas-skills"
  description = "S3 bucket name for skill artifacts. Must be globally unique."
}

variable "fargate_task_role_arn" {
  type        = string
  description = "ARN of the ECS Fargate task IAM role that needs S3 access."
}

variable "environment" {
  type        = string
  default     = "prod"
  description = "Deployment environment tag (prod, staging, dev)."
}

# ---------------------------------------------------------------------------
# S3 bucket
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "skills" {
  bucket = var.bucket_name

  tags = {
    Name        = "Hermes SaaS Skill Artifacts"
    Environment = var.environment
    ManagedBy   = "terraform"
    Plan        = "hermes-001-D"
  }
}

# Block all public access (skills are internal; no public URLs needed).
resource "aws_s3_bucket_public_access_block" "skills" {
  bucket = aws_s3_bucket.skills.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Server-side encryption at rest with SSE-S3 (AES-256).
# Upgrade to SSE-KMS with customer-managed key for stricter key control.
resource "aws_s3_bucket_server_side_encryption_configuration" "skills" {
  bucket = aws_s3_bucket.skills.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Versioning: protects against accidental skill overwrites or deletions.
resource "aws_s3_bucket_versioning" "skills" {
  bucket = aws_s3_bucket.skills.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Lifecycle: expire non-current versions after 30 days to control storage cost.
resource "aws_s3_bucket_lifecycle_configuration" "skills" {
  bucket = aws_s3_bucket.skills.id

  # Expire non-current versions (previous SKILL.md iterations) after 30 days.
  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {
      prefix = ""
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  # Transition infrequently-accessed skills to S3-IA after 90 days.
  # Skills that haven't been used in 90 days are likely stale.
  rule {
    id     = "transition-to-ia"
    status = "Enabled"

    filter {
      prefix = "hermes-skills/"
    }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
  }
}

# CORS: allow browser-based skill editor (future feature) to read skill files.
resource "aws_s3_bucket_cors_configuration" "skills" {
  bucket = aws_s3_bucket.skills.id

  cors_rule {
    allowed_headers = ["Content-Type", "Authorization"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["https://*.hermes-agent.ai"]  # Update with actual domain.
    expose_headers  = ["ETag"]
    max_age_seconds = 3600
  }
}

# ---------------------------------------------------------------------------
# IAM policy for Fargate task role
# ---------------------------------------------------------------------------

# Policy document: allows the Fargate task role to read/write skills.
# This is a DATA source — it does NOT create an IAM policy resource.
# You must attach this policy to the Fargate task role separately.
#
# Tenant isolation: per-tenant IAM policies should be provisioned with
# a Condition on s3:prefix = "hermes-skills/{tenant_slug}/*" instead of
# this global policy. This policy is for the hermes_app service role that
# manages ALL tenants (used by the gateway Fargate task).
data "aws_iam_policy_document" "skills_rw" {
  statement {
    sid    = "HermesSkillsReadWrite"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:GetObjectVersion",
    ]

    resources = [
      "${aws_s3_bucket.skills.arn}/hermes-skills/*",
    ]
  }

  statement {
    sid    = "HermesSkillsList"
    effect = "Allow"

    actions = [
      "s3:ListBucket",
      "s3:ListBucketVersions",
    ]

    resources = [aws_s3_bucket.skills.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["hermes-skills/*"]
    }
  }

  # HeadBucket health check: gateway/health.py calls s3.head_bucket() to verify
  # the bucket exists. AWS evaluates this against s3:ListBucket without a prefix
  # argument, so the prefix-conditioned statement above denies. Grant unconditional
  # ListBucket-on-bucket here (the bucket-level only; key-listing still goes
  # through the conditioned statement above).
  statement {
    sid       = "HermesSkillsHeadBucket"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.skills.arn]
  }
}

# Create the IAM policy as a managed policy so it can be attached to the role.
resource "aws_iam_policy" "skills_rw" {
  name        = "HermesSkillsS3ReadWrite"
  description = "Allows Hermes Fargate task to read/write skill artifacts in S3."
  policy      = data.aws_iam_policy_document.skills_rw.json

  tags = {
    ManagedBy = "terraform"
    Plan      = "hermes-001-D"
  }
}

# Attach the policy to the Fargate task role.
# Note: aws_iam_role_policy_attachment requires the role NAME, not the ARN.
# Extract the role name from the ARN format `arn:aws:iam::<acct>:role/<name>`.
resource "aws_iam_role_policy_attachment" "skills_fargate" {
  role       = element(split("/", var.fargate_task_role_arn), length(split("/", var.fargate_task_role_arn)) - 1)
  policy_arn = aws_iam_policy.skills_rw.arn
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "bucket_name" {
  description = "S3 bucket name to set as S3_SKILLS_BUCKET env var."
  value       = aws_s3_bucket.skills.id
}

output "bucket_arn" {
  description = "S3 bucket ARN."
  value       = aws_s3_bucket.skills.arn
}

output "bucket_region" {
  description = "S3 bucket region."
  value       = var.aws_region
}

output "iam_policy_arn" {
  description = "IAM policy ARN to attach to additional task roles if needed."
  value       = aws_iam_policy.skills_rw.arn
}
