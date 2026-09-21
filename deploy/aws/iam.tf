# ---------------------------------------------------------------------------
# The runtime role.
#
# It carries no customer-data permissions. Everything it can do is here, in one
# readable place: pull its own image, call the models it was told about, read the
# secrets in its own prefix, write its own logs, and assume exactly the integration
# roles declared for this tenant. Nothing else.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "runtime_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "runtime" {
  name                 = "${local.name_prefix}-runtime"
  description          = "NOVA runtime for tenant ${var.tenant_id}. No customer-data permissions by design."
  assume_role_policy   = data.aws_iam_policy_document.runtime_trust.json
  max_session_duration = var.max_session_duration

  # An explicit boundary, so that even a mistaken future policy attachment cannot
  # widen this role past what the boundary allows.
  permissions_boundary = aws_iam_policy.runtime_boundary.arn
}

resource "aws_iam_instance_profile" "runtime" {
  name = "${local.name_prefix}-runtime"
  role = aws_iam_role.runtime.name
}

# --- the boundary ----------------------------------------------------------

data "aws_iam_policy_document" "runtime_boundary" {
  # Services this deployment is allowed to touch at all. The boundary is a ceiling,
  # not a grant: nothing is permitted unless an attached policy also allows it.
  statement {
    sid    = "ServicesThisDeploymentUses"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetAuthorizationToken",
      "ecr:GetDownloadUrlForLayer",
      "logs:CreateLogStream",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents",
      "secretsmanager:GetSecretValue",
      "ssm:GetParameter",
      "ssm:GetParameters",
      # Registration and the five-minute heartbeat. The SSM Agent's health module
      # calls exactly this one API, and an instance whose heartbeat is denied never
      # becomes a managed node — so `aws ssm start-session` fails and the deployment
      # has no way in at all, this being the only route (no inbound port, no key).
      # AmazonSSMManagedInstanceCore grants it; a boundary that omits it caps the
      # grant away, which is what happened on the first deployment.
      "ssm:UpdateInstanceInformation",
      # Session Manager's control and data channels — the session itself.
      "ssmmessages:*",
      "ec2messages:*",
      "kms:Decrypt",
      "kms:GenerateDataKey",
      "sts:AssumeRole",
    ]
    resources = ["*"]
  }

  # Whatever the integrations need, capped to the same literal resources they name.
  # Without this the boundary would forbid the very grants the integration roles make.
  dynamic "statement" {
    for_each = length(local.integration_statements) > 0 ? [1] : []

    content {
      sid       = "IntegrationCeiling"
      effect    = "Allow"
      actions   = local.all_integration_actions
      resources = local.all_integration_resources
    }
  }

  # The refusal that survives every future edit. A role that cannot change IAM cannot
  # grant itself anything, whatever is attached to it later.
  statement {
    sid    = "NeverSelfEscalate"
    effect = "Deny"
    actions = [
      "iam:*",
      "organizations:*",
      "account:*",
      "sts:AssumeRoleWithSAML",
      "sts:AssumeRoleWithWebIdentity",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "runtime_boundary" {
  name        = "${local.name_prefix}-runtime-boundary"
  description = "Permissions ceiling for the NOVA runtime role. Nothing attached to that role can exceed this."
  policy      = data.aws_iam_policy_document.runtime_boundary.json
}

# --- what the runtime may actually do --------------------------------------

data "aws_iam_policy_document" "runtime" {
  dynamic "statement" {
    for_each = length(local.bedrock_resources) > 0 ? [1] : []

    content {
      sid       = "InvokeDeclaredModels"
      effect    = "Allow"
      actions   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
      resources = local.bedrock_resources
    }
  }

  statement {
    sid    = "PullItsOwnImage"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [local.image_repository_arn]
  }

  statement {
    sid       = "EcrAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # This call takes no resource; AWS models it as account-wide.
  }

  statement {
    sid       = "WriteItsOwnLogs"
    effect    = "Allow"
    actions   = ["logs:CreateLogStream", "logs:DescribeLogStreams", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.runtime.arn}:*"]
  }

  dynamic "statement" {
    for_each = var.secret_prefix == "" ? [] : [1]

    content {
      sid       = "ReadItsOwnSecrets"
      effect    = "Allow"
      actions   = ["secretsmanager:GetSecretValue"]
      resources = [local.secret_arn_pattern]
    }
  }

  statement {
    sid       = "UseItsOwnKey"
    effect    = "Allow"
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [local.kms_key_arn]
  }

  # Exactly the integration roles declared for this tenant, named one by one. This is
  # the only route from the runtime to any customer system, and it is enumerable.
  dynamic "statement" {
    for_each = length(aws_iam_role.integration) > 0 ? [1] : []

    content {
      sid       = "AssumeDeclaredIntegrationsOnly"
      effect    = "Allow"
      actions   = ["sts:AssumeRole"]
      resources = [for role in aws_iam_role.integration : role.arn]
    }
  }
}

resource "aws_iam_role_policy" "runtime" {
  name   = "runtime"
  role   = aws_iam_role.runtime.id
  policy = data.aws_iam_policy_document.runtime.json
}

# Session Manager, so the instance needs no inbound port and no SSH key. An operator
# reaching the box is an auditable API call rather than a key someone still has.
resource "aws_iam_role_policy_attachment" "session_manager" {
  role       = aws_iam_role.runtime.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# ---------------------------------------------------------------------------
# One role per integration.
#
# Trust names the runtime role and nothing else, so a role that leaks by name is
# still useless to anyone who is not this deployment.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "integration_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.runtime.arn]
    }

    # A confused-deputy guard: even a principal that could assume this role must
    # present the external id, which only this deployment knows to send.
    condition {
      test     = "StringEquals"
      variable = "sts:ExternalId"
      values   = [local.external_id]
    }
  }
}

resource "aws_iam_role" "integration" {
  for_each = { for i in var.integrations : i.id => i }

  name                 = "${local.name_prefix}-int-${each.key}"
  description          = each.value.description != "" ? each.value.description : "NOVA integration ${each.key}"
  assume_role_policy   = data.aws_iam_policy_document.integration_trust.json
  max_session_duration = var.max_session_duration
  permissions_boundary = aws_iam_policy.runtime_boundary.arn
}

data "aws_iam_policy_document" "integration" {
  for_each = { for i in var.integrations : i.id => i }

  dynamic "statement" {
    for_each = each.value.statements

    content {
      effect    = "Allow"
      actions   = statement.value.actions
      resources = statement.value.resources

      dynamic "condition" {
        for_each = statement.value.condition

        content {
          test     = condition.key
          variable = keys(condition.value)[0]
          values   = values(condition.value)[0]
        }
      }
    }
  }
}

resource "aws_iam_role_policy" "integration" {
  for_each = { for i in var.integrations : i.id => i }

  name   = "integration"
  role   = aws_iam_role.integration[each.key].id
  policy = data.aws_iam_policy_document.integration[each.key].json
}
