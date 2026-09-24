# ---------------------------------------------------------------------------
# Operator access to the Control Centre
# ---------------------------------------------------------------------------
#
# The control plane listens on the instance's loopback only (see the nova unit in
# user_data.sh.tftpl); nothing reaches it from the network. People reach it through an SSM
# port-forwarding session, so *who may open that session* is the Control Centre's login,
# and it is AWS's: an IAM principal with this policy, authenticated however the account
# authenticates people (IAM Identity Center with MFA, or IAM users with MFA below).
#
# The policy grants exactly one thing — forwarding a local port to the Control Centre port
# on this one instance — and nothing else Session Manager can do:
#
#   no shell      ssm:SessionDocumentAccessCheck makes Session Manager check the document
#                 the caller names, so a StartSession without one (which means the default
#                 shell) is refused rather than quietly allowed;
#   no other port the session document below is NOVA's own, with the remote port fixed to
#                 the Control Centre's. AWS's generic AWS-StartPortForwardingSession takes
#                 any port, which would let this login reach anything else listening on
#                 the host.
#
# Attach it to an IAM Identity Center permission set (the intended shape — see
# deploy/aws/ACCESS.md), or enable the IAM group below for teams without Identity Center.

#: The control plane's port, set in the image's entrypoint (NOVA_BIND_PORT, default 8787).
locals {
  control_port = 8787
}

resource "aws_ssm_document" "console" {
  name            = "${local.name_prefix}-console"
  document_type   = "Session"
  document_format = "JSON"
  content = jsonencode({
    schemaVersion = "1.0"
    description   = "Forward a local port to the ${var.tenant_id} Control Centre. Nothing else."
    sessionType   = "Port"
    parameters = {
      portNumber = {
        type          = "String"
        description   = "The Control Centre port on the instance (fixed)."
        allowedValues = [tostring(local.control_port)]
        default       = tostring(local.control_port)
      }
      localPortNumber = {
        type           = "String"
        description    = "Port on your machine."
        allowedPattern = "^([1-9][0-9]{0,4})$"
        default        = tostring(local.control_port)
      }
    }
    properties = {
      portNumber      = "{{ portNumber }}"
      type            = "LocalPortForwarding"
      localPortNumber = "{{ localPortNumber }}"
    }
  })
}

data "aws_iam_policy_document" "console_access" {
  statement {
    sid     = "ForwardTheControlCentrePortOnThisInstance"
    effect  = "Allow"
    actions = ["ssm:StartSession"]
    resources = [
      aws_instance.runtime.arn,
      aws_ssm_document.console.arn,
    ]
    condition {
      test     = "BoolIfExists"
      variable = "ssm:SessionDocumentAccessCheck"
      values   = ["true"]
    }
  }

  # Ending or resuming one's own session only. Matched on the tag Session Manager stamps
  # with the starter's user id, which is AWS's documented form and holds for IAM users and
  # Identity Center sessions alike — a session-id prefix does not: an Identity Center
  # session is named after the person's sign-in, which neither aws:username nor aws:userid
  # is, so a prefix rule left those users unable to close their own sessions.
  statement {
    sid       = "ManageOwnSessionsOnly"
    effect    = "Allow"
    actions   = ["ssm:TerminateSession", "ssm:ResumeSession"]
    resources = ["arn:${local.partition}:ssm:*:${local.account_id}:session/*"]
    condition {
      test     = "StringLike"
      variable = "ssm:resourceTag/aws:ssmmessages:session-id"
      values   = ["$${aws:userid}"]
    }
  }
}

resource "aws_iam_policy" "console_access" {
  name        = "${local.name_prefix}-console-access"
  description = "Open the ${var.tenant_id} Control Centre through an SSM port forward. No shell."
  policy      = data.aws_iam_policy_document.console_access.json
}

# -- optional: an IAM group, for accounts without IAM Identity Center ------------------

# Every member must have MFA: everything but managing their own MFA device is denied until
# they sign in with it. This is AWS's documented pattern for "MFA or nothing".
data "aws_iam_policy_document" "require_mfa" {
  statement {
    sid    = "ManageOwnMfaDevice"
    effect = "Allow"
    actions = [
      "iam:CreateVirtualMFADevice", "iam:EnableMFADevice", "iam:ListMFADevices",
      "iam:ResyncMFADevice", "iam:GetUser", "iam:ChangePassword",
    ]
    resources = [
      "arn:${local.partition}:iam::${local.account_id}:user/$${aws:username}",
      "arn:${local.partition}:iam::${local.account_id}:mfa/$${aws:username}",
    ]
  }
  statement {
    sid    = "DenyEverythingElseWithoutMfa"
    effect = "Deny"
    not_actions = [
      "iam:CreateVirtualMFADevice", "iam:EnableMFADevice", "iam:ListMFADevices",
      "iam:ResyncMFADevice", "iam:GetUser", "iam:ChangePassword", "sts:GetSessionToken",
    ]
    resources = ["*"]
    condition {
      test     = "BoolIfExists"
      variable = "aws:MultiFactorAuthPresent"
      values   = ["false"]
    }
  }
}

resource "aws_iam_group" "console" {
  count = var.console_iam_group_enabled ? 1 : 0
  name  = "${local.name_prefix}-console"
}

resource "aws_iam_group_policy_attachment" "console_access" {
  count      = var.console_iam_group_enabled ? 1 : 0
  group      = aws_iam_group.console[0].name
  policy_arn = aws_iam_policy.console_access.arn
}

resource "aws_iam_group_policy" "console_require_mfa" {
  count  = var.console_iam_group_enabled ? 1 : 0
  name   = "require-mfa"
  group  = aws_iam_group.console[0].name
  policy = data.aws_iam_policy_document.require_mfa.json
}
