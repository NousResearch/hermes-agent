locals {
  name_prefix = "nova-${var.tenant_id}"

  # Distinguishes this deployment when it assumes an integration role. Derived rather than
  # supplied so that two deployments of the same tenant in one account cannot share it.
  external_id = "nova-${var.tenant_id}-${data.aws_caller_identity.current.account_id}-${var.region}"

  partition  = data.aws_partition.current.partition
  account_id = data.aws_caller_identity.current.account_id

  # A Bedrock model id is one of two different things wearing the same shape, and the ARN
  # you need is different for each. Rendering every id as a foundation-model ARN — which
  # this did — produces an ARN that does not exist for the second kind, and the failure
  # surfaces as AccessDenied at the first inference, not at apply.
  #
  #   anthropic.claude-sonnet-4-6       a FOUNDATION MODEL, invoked on-demand in one region
  #                                     arn:...:bedrock:<region>::foundation-model/<id>
  #   eu.anthropic.claude-sonnet-4-6    an INFERENCE PROFILE, account-scoped, routing across
  #                                     a geography
  #                                     arn:...:bedrock:<region>:<account>:inference-profile/<id>
  #
  # The region prefix is what distinguishes them. This list mirrors the runtime's own
  # (`agent/anthropic_message_convert.py::_BEDROCK_REGION_PREFIXES`); the two must agree,
  # because IAM has to name exactly the id boto3 ends up invoking.
  bedrock_profile_prefixes = ["global.", "us.", "eu.", "apac.", "ap.", "au.", "jp.", "ca.", "sa.", "me.", "af."]

  bedrock_profile_ids = [
    for id in var.bedrock_model_ids : id
    if anytrue([for prefix in local.bedrock_profile_prefixes : startswith(id, prefix)])
  ]
  bedrock_foundation_ids = [
    for id in var.bedrock_model_ids : id
    if !anytrue([for prefix in local.bedrock_profile_prefixes : startswith(id, prefix)])
  ]

  # "eu.anthropic.claude-sonnet-4-6" -> "anthropic.claude-sonnet-4-6". A profile is a router;
  # the thing it routes to is the bare foundation model, and AWS checks permission on both.
  bedrock_profile_model_ids = [
    for id in local.bedrock_profile_ids :
    join(".", slice(split(".", id), 1, length(split(".", id))))
  ]

  bedrock_resources = distinct(concat(
    # Plain foundation models: on-demand, in this region only.
    [
      for id in local.bedrock_foundation_ids :
      "arn:${local.partition}:bedrock:${var.region}::foundation-model/${id}"
    ],
    # The profile itself, account-scoped in the requesting region.
    [
      for id in local.bedrock_profile_ids :
      "arn:${local.partition}:bedrock:${var.region}:${local.account_id}:inference-profile/${id}"
    ],
    # And the model behind it, in every region that profile may route to. AWS requires
    # InvokeModel on the foundation model in the requesting region AND in each destination
    # region of the profile; permission on the profile alone is not enough.
    flatten([
      for id in local.bedrock_profile_model_ids : [
        for profile_region in var.bedrock_profile_regions :
        "arn:${local.partition}:bedrock:${profile_region}::foundation-model/${id}"
      ]
    ]),
    # Explicit ARNs, for a deployment that would rather state them itself.
    var.bedrock_inference_profile_arns,
  ))

  # "111122223333.dkr.ecr.eu-west-2.amazonaws.com/nova:1.4.0" -> the repository it lives in.
  # Scoped to the one repository rather than the whole registry: an instance that can pull
  # any image in the account can pull one nobody reviewed.
  #
  # This deployment legitimately runs two images from two repositories — the control plane
  # and, when declared, the Hermes runtime that dispatches its work. They are parsed by one
  # expression rather than two copies of it: the tag-versus-digest handling below is the
  # fiddly part, and a second hand-written copy is a second place to get it subtly wrong.
  # An empty worker_image_uri drops out here, so a control-plane-only deployment derives
  # exactly the one ARN it did before.
  ecr_image_uris = {
    for name, uri in { control_plane = var.image_uri, worker = var.worker_image_uri } :
    name => uri if trimspace(uri) != ""
  }

  # The registry host: "111122223333.dkr.ecr.eu-west-2.amazonaws.com". Its account and
  # region are positional — index 3 is the region for both `ecr.` and `ecr-fips.` hosts,
  # and in China, where the suffix is longer but the prefix is not.
  ecr_registries = {
    for name, uri in local.ecr_image_uris : name => split("/", uri)[0]
  }

  # Everything after the host, with a ":tag" or an "@sha256:..." digest stripped. The
  # `join`/`slice` keeps a namespaced repository ("team/nova") whole.
  ecr_repositories = {
    for name, uri in local.ecr_image_uris :
    name => split(":", split("@", join("/", slice(split("/", uri), 1, length(split("/", uri)))))[0])[0]
  }

  ecr_repository_arns = {
    for name, repository in local.ecr_repositories :
    name => format(
      "arn:%s:ecr:%s:%s:repository/%s",
      local.partition,
      split(".", local.ecr_registries[name])[3],
      split(".", local.ecr_registries[name])[0],
      repository,
    )
  }

  # Sorted by key, so a plan does not churn on map ordering, and deduplicated: pointing
  # both images at one repository under different tags is a perfectly ordinary layout, and
  # it should produce one ARN rather than the same one twice. IAM would ignore the
  # duplicate; a policy document that lists it is still a policy nobody can read at a
  # glance, and it churns the diff every time the other image moves.
  image_repository_arns = distinct(values(local.ecr_repository_arns))

  secret_arn_pattern = "arn:${local.partition}:secretsmanager:${var.region}:${local.account_id}:secret:${var.secret_prefix}*"

  kms_key_arn = var.kms_key_arn != "" ? var.kms_key_arn : aws_kms_key.state[0].arn

  # Named once: the key policy scopes CloudWatch Logs to exactly this log group, so the
  # two must not be able to drift apart.
  log_group_name = "/nova/${var.tenant_id}"

  integration_statements = flatten([for i in var.integrations : i.statements])

  all_integration_actions = distinct(flatten([
    for s in local.integration_statements : s.actions
  ]))

  all_integration_resources = distinct(flatten([
    for s in local.integration_statements : s.resources
  ]))
}

# ---------------------------------------------------------------------------
# Key, log group, volume
# ---------------------------------------------------------------------------

# The key policy.
#
# Without one, KMS applies its default: a single statement granting the account root
# `kms:*`, which delegates access control to IAM. That is enough for EBS and for the
# runtime role, whose grant is an IAM policy (iam.tf) — but NOT for CloudWatch Logs.
# CWL calls KMS as a *service principal*, not as an IAM identity, so no IAM policy can
# reach it; the key policy is the only place it can be allowed. Creating the log group
# fails with "The specified KMS key does not exist or is not allowed to be used with
# Arn '...log-group:/nova/<tenant>'" until this exists.
#
# Only for the key this module creates. An externally supplied `kms_key_arn` is a
# customer-managed key whose policy is theirs — the `count` below already scopes this,
# and there is deliberately no `aws_kms_key_policy` resource that would reach out and
# rewrite a key we do not own.
data "aws_iam_policy_document" "state_key" {
  count = var.kms_key_arn == "" ? 1 : 0

  # Keep the default statement. Removing it orphans the key: KMS would no longer honour
  # any IAM policy against it, including this account's administrators, and a key policy
  # can only be changed by a principal the key policy already allows. There is no
  # recovery from that short of AWS support, which is why AWS documents it as the one
  # statement you do not drop.
  statement {
    sid       = "EnableIAMUserPermissions"
    effect    = "Allow"
    actions   = ["kms:*"]
    resources = ["*"]

    principals {
      type        = "AWS"
      identifiers = ["arn:${local.partition}:iam::${local.account_id}:root"]
    }
  }

  # CloudWatch Logs, in this deployment's region only. The service principal is
  # regional — `logs.eu-west-2.amazonaws.com` cannot be used by CWL in another region —
  # and it is read from the provider rather than written down, so a deployment into a
  # different region is correct without editing this file.
  statement {
    sid    = "AllowCloudWatchLogs"
    effect = "Allow"

    # The set AWS documents for log-group encryption. Narrower than it looks: KMS has no
    # single "use this key" action, so encrypt, decrypt, re-encrypt and data-key
    # generation are each named. `Describe*` is metadata only. Notably absent are the
    # management actions — no PutKeyPolicy, no ScheduleKeyDeletion, no CreateGrant.
    actions = [
      "kms:Encrypt*",
      "kms:Decrypt*",
      "kms:ReEncrypt*",
      "kms:GenerateDataKey*",
      "kms:Describe*",
    ]
    resources = ["*"]

    principals {
      type        = "Service"
      identifiers = ["logs.${data.aws_region.current.name}.amazonaws.com"]
    }

    # The real constraint. CWL sets the log group's ARN as the encryption context on
    # every call, so this limits the grant to log groups this deployment owns — not
    # every log group in the account. Without it, any log group in the region could be
    # encrypted with this tenant's key.
    #
    # `ArnLike` with a trailing wildcard rather than `ArnEquals`: the context is
    # documented as the bare log-group ARN, but a `:*` suffix appears in some responses
    # and an exact match that guessed wrong would fail the apply a second time. The
    # wildcard sits after the full tenant-scoped name, so the widening is bounded by the
    # prefix this module itself creates.
    condition {
      test     = "ArnLike"
      variable = "kms:EncryptionContext:aws:logs:arn"
      values   = ["arn:${local.partition}:logs:${data.aws_region.current.name}:${local.account_id}:log-group:${local.log_group_name}*"]
    }
  }
}

resource "aws_kms_key" "state" {
  count = var.kms_key_arn == "" ? 1 : 0

  description             = "NOVA state, logs and secrets for tenant ${var.tenant_id}"
  enable_key_rotation     = true
  deletion_window_in_days = 30
  policy                  = data.aws_iam_policy_document.state_key[0].json
}

resource "aws_kms_alias" "state" {
  count = var.kms_key_arn == "" ? 1 : 0

  name          = "alias/${local.name_prefix}"
  target_key_id = aws_kms_key.state[0].key_id
}

resource "aws_cloudwatch_log_group" "runtime" {
  name              = local.log_group_name
  retention_in_days = var.log_retention_days
  kms_key_id        = local.kms_key_arn
}

# State lives on its own volume so that replacing the instance does not replace the
# audit log. SQLite on an attached block device, single node: see the module README.
resource "aws_ebs_volume" "state" {
  availability_zone = data.aws_subnet.runtime.availability_zone
  size              = var.volume_gb
  type              = "gp3"
  encrypted         = true
  kms_key_id        = local.kms_key_arn

  tags = { Name = "${local.name_prefix}-state" }

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_volume_attachment" "state" {
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.state.id
  instance_id = aws_instance.runtime.id
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

data "aws_subnet" "runtime" {
  id = var.subnet_id
}

data "aws_ssm_parameter" "al2023" {
  count = var.ami_id == "" ? 1 : 0
  name  = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

# No ingress rules at all. The instance is reached through SSM Session Manager, which
# means there is no port to expose, no key to rotate and no bastion to maintain.
resource "aws_security_group" "runtime" {
  name        = "${local.name_prefix}-runtime"
  description = "NOVA runtime: egress only, reached via SSM Session Manager."
  vpc_id      = var.vpc_id

  tags = { Name = "${local.name_prefix}-runtime" }
}

resource "aws_vpc_security_group_egress_rule" "https" {
  security_group_id = aws_security_group.runtime.id
  description       = "Model endpoint, ECR, Secrets Manager, SSM"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
}

resource "aws_instance" "runtime" {
  ami                    = var.ami_id != "" ? var.ami_id : data.aws_ssm_parameter.al2023[0].value
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [aws_security_group.runtime.id]
  iam_instance_profile   = aws_iam_instance_profile.runtime.name

  # IMDSv2 only: a server-side request forgery against anything running here cannot read
  # the instance credentials without a token it has no way to obtain.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    encrypted   = true
    kms_key_id  = local.kms_key_arn
    volume_size = 30
    volume_type = "gp3"
  }

  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    image_uri    = var.image_uri
    tenant_id    = var.tenant_id
    region       = var.region
    log_group    = aws_cloudwatch_log_group.runtime.name
    external_id  = local.external_id
    state_device = "/dev/xvdf"
    state_mount  = "/var/lib/nova"
    integrations = join(",", [for i in var.integrations : i.id])
    # Empty deploys the control plane alone, which creates durable tasks that nothing
    # claims. See variables.tf for why executing work needs the other image.
    worker_image_uri = var.worker_image_uri
  })

  tags = { Name = "${local.name_prefix}-runtime" }
}
