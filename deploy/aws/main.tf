locals {
  name_prefix = "nova-${var.tenant_id}"

  # Distinguishes this deployment when it assumes an integration role. Derived rather than
  # supplied so that two deployments of the same tenant in one account cannot share it.
  external_id = "nova-${var.tenant_id}-${data.aws_caller_identity.current.account_id}-${var.region}"

  partition  = data.aws_partition.current.partition
  account_id = data.aws_caller_identity.current.account_id

  bedrock_resources = concat(
    [
      for id in var.bedrock_model_ids :
      "arn:${local.partition}:bedrock:${var.region}::foundation-model/${id}"
    ],
    var.bedrock_inference_profile_arns,
  )

  # "111122223333.dkr.ecr.eu-west-2.amazonaws.com/nova:1.4.0" -> the repository it lives in.
  # Scoped to the one repository rather than the whole registry: an instance that can pull
  # any image in the account can pull one nobody reviewed.
  image_registry   = split("/", var.image_uri)[0]
  image_repository = split(":", split("@", join("/", slice(split("/", var.image_uri), 1, length(split("/", var.image_uri)))))[0])[0]
  image_account_id = split(".", local.image_registry)[0]
  image_region     = split(".", local.image_registry)[3]

  image_repository_arn = "arn:${local.partition}:ecr:${local.image_region}:${local.image_account_id}:repository/${local.image_repository}"

  secret_arn_pattern = "arn:${local.partition}:secretsmanager:${var.region}:${local.account_id}:secret:${var.secret_prefix}*"

  kms_key_arn = var.kms_key_arn != "" ? var.kms_key_arn : aws_kms_key.state[0].arn

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

resource "aws_kms_key" "state" {
  count = var.kms_key_arn == "" ? 1 : 0

  description             = "NOVA state, logs and secrets for tenant ${var.tenant_id}"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "state" {
  count = var.kms_key_arn == "" ? 1 : 0

  name          = "alias/${local.name_prefix}"
  target_key_id = aws_kms_key.state[0].key_id
}

resource "aws_cloudwatch_log_group" "runtime" {
  name              = "/nova/${var.tenant_id}"
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
  })

  tags = { Name = "${local.name_prefix}-runtime" }
}
