# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

variable "tenant_id" {
  description = "The tenant this deployment serves. One tenant per deployment, by design."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$", var.tenant_id))
    error_message = "tenant_id must be lowercase alphanumeric with hyphens, 3-40 characters."
  }
}

variable "region" {
  description = "AWS region. The customer's data-residency answer; this module has no default."
  type        = string
}

variable "tags" {
  description = "Extra tags applied to every resource."
  type        = map(string)
  default     = {}
}

# ---------------------------------------------------------------------------
# Network. Bring your own: this module does not create a VPC, because a customer
# with an existing landing zone should not be handed a second one.
# ---------------------------------------------------------------------------

variable "vpc_id" {
  description = "VPC to deploy into."
  type        = string
}

variable "subnet_id" {
  description = <<-EOT
    Private subnet for the runtime instance. Private is not enforceable here, but it is the
    intent: the instance takes no inbound traffic and is reached through SSM Session Manager,
    so it needs egress only. A subnet with a NAT gateway or the relevant VPC endpoints.
  EOT
  type        = string
}

# ---------------------------------------------------------------------------
# Compute. Single-node by design: state is SQLite on an attached volume, whose
# guarantees hold across processes on one host and not across hosts.
# ---------------------------------------------------------------------------

variable "instance_type" {
  description = "EC2 instance type for the runtime."
  type        = string
  default     = "t3.large"
}

variable "image_uri" {
  description = <<-EOT
    Container image the instance runs, in the customer's own registry
    (e.g. 111122223333.dkr.ecr.eu-west-2.amazonaws.com/nova:1.4.0).

    Building and pushing that image is a separate, deliberate step: this module grants the
    instance permission to pull it and does not build it. Pin a digest or an immutable tag —
    a mutable tag means the thing that passed review is not the thing that runs.
  EOT
  type        = string

  validation {
    condition     = length(trimspace(var.image_uri)) > 0
    error_message = "image_uri is required."
  }
}

variable "worker_image_uri" {
  description = <<-EOT
    Full Hermes runtime image that runs the kanban dispatcher, in the customer's own
    registry. Empty (the default) deploys the control plane alone.

    **This is the half that executes work, and it is a different image on purpose.** The
    NOVA control-plane image governs and observes; it holds neither the runtime's
    dependencies nor a model credential, so it cannot run an agent and does not pretend
    to. The dispatcher claims `ready` tasks and shells out to `hermes -p <assignee>
    chat -q`, which needs the whole runtime. Deploy with this empty and the Control
    Center will create durable tasks that nothing ever claims — NOVA now says so on the
    tasks screen rather than leaving it to be discovered.

    Built from the repository's root `Dockerfile`. The unit runs it with HERMES_UID=10001
    so it matches the state volume's owner, and HERMES_HOME pointed at the same home the
    control plane serves, so both processes see one board.
  EOT
  type        = string
  default     = ""
}

variable "ami_id" {
  description = <<-EOT
    AMI for the host. Empty selects the latest Amazon Linux 2023 via SSM parameter, which is
    convenient for a first deployment and wrong for a stable one: a new AMI release will
    replace the instance. Pin an AMI id before this carries production work.
  EOT
  type        = string
  default     = ""
}

variable "volume_gb" {
  description = "Size of the encrypted EBS volume holding NOVA's state."
  type        = number
  default     = 100

  validation {
    condition     = var.volume_gb >= 20
    error_message = "volume_gb must be at least 20."
  }
}

variable "kms_key_arn" {
  description = <<-EOT
    Customer-managed KMS key for the state volume, the log group and the secrets this
    deployment reads. Empty creates one with a rotation schedule. Supply your own if your
    key policy is centrally managed.
  EOT
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention for the runtime's operational logs."
  type        = number
  default     = 365
}

# ---------------------------------------------------------------------------
# Inference. Bedrock is the inference layer, not the hosting target.
# ---------------------------------------------------------------------------

variable "bedrock_model_ids" {
  description = <<-EOT
    Bedrock model ids the runtime role may invoke, e.g.
    ["eu.anthropic.claude-sonnet-4-20250514-v1:0"].

    Enumerated rather than wildcarded on purpose: "which models can this system call" is a
    question a reviewer will ask, and the answer should be a list they can read. An empty
    list grants no Bedrock access at all, which is the correct setting for a deployment whose
    models are reached over an endpoint instead.
  EOT
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for id in var.bedrock_model_ids : !strcontains(id, "*")])
    error_message = "Bedrock model ids must be literal. A wildcard here grants every model in the region."
  }
}

variable "bedrock_inference_profile_arns" {
  description = <<-EOT
    ARNs of cross-region inference profiles the runtime may invoke, if you use them. Invoking
    through a profile needs both this and the underlying model ids above.
  EOT
  type        = list(string)
  default     = []
}

# ---------------------------------------------------------------------------
# Secrets. NOVA names credentials; it never holds them. This is where the names
# are allowed to resolve to values.
# ---------------------------------------------------------------------------

variable "secret_prefix" {
  description = <<-EOT
    Secrets Manager path prefix the runtime may read, e.g. "nova/acme/". Scoped to a prefix
    rather than to "*" so that adding a secret to this deployment is a deliberate act of
    naming it inside the prefix, and reading anything else is a permissions failure rather
    than a silent success.
  EOT
  type        = string
  default     = ""

  validation {
    condition     = !strcontains(var.secret_prefix, "*")
    error_message = "secret_prefix must be a literal prefix, not a wildcard pattern."
  }
}

# ---------------------------------------------------------------------------
# Integrations. The whole point of the module.
# ---------------------------------------------------------------------------

variable "integrations" {
  description = <<-EOT
    Customer systems agents may reach, each as a separately-scoped role assumed by the
    runtime role. Generated from the tenant bundle by `nova deploy render`; hand-editing is
    allowed and the same rules apply either way.

    The runtime role itself carries NO customer-data permissions. Every grant lives in one of
    these roles, so "what can the agents reach?" is answered by reading this list, and adding
    to it is an IAM change the customer's own security team reviews.
  EOT

  type = list(object({
    id          = string
    description = optional(string, "")
    statements = list(object({
      actions   = list(string)
      resources = list(string)
      condition = optional(map(map(list(string))), {})
    }))
  }))

  default = []

  validation {
    condition     = length(var.integrations) == length(distinct([for i in var.integrations : i.id]))
    error_message = "Integration ids must be unique."
  }

  validation {
    condition = alltrue([
      for i in var.integrations : can(regex("^[a-z0-9][a-z0-9_-]{1,48}[a-z0-9]$", i.id))
    ])
    error_message = "Integration ids must be lowercase alphanumeric with hyphens or underscores."
  }

  # The same refusals `nova deploy render` applies, restated here because this variable can
  # be hand-written. A check that only runs in the generator is a check an operator can skip.
  validation {
    condition = alltrue(flatten([
      for i in var.integrations : [
        for s in i.statements : [for a in s.actions : !strcontains(a, "*")]
      ]
    ]))
    error_message = "Integration actions must be literal. A wildcard action grants more than anyone reviewed."
  }

  validation {
    condition = alltrue(flatten([
      for i in var.integrations : [
        for s in i.statements : [for r in s.resources : r != "*"]
      ]
    ]))
    error_message = "Integration resources must not be \"*\". Name the buckets, tables or queues."
  }

  validation {
    condition = alltrue(flatten([
      for i in var.integrations : [
        for s in i.statements : [
          for a in s.actions :
          !contains(["iam", "sts", "organizations", "account", "kms"], lower(split(":", a)[0]))
        ]
      ]
    ]))
    error_message = <<-EOT
      Integration actions may not use iam, sts, organizations, account or kms. Those grant the
      ability to grant, which turns a scoped integration into an unscoped one. Attach the
      permission to the integration role in your own IAM pipeline if you genuinely need it.
    EOT
  }
}

variable "max_session_duration" {
  description = "Maximum session length when the runtime assumes an integration role."
  type        = number
  default     = 3600
}
