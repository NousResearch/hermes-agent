---
name: terraform
description: Provision and manage infrastructure as code with Terraform. Covers the full IaC lifecycle — HCL configuration, state management, planning and applying changes, modules, workspaces, remote backends, and importing existing resources. Works with any provider (AWS, GCP, Azure, Kubernetes, and more). No API keys required — credentials come from your existing CLI config (aws configure, gcloud auth, etc.).
version: 1.0.0
author: dogiladeveloper
license: MIT
metadata:
  hermes:
    tags: [Terraform, IaC, Infrastructure, DevOps, AWS, GCP, Azure, HCL, Cloud, Provisioning, Modules, State]
    related_skills: [github-pr-workflow, github-repo-management]
    homepage: https://github.com/dogiladeveloper
---

# Terraform

Provision and manage any infrastructure with code.

## Prerequisites

- Terraform installed (`terraform version` to check)
- Install: `brew install terraform` / `choco install terraform` / [tfenv](https://github.com/tfutils/tfenv)
- Provider credentials configured via CLI (e.g. `aws configure`, `gcloud auth application-default login`)

## Quick Reference

| Action | Command |
|--------|---------|
| Initialize working directory | `terraform init` |
| Preview changes | `terraform plan` |
| Apply changes | `terraform apply` |
| Destroy all resources | `terraform destroy` |
| Show current state | `terraform show` |
| List resources in state | `terraform state list` |
| Format HCL files | `terraform fmt -recursive` |
| Validate configuration | `terraform validate` |
| Import existing resource | `terraform import <addr> <id>` |
| Unlock stuck state | `terraform force-unlock <lock-id>` |

## Helper Script

This skill includes `scripts/terraform_manager.py` — a zero-dependency CLI tool
for inspecting Terraform workspaces, state, and plan output.

```bash
python scripts/terraform_manager.py status              # workspace + state summary
python scripts/terraform_manager.py resources           # list all resources in state
python scripts/terraform_manager.py plan-summary        # parse and summarize a saved plan
python scripts/terraform_manager.py outputs             # show all output values
python scripts/terraform_manager.py workspaces          # list workspaces and active one
python scripts/terraform_manager.py validate            # validate + fmt check
python scripts/terraform_manager.py costs               # estimate resource count by type
```

---

## 1. HCL Basics

Terraform configuration is written in HCL (HashiCorp Configuration Language).

### File structure

```
my-infra/
├── main.tf          # main resources
├── variables.tf     # input variable declarations
├── outputs.tf       # output value declarations
├── providers.tf     # provider configuration
├── versions.tf      # required versions
└── terraform.tfvars # variable values (gitignored if secrets)
```

### providers.tf

```hcl
terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote backend (optional — see Section 6)
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
  # Credentials come from: ~/.aws/credentials, env vars, or IAM role
  # Never hardcode access keys here
}
```

### variables.tf

```hcl
variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "environment must be dev, staging, or production."
  }
}

variable "instance_count" {
  description = "Number of EC2 instances"
  type        = number
  default     = 2
}

variable "tags" {
  description = "Common tags applied to all resources"
  type        = map(string)
  default     = {}
}
```

### outputs.tf

```hcl
output "instance_ids" {
  description = "IDs of created EC2 instances"
  value       = aws_instance.app[*].id
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "database_endpoint" {
  description = "RDS endpoint (sensitive)"
  value       = aws_db_instance.main.endpoint
  sensitive   = true   # hidden in plan output, shown with terraform output -raw
}
```

---

## 2. Core Workflow

```bash
# 1. Initialize — download providers, set up backend
terraform init

# Re-initialize after adding a new provider
terraform init -upgrade

# 2. Format — keep HCL tidy
terraform fmt -recursive

# 3. Validate — check for syntax and config errors
terraform validate

# 4. Plan — preview what will change (no changes made)
terraform plan

# Save plan to file (required for apply in CI)
terraform plan -out=tfplan

# Plan for a specific resource only
terraform plan -target=aws_instance.app

# 5. Apply — make the changes
terraform apply           # prompts for confirmation
terraform apply -auto-approve  # skip prompt (CI/CD)
terraform apply tfplan    # apply a saved plan exactly

# 6. Destroy — remove all resources
terraform destroy
terraform destroy -auto-approve
terraform destroy -target=aws_instance.app  # destroy one resource
```

---

## 3. Variables & tfvars

```bash
# Pass variables on the command line
terraform plan -var="environment=staging" -var="instance_count=3"

# Use a tfvars file
terraform plan -var-file="staging.tfvars"

# Terraform auto-loads these files (no -var-file needed):
#   terraform.tfvars
#   terraform.tfvars.json
#   *.auto.tfvars
```

Example `staging.tfvars`:

```hcl
aws_region     = "eu-west-1"
environment    = "staging"
instance_count = 2
tags = {
  Team    = "platform"
  CostCenter = "engineering"
}
```

Environment variables (useful in CI):

```bash
# Terraform reads TF_VAR_<name> automatically
export TF_VAR_environment=production
export TF_VAR_instance_count=5
terraform apply
```

---

## 4. State Management

Terraform tracks real infrastructure in a state file (`terraform.tfstate`).

```bash
# List all resources tracked in state
terraform state list

# Show details of one resource
terraform state show aws_instance.app

# Move a resource to a new address (after renaming in HCL)
terraform state mv aws_instance.old aws_instance.new

# Remove a resource from state without destroying it
terraform state rm aws_instance.app

# Pull remote state to local file
terraform state pull > state-backup.json

# Push local state to remote backend
terraform state push state-backup.json
```

### Importing existing resources

When infrastructure was created outside Terraform:

```bash
# Import a single resource
terraform import aws_s3_bucket.my_bucket my-existing-bucket-name

# Import an EC2 instance
terraform import aws_instance.web i-0abc123def456789

# Terraform 1.5+: import blocks in HCL (preferred)
```

```hcl
# import block (Terraform 1.5+)
import {
  to = aws_s3_bucket.my_bucket
  id = "my-existing-bucket-name"
}

resource "aws_s3_bucket" "my_bucket" {
  bucket = "my-existing-bucket-name"
}
```

```bash
# Generate config for imported resource automatically
terraform plan -generate-config-out=generated.tf
```

---

## 5. Modules

Modules are reusable packages of Terraform configuration.

### Using a module

```hcl
# From Terraform Registry
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "my-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true

  tags = var.tags
}

# From a local path
module "app" {
  source = "./modules/app"

  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnets
  environment = var.environment
}

# Reference module outputs
resource "aws_security_group" "app" {
  vpc_id = module.vpc.vpc_id
}
```

### Writing a module

```
modules/app/
├── main.tf
├── variables.tf
└── outputs.tf
```

```hcl
# modules/app/variables.tf
variable "vpc_id"     { type = string }
variable "subnet_ids" { type = list(string) }
variable "environment"{ type = string }

# modules/app/outputs.tf
output "sg_id" {
  value = aws_security_group.app.id
}
```

```bash
# After adding or changing a module source
terraform init
terraform get   # download/update modules only
```

---

## 6. Workspaces

Workspaces let you manage multiple environments (dev/staging/prod) with one config.

```bash
# List workspaces
terraform workspace list

# Create and switch to a new workspace
terraform workspace new staging
terraform workspace new production

# Switch workspace
terraform workspace select production

# Show current workspace
terraform workspace show

# Delete a workspace (must not be active)
terraform workspace delete staging
```

Use the workspace name in resources:

```hcl
locals {
  env = terraform.workspace  # "default", "staging", "production"

  instance_counts = {
    default    = 1
    staging    = 2
    production = 5
  }
}

resource "aws_instance" "app" {
  count         = local.instance_counts[local.env]
  instance_type = local.env == "production" ? "m5.large" : "t3.micro"

  tags = {
    Environment = local.env
  }
}
```

---

## 7. Remote Backends

Store state remotely so teams can collaborate safely.

### S3 + DynamoDB (AWS)

```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state-bucket"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-locks"  # prevents concurrent applies
  }
}
```

```bash
# Create the S3 bucket and DynamoDB table first (bootstrap)
aws s3api create-bucket --bucket my-terraform-state-bucket --region us-east-1
aws dynamodb create-table \
  --table-name terraform-state-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

### Terraform Cloud / HCP Terraform (free tier)

```hcl
terraform {
  cloud {
    organization = "my-org"
    workspaces {
      name = "my-project-prod"
    }
  }
}
```

```bash
terraform login   # authenticate with Terraform Cloud
terraform init
```

---

## 8. Expressions & Functions

```hcl
locals {
  # Conditional
  instance_type = var.environment == "production" ? "m5.large" : "t3.micro"

  # String interpolation
  name_prefix = "${var.environment}-${var.project}"

  # List and map operations
  all_cidrs   = concat(var.private_cidrs, var.public_cidrs)
  tag_map     = merge(var.common_tags, { Environment = var.environment })

  # For expression (transform a list)
  uppercase_names = [for name in var.names : upper(name)]

  # For expression (filter)
  prod_instances = [for i in var.instances : i if i.env == "production"]

  # For expression (map to map)
  instance_map = { for inst in var.instances : inst.name => inst.id }
}

# Common built-in functions
locals {
  # String
  lower_env   = lower(var.environment)          # "production" -> "production"
  trimmed     = trimspace("  hello  ")          # "hello"
  replaced    = replace(var.name, "-", "_")     # "my-app" -> "my_app"

  # Collections
  length      = length(var.subnet_ids)          # count items
  first       = element(var.subnet_ids, 0)      # first element
  flattened   = flatten([[1,2],[3,4]])           # [1,2,3,4]
  unique_list = distinct(var.possibly_duped)    # remove duplicates

  # Encoding
  b64         = base64encode("hello")
  json_str    = jsonencode({ key = "value" })
  parsed      = jsondecode(data.http.api.body)

  # File
  user_data   = file("${path.module}/scripts/init.sh")
  tpl_result  = templatefile("${path.module}/tpl/config.tpl", {
    host = var.db_host
    port = var.db_port
  })
}
```

---

## 9. Data Sources

Read existing infrastructure without managing it.

```hcl
# Look up an existing VPC by tag
data "aws_vpc" "existing" {
  tags = {
    Name = "production-vpc"
  }
}

# Look up the latest Amazon Linux 2023 AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

# Look up availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# Use data sources in resources
resource "aws_instance" "app" {
  ami               = data.aws_ami.amazon_linux.id
  subnet_id         = data.aws_vpc.existing.main_route_table_id
  availability_zone = data.aws_availability_zones.available.names[0]
}
```

---

## 10. Troubleshooting

### Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Error: No valid credential sources` | Provider not authenticated | Run `aws configure` / `gcloud auth` |
| `Error acquiring the state lock` | Concurrent apply or crashed run | `terraform force-unlock <lock-id>` |
| `Error: Resource already exists` | Resource in cloud but not in state | `terraform import` the resource |
| `Error: Invalid provider configuration` | Wrong region / missing required field | Check `providers.tf` |
| `Error: Cycle detected` | Circular resource dependency | Break cycle with `depends_on` or refactor |
| `Provider produced inconsistent result` | Provider bug or race condition | Re-run `terraform apply` |

### Debugging

```bash
# Verbose logging (levels: TRACE, DEBUG, INFO, WARN, ERROR)
TF_LOG=DEBUG terraform plan 2>debug.log

# Log to file
TF_LOG=DEBUG TF_LOG_PATH=./terraform.log terraform apply

# Show exactly what API calls are being made
TF_LOG=TRACE terraform plan 2>&1 | grep -i "request\|response" | head -50

# Refresh state from real infrastructure (re-reads all resources)
terraform refresh

# Validate without network calls
terraform validate
```

### Drift detection

```bash
# Check if real infrastructure matches state
terraform plan -refresh-only

# Apply the refresh (update state to match reality, no resource changes)
terraform apply -refresh-only
```

---

## Contributing

Skill authored by **dogiladeveloper**.

- GitHub: [github.com/dogiladeveloper](https://github.com/dogiladeveloper)
- Discord: `dogiladeveloper`
- Twitter/X: [@dogiladeveloper](https://twitter.com/dogiladeveloper)

Issues, improvements, and pull requests are welcome!
