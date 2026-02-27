---
name: terraform-iac
description: Manage infrastructure as code with Terraform. Create, plan, apply, and destroy resources across AWS, Azure, GCP, and other providers. Handle state management and workspace operations.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [Terraform, IaC, Infrastructure, AWS, Azure, GCP, DevOps]
    related_skills: [kubernetes-management, cloud-clis]
---

# Terraform Infrastructure as Code

Manage cloud infrastructure using Terraform. Works with AWS, Azure, GCP, and 100+ other providers.

## Prerequisites

- Terraform installed (`command -v terraform`)
- Provider credentials configured (AWS profile, service principal, etc.)

### Setup

```bash
# Verify installation
terraform version

# Initialize working directory
terraform init

# Validate configuration
terraform validate
```

---

## 1. Core Workflow

### Initialize

```bash
# Basic init
terraform init

# Upgrade providers
terraform init -upgrade

# Reconfigure (migrate backends)
terraform init -reconfigure

# Migrate state
terraform init -migrate-state
```

### Plan and Apply

```bash
# Generate execution plan
terraform plan

# Save plan to file
terraform plan -out=tfplan

# Apply changes
terraform apply

# Apply saved plan
terraform apply tfplan

# Auto-approve (use with caution)
terraform apply -auto-approve

# Apply specific target
terraform apply -target=aws_instance.web

# Apply with variables
terraform apply -var="instance_type=t3.large" -var="region=us-west-2"

# Apply with var file
terraform apply -var-file=production.tfvars
```

### Destroy

```bash
# Preview destroy
terraform plan -destroy

# Destroy all resources
terraform destroy

# Auto-approve destroy
terraform destroy -auto-approve

# Destroy specific resource
terraform destroy -target=aws_instance.web
```

---

## 2. State Management

```bash
# Show current state
terraform show

# List resources in state
terraform state list

# Show specific resource
terraform state show aws_instance.web

# Pull state to local
terraform state pull > state.json

# Push state from file
terraform state push state.json

# Move resource in state
terraform state mv aws_instance.old aws_instance.new

# Remove resource from state
terraform state rm aws_instance.unwanted

# Import existing resource
terraform import aws_instance.web i-1234567890abcdef0

# Refresh state
terraform refresh
```

---

## 3. Workspaces

```bash
# List workspaces
terraform workspace list

# Create workspace
terraform workspace new production

# Select workspace
terraform workspace select production

# Show current workspace
terraform workspace show

# Delete workspace
terraform workspace delete staging

# Force delete (with resources)
terraform workspace delete -force staging
```

---

## 4. Variables and Outputs

```bash
# Show outputs
terraform output

# Show specific output
terraform output instance_ip

# Show outputs as JSON
terraform output -json

# Show raw output value
terraform output -raw instance_ip
```

---

## 5. Formatting and Validation

```bash
# Format code
terraform fmt

# Format recursive
terraform fmt -recursive

# Check formatting (CI/CD)
terraform fmt -check

# Validate configuration
terraform validate

# Validate with variables
terraform validate -var-file=prod.tfvars
```

---

## 6. Module Operations

```bash
# Get modules
terraform get

# Update modules
terraform get -update

# Graph dependencies
terraform graph | dot -Tpng > graph.png
```

---

## 7. Best Practices

### File Structure
```
project/
├── main.tf          # Main resources
├── variables.tf     # Input variables
├── outputs.tf       # Output values
├── providers.tf     # Provider configuration
├── backend.tf       # Backend configuration
├── terraform.tfvars # Variable values
└── modules/         # Local modules
    └── vpc/
```

### Environment Separation
```bash
# Using workspaces
terraform workspace select dev
terraform apply

terraform workspace select prod
terraform apply

# Or using directories
environments/
├── dev/
│   ├── main.tf
│   └── terraform.tfvars
└── prod/
    ├── main.tf
    └── terraform.tfvars
```

---

## 8. Debugging

```bash
# Enable detailed logging
export TF_LOG=DEBUG
export TF_LOG_PATH=terraform.log

# Trace level
export TF_LOG=TRACE

# Disable logging
unset TF_LOG
unset TF_LOG_PATH
```
