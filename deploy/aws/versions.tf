terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

# Region comes from the caller. This module never pins one: a customer's data-residency
# answer is theirs to give, and a default here would be a default in the wrong place.
provider "aws" {
  region = var.region

  default_tags {
    tags = merge(
      {
        "nova:tenant"     = var.tenant_id
        "nova:managed-by" = "terraform"
      },
      var.tags,
    )
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
data "aws_region" "current" {}
