# CSMS Usage Guide

## Create Secret

hcloud CSMS CreateSecret --secret_name=prod-db-password --secret_string='{"password":"CHANGE_ME"}'

## List Secrets (metadata only)

hcloud CSMS ListSecrets

## Describe Secret

hcloud CSMS DescribeSecret --secret_name=prod-db-password

## List Versions

hcloud CSMS ListSecretVersions --secret_name=prod-db-password

## Runtime Injection (Terraform)

data "huaweicloud_csms_secret" "db" {
secret_name = "prod-db-password"
}
Use: data.huaweicloud_csms_secret.db.secret_string

## Rotation

- Enable auto-rotation: hcloud CSMS EnableSecretRotation --secret_name=<name> --rotation_interval=30
- Rotation function ARN: provide Lambda to generate new secret value

## Policy Rules

- hcloud CSMS DownloadSecret -> BLOCKED (use runtime injection)
- hcloud CSMS ShowSecretVersion -> BLOCKED
- hcloud CSMS ListSecrets -> ALLOWED (metadata only)
- hcloud CSMS DescribeSecret -> ALLOWED (metadata only)
