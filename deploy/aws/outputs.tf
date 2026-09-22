output "runtime_role_arn" {
  description = "The role the runtime runs as. Carries no customer-data permissions."
  value       = aws_iam_role.runtime.arn
}

output "integration_role_arns" {
  description = "Every customer system the agents can reach, by integration id. If this map is empty, they can reach none."
  value       = { for id, role in aws_iam_role.integration : id => role.arn }
}

output "integration_external_id" {
  description = "External id the runtime must present when assuming an integration role."
  value       = local.external_id
}

output "instance_id" {
  description = "Reach it with: aws ssm start-session --target <this>"
  value       = aws_instance.runtime.id
}

output "log_group_name" {
  description = "CloudWatch Logs group carrying the runtime's operational logs."
  value       = aws_cloudwatch_log_group.runtime.name
}

output "kms_key_arn" {
  description = "Key protecting the state volume, the logs and the secrets."
  value       = local.kms_key_arn
}

output "state_volume_id" {
  description = "The volume holding NOVA's state, including the audit log."
  value       = aws_ebs_volume.state.id
}

output "bootstrap_sha256" {
  description = <<-DESC
    Hash of the bootstrap this configuration renders. The instance ignores user_data changes
    (see the lifecycle block in main.tf: on a booted host cloud-init will not re-run the
    script, so the only way to apply a new one is to replace the instance), which means
    Terraform no longer reports the difference by itself. This is where it shows instead.

    What the host actually ran:
      aws ssm send-command --instance-ids "$(terraform output -raw instance_id)" \
        --document-name AWS-RunShellScript \
        --parameters 'commands=["sha256sum /var/lib/cloud/instance/user-data.txt"]'

    A mismatch means the running instance predates this module's bootstrap. Rolling it
    forward is `terraform apply -replace=aws_instance.runtime`, which keeps the state volume
    (prevent_destroy, and a replacement detaches it rather than destroying it) and reboots
    into the current script. A hash, not the script itself, because the bootstrap carries the
    integration external id and plan output ends up in CI logs.
  DESC
  value       = sha256(local.bootstrap)
}
