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
