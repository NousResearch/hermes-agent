# Provider Notes

## Bedrock

Hermes can use AWS Bedrock credentials for models available to the configured AWS account and region. The exact Bedrock `modelId` is the source of truth.

For OpenAI models on Bedrock, verify availability with AWS before configuring Hermes. Do not assume an OpenAI API model name is identical to the Bedrock model ID.

## GPT/OpenAI-Compatible Providers

If a desired GPT model is not exposed by Bedrock in the target region/account, configure a separate OpenAI-compatible provider rather than trying to use AWS credentials for it.

## Pilot Default

Prefer a lab profile with explicit provider/model settings and smart/manual approvals. Keep credentials in the Hermes profile or cloud IAM, never in this repo.
