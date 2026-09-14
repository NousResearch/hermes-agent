# Service Limits

## Rate limits

The standard plan allows 1,000 API requests per minute per organisation.
Bursts up to 2,000 are absorbed for 30 seconds before requests are rejected
with HTTP 429. The enterprise plan raises the sustained limit to 10,000.

## Retention

Event data is retained for 90 days on the standard plan and 400 days on the
enterprise plan. Deleted records are purged from backups within 35 days.

## Supported regions

Data can be pinned to eu-west-1, us-east-1 or ap-southeast-2. Pinning is set
at organisation creation and cannot be changed afterwards without a migration
arranged through support.
