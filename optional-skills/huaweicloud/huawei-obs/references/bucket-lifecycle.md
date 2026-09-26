# OBS Bucket Lifecycle

> KooCLI OBS uses obsutil-style commands. `OBS CreateBucket` → `OBS mb`. `SetBucketWebsite` is NOT supported in KooCLI OBS (use API/SDK).

## Create Bucket

hcloud OBS mb obs://<bucket> -location=<region>

## Lifecycle Rules (JSON)

{
"Rules": [{
"ID": "move-to-ia-after-30d",
"Status": "Enabled",
"Filter": {"Prefix": ""},
"Transitions": [{"Days": 30, "StorageClass": "STANDARD_IA"}],
"Expiration": {"Days": 365}
}]
}

## Apply

hcloud OBS lifecycle obs://<bucket> -method=put -localfile=<lifecycle.json>

## Static Website

Not available via KooCLI OBS CLI. Use `hcloud OBS chattri obs://<bucket> -acl=public-read` for bucket ACL, or use API/SDK for website configuration.

## Cross-Region Replication

Requires: source bucket versioning enabled, destination bucket in different region, replication IAM role.
