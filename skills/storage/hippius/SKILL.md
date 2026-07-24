---
name: hippius-storage
description: Decentralized file storage on Bittensor Subnet 75 via Hippius S3-compatible API
version: 1.0.0
author: het4rk
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Storage, Decentralized, Bittensor, S3, IPFS, Hippius]
    requires_toolsets: [terminal]
required_environment_variables:
  - name: HIPPIUS_S3_ACCESS_KEY
    prompt: "Hippius S3 Access Key (starts with hip_)"
    help: "Get your access key at https://console.hippius.com/dashboard/settings"
    required_for: "S3-compatible storage operations"
  - name: HIPPIUS_S3_SECRET_KEY
    prompt: "Hippius S3 Secret Key"
    help: "Get your secret key at https://console.hippius.com/dashboard/settings"
    required_for: "S3-compatible storage operations"
---

# Hippius Decentralized Storage

Hippius is Bittensor Subnet 75's decentralized storage layer, powered by the **Arion** mesh network. It exposes an S3-compatible API endpoint at `https://s3.hippius.com`, meaning any tool that works with AWS S3 (boto3, the AWS CLI, rclone, s3cmd) works with Hippius without modification — just swap the endpoint and credentials.

Files are distributed across 400+ miners in 15+ countries using Reed-Solomon erasure coding with self-healing replication, providing durable, censorship-resistant storage with no single point of failure.

## When to Use

Invoke this skill when the user wants to:
- Upload files or directories to decentralized storage
- Download files from Hippius buckets
- List buckets and objects
- Share files via time-limited presigned URLs
- Sync a local directory to/from the Hippius network
- Create or delete storage buckets
- Back up data to censorship-resistant, geo-distributed storage
- Query storage usage statistics
- Replace or supplement centralized cloud storage (S3, GCS, Azure Blob)

## Quick Reference

| Operation | boto3 method | AWS CLI equivalent |
|---|---|---|
| List buckets | `s3.list_buckets()` | `aws s3 ls` |
| List objects | `s3.list_objects_v2(Bucket=…)` | `aws s3 ls s3://bucket/` |
| Upload file | `s3.upload_file(…)` | `aws s3 cp file s3://bucket/key` |
| Download file | `s3.download_file(…)` | `aws s3 cp s3://bucket/key file` |
| Create bucket | `s3.create_bucket(Bucket=…)` | `aws s3 mb s3://bucket` |
| Delete object | `s3.delete_object(…)` | `aws s3 rm s3://bucket/key` |
| Presigned URL | `s3.generate_presigned_url(…)` | `aws s3 presign s3://bucket/key` |
| Sync directory | n/a | `aws s3 sync ./dir s3://bucket/prefix/` |

---

## Setup

### Install dependencies

```bash
pip install boto3
# or for AWS CLI:
brew install awscli   # macOS
# apt install awscli  # Debian/Ubuntu
```

### Configure environment variables

```bash
export HIPPIUS_S3_ACCESS_KEY="hip_xxxxxxxxxxxxxxxxxxxx"
export HIPPIUS_S3_SECRET_KEY="your-secret-key"
```

Add to `~/.zshrc` or `~/.bashrc` to persist across sessions.

### boto3 client (reusable snippet)

```python
import boto3, os

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.hippius.com",
    aws_access_key_id=os.environ["HIPPIUS_S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["HIPPIUS_S3_SECRET_KEY"],
    region_name="decentralized",
)
```

### AWS CLI profile

```bash
aws configure --profile hippius
# AWS Access Key ID:     hip_xxxxxxxxxxxxxxxxxxxx
# AWS Secret Access Key: your-secret-key
# Default region name:   decentralized
# Default output format: json
```

Then pass `--profile hippius --endpoint-url https://s3.hippius.com` to every `aws s3` command, or set:

```bash
export AWS_PROFILE=hippius
export AWS_ENDPOINT_URL=https://s3.hippius.com
```

---

## Procedures

### Upload a file

**boto3:**
```python
s3.upload_file(
    Filename="/path/to/local/file.txt",
    Bucket="my-bucket",
    Key="remote/path/file.txt",
)
print("Upload complete")
```

**AWS CLI:**
```bash
aws s3 cp /path/to/local/file.txt s3://my-bucket/remote/path/file.txt \
    --endpoint-url https://s3.hippius.com --profile hippius
```

For files larger than 100 MB, multipart upload is triggered automatically. The minimum part size is **10 MB** — do not set `multipart_chunksize` below this value.

```python
from boto3.s3.transfer import TransferConfig

config = TransferConfig(multipart_chunksize=10 * 1024 * 1024)  # 10 MB minimum
s3.upload_file("large_file.bin", "my-bucket", "large_file.bin", Config=config)
```

---

### Download a file

**boto3:**
```python
s3.download_file(
    Bucket="my-bucket",
    Key="remote/path/file.txt",
    Filename="/path/to/local/output.txt",
)
print("Download complete")
```

**AWS CLI:**
```bash
aws s3 cp s3://my-bucket/remote/path/file.txt /path/to/local/output.txt \
    --endpoint-url https://s3.hippius.com --profile hippius
```

---

### List buckets

**boto3:**
```python
response = s3.list_buckets()
for bucket in response.get("Buckets", []):
    print(bucket["Name"], bucket["CreationDate"])
```

**AWS CLI:**
```bash
aws s3 ls --endpoint-url https://s3.hippius.com --profile hippius
```

---

### List objects in a bucket

**boto3** (handles pagination automatically):
```python
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket="my-bucket", Prefix="optional/prefix/"):
    for obj in page.get("Contents", []):
        print(obj["Key"], obj["Size"], obj["LastModified"])
```

**AWS CLI:**
```bash
aws s3 ls s3://my-bucket/optional/prefix/ --recursive \
    --endpoint-url https://s3.hippius.com --profile hippius
```

---

### Create a bucket

**boto3:**
```python
s3.create_bucket(Bucket="my-new-bucket")
print("Bucket created")
```

**AWS CLI:**
```bash
aws s3 mb s3://my-new-bucket \
    --endpoint-url https://s3.hippius.com --profile hippius
```

Bucket names must be globally unique, lowercase, 3–63 characters, and contain only letters, numbers, and hyphens.

---

### Generate a presigned URL

Share a file without exposing credentials. The URL expires after the specified duration.

**boto3:**
```python
url = s3.generate_presigned_url(
    ClientMethod="get_object",
    Params={"Bucket": "my-bucket", "Key": "file.txt"},
    ExpiresIn=3600,  # seconds (1 hour)
)
print(url)
```

**AWS CLI:**
```bash
aws s3 presign s3://my-bucket/file.txt --expires-in 3600 \
    --endpoint-url https://s3.hippius.com --profile hippius
```

---

### Sync a directory

Mirror a local directory to Hippius (only uploads changed/new files):

```bash
aws s3 sync ./local-dir/ s3://my-bucket/backup/ \
    --endpoint-url https://s3.hippius.com --profile hippius

# Reverse: download from Hippius to local
aws s3 sync s3://my-bucket/backup/ ./local-dir/ \
    --endpoint-url https://s3.hippius.com --profile hippius

# Delete remote files that no longer exist locally
aws s3 sync ./local-dir/ s3://my-bucket/backup/ --delete \
    --endpoint-url https://s3.hippius.com --profile hippius
```

---

### Query storage stats (helper script)

Run the bundled helper to get a usage overview:

```bash
python3 skills/storage/hippius/scripts/query_storage.py
# With a specific bucket:
python3 skills/storage/hippius/scripts/query_storage.py --bucket my-bucket
```

---

## Pitfalls

| Issue | Detail |
|---|---|
| **Rate limit** | 100 requests/minute per access key. Use pagination and avoid tight loops over the API. |
| **Multipart minimum part size** | Parts must be at least **10 MB** (except the final part). Setting `multipart_chunksize` below 10 MB will fail. |
| **No versioning** | Hippius does not support S3 bucket versioning. Overwrites are permanent. |
| **Region name** | Always set `region_name="decentralized"`. Some boto3 calls fail with a blank or incorrect region. |
| **Bucket name uniqueness** | Bucket names are global across all Hippius users, not scoped to your account. |
| **Large file timeouts** | For very large uploads (>1 GB) on slow connections, increase the boto3 socket timeout via `boto3.session.Config(connect_timeout=60, read_timeout=300)`. |

---

## Verification

After completing an operation, verify success:

```bash
# Confirm upload exists
aws s3 ls s3://my-bucket/remote/path/file.txt \
    --endpoint-url https://s3.hippius.com --profile hippius

# Check object metadata
aws s3api head-object --bucket my-bucket --key remote/path/file.txt \
    --endpoint-url https://s3.hippius.com --profile hippius

# Confirm download integrity (compare checksums)
md5sum /path/to/local/file.txt
# should match the ETag returned by head-object for single-part uploads
```

## References

- Storage guide and architecture overview: `skills/storage/hippius/references/storage_guide.md`
- Helper script: `skills/storage/hippius/scripts/query_storage.py`
- Hippius console: https://console.hippius.com/dashboard/settings
