# Hippius Storage Reference Guide

## What is Hippius?

Hippius is the decentralized storage layer of **Bittensor Subnet 75**, operated under the **Arion** mesh network architecture. It provides durable, geo-distributed object storage accessible through a standard S3-compatible API — no proprietary SDK required.

Key facts:
- **Endpoint:** `https://s3.hippius.com`
- **Region name (required):** `decentralized`
- **Rate limit:** 100 requests/minute per access key
- **Minimum multipart part size:** 10 MB
- **Versioning:** not supported (overwrites are permanent)
- **Compatible tools:** boto3, AWS CLI, rclone, s3cmd, MinIO client

---

## Arion Architecture Overview

Arion is the Bittensor subnet validator/miner protocol that powers Hippius storage. Understanding it helps reason about durability and latency.

```
Client
  │
  ▼
Hippius S3 Gateway  (https://s3.hippius.com)
  │  Translates S3 API calls to Arion internal protocol
  ▼
Arion Coordinator
  │  Splits objects into shards, applies erasure coding
  ├──► Miner Node A  (e.g. US-East)
  ├──► Miner Node B  (e.g. EU-West)
  ├──► Miner Node C  (e.g. Asia-Pacific)
  └──► Miner Node … (400+ nodes across 15+ countries)
```

### Erasure coding

Hippius uses **Reed-Solomon erasure coding** to distribute file shards across multiple miners. A file can be reconstructed even if a fraction of miner nodes are offline. This means:
- No single point of failure
- Data survives individual node churn (common in decentralized networks)
- Reads may be served from the geographically nearest available shard

### Self-healing

The Arion coordinator continuously monitors shard availability. If a miner goes offline and drops below the minimum redundancy threshold, the coordinator re-replicates affected shards to healthy miners automatically.

### Bittensor incentives

Miners earn TAO (Bittensor's native token) for reliably serving storage capacity. Validators score miners based on availability and retrieval latency, creating economic pressure for high uptime.

---

## S3 vs IPFS: Choosing the Right Interface

Hippius supports two access patterns. Use S3 for most automation; use IPFS when content-addressing or public sharing is the priority.

| Aspect | S3-compatible (Hippius) | IPFS |
|---|---|---|
| **Addressing** | Mutable path: `s3://bucket/key` | Immutable content hash: `ipfs://QmXyz…` |
| **Access control** | Bucket policies, presigned URLs | Public by default (hash = address) |
| **Mutability** | Yes — overwrite keys freely | No — changing content changes the CID |
| **Tooling** | boto3, AWS CLI, rclone, s3cmd | IPFS CLI, Kubo, web3.storage clients |
| **Latency** | Low for cached/nearby shards | Variable (DHT lookup required) |
| **Best for** | Application data, backups, private files | Content distribution, NFT metadata, public archives |
| **Auth required** | Yes (access key + secret key) | No (public gateway) |
| **Hippius integration** | Native (primary interface) | Via IPFS pinning API |

### When to use S3 (most cases)
- Automated backups and archiving
- Application file storage (user uploads, ML datasets, model weights)
- Private or access-controlled data
- Anything that needs mutable keys (overwrite-in-place semantics)
- Integrating with existing S3-compatible tooling

### When to use IPFS
- Publishing immutable public content (documentation, media, datasets)
- NFT metadata or on-chain content references
- Content you want to verify by hash (tamper-evident archives)
- Interoperability with the broader IPFS ecosystem

---

## Comparison: Hippius vs Centralised Cloud Storage

| Feature | Hippius (Bittensor SN75) | AWS S3 | Cloudflare R2 |
|---|---|---|---|
| **Infrastructure** | 400+ decentralized miners | AWS data centers | Cloudflare edge PoPs |
| **Censorship resistance** | High (no single operator) | Low | Medium |
| **Geo-distribution** | 15+ countries, automatic | Requires multi-region config | Global CDN |
| **Egress fees** | None | ~$0.09/GB | None |
| **API compatibility** | S3-compatible | Native S3 | S3-compatible |
| **Versioning** | No | Yes | No |
| **Uptime SLA** | Network-level (no formal SLA) | 99.99% | 99.9% |
| **Best for** | Decentralized apps, privacy-sensitive data | General cloud workloads | Static assets, low-egress workloads |

---

## Common Patterns

### Backup script (cron-friendly)

```bash
#!/bin/bash
set -e
DATE=$(date +%Y-%m-%d)
BACKUP_DIR="/var/backups/myapp"
BUCKET="myapp-backups"

tar czf /tmp/backup-${DATE}.tar.gz "${BACKUP_DIR}"
aws s3 cp /tmp/backup-${DATE}.tar.gz "s3://${BUCKET}/daily/backup-${DATE}.tar.gz" \
    --endpoint-url https://s3.hippius.com --profile hippius
rm /tmp/backup-${DATE}.tar.gz
echo "Backup uploaded: s3://${BUCKET}/daily/backup-${DATE}.tar.gz"
```

### ML dataset upload

```python
import boto3, os

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.hippius.com",
    aws_access_key_id=os.environ["HIPPIUS_S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["HIPPIUS_S3_SECRET_KEY"],
    region_name="decentralized",
)

# Upload an entire dataset directory
import subprocess
subprocess.run([
    "aws", "s3", "sync", "./dataset/", "s3://ml-datasets/my-dataset/",
    "--endpoint-url", "https://s3.hippius.com",
    "--profile", "hippius",
], check=True)
```

### Presigned URL for secure sharing

```python
url = s3.generate_presigned_url(
    ClientMethod="get_object",
    Params={"Bucket": "my-bucket", "Key": "reports/q4-2025.pdf"},
    ExpiresIn=86400,  # 24 hours
)
# Share this URL — it works without credentials, expires automatically
print(f"Share link (valid 24h): {url}")
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `InvalidAccessKeyId` | Wrong or missing `HIPPIUS_S3_ACCESS_KEY` | Check env var, ensure key starts with `hip_` |
| `SignatureDoesNotMatch` | Wrong secret key | Re-check `HIPPIUS_S3_SECRET_KEY` |
| `NoSuchBucket` | Bucket doesn't exist or wrong name | Run `aws s3 ls` to list your buckets |
| `EntityTooSmall` on multipart | Part size below 10 MB | Set `multipart_chunksize=10*1024*1024` |
| `SlowDown` / 503 | Rate limit exceeded (100 req/min) | Add retry logic with exponential backoff |
| Timeout on large upload | Connection or read timeout | Increase boto3 `read_timeout` to 300+ seconds |
| boto3 `Could not connect to endpoint` | Wrong endpoint URL | Confirm `endpoint_url="https://s3.hippius.com"` |
