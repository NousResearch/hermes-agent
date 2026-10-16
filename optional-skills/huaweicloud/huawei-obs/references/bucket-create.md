# OBS Bucket Creation Reference

From Huawei Cloud marketplace best practices.

## Naming Rules

| Rule                           | Detail                               |
| ------------------------------ | ------------------------------------ |
| Length                         | 3-63 characters                      |
| Characters                     | Lowercase letters, numbers, `-`, `.` |
| No IP format                   | Cannot be IP address pattern         |
| No leading/trailing `-` or `.` | Start and end must be alphanumeric   |
| No `..`                        | Adjacent dots forbidden              |
| No `-.` or `.-`                | Adjacent dot-hyphen combos forbidden |

## Create Bucket

```bash
hcloud OBS mb obs://<bucket> [-acl=xxx] [-location=<r>] [-sc=xxx]
```

| Option      | Values                                  | Default        |
| ----------- | --------------------------------------- | -------------- |
| `-acl`      | private, public-read, public-read-write | private        |
| `-location` | cn-south-1, cn-north-4, ...             | config default |
| `-sc`       | standard, warm, cold, deep-archive      | standard       |

## Quick Verify

```bash
hcloud OBS ls
hcloud OBS stat obs://<bucket>
```

## Storage Class Guide

| Class        | Access Frequency | Min Storage | Retrieval Cost |
| ------------ | ---------------- | ----------- | -------------- |
| standard     | Frequent         | None        | No             |
| warm         | Infrequent       | 30 days     | Yes            |
| cold         | Rare             | 90 days     | Yes            |
| deep-archive | Archive          | 180 days    | Yes            |

## Critical Rules

- Bucket name is globally unique — test with `hcloud OBS stat` before creating
- Region is immutable after creation
- ACL does NOT cascade to objects — set both bucket-level and object-level access
- OBS uses separate credentials (`~/.obsutilconfig`, configure via `hcloud OBS config -i`)
