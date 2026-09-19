# OBS Single-File Quick Share

Host one file and get a shareable link in seconds. Two options:

| Option        | Link type                 | Bucket/object visibility      | Expiry                     |
| ------------- | ------------------------- | ----------------------------- | -------------------------- |
| Presigned URL | Time-limited private link | Keep private                  | `-e=<seconds>`, max 7 days |
| Public URL    | Permanent public link     | Set object `-acl=public-read` | None                       |

## Option 1: Presigned URL (private, time-limited)

No ACL change needed. The object stays private; the link grants temporary access.

```bash
hcloud OBS cp <file> obs://<bucket>/<key> -f
hcloud OBS sign obs://<bucket>/<key> -e=3600   # valid 1 hour, max 604800 (7 days)
```

The `sign` command prints the full presigned URL directly — share it as-is.

## Option 2: Public URL (permanent, public-read)

```bash
hcloud OBS cp <file> obs://<bucket>/<key> -f
hcloud OBS chattri obs://<bucket>/<key> -acl=public-read
```

The shareable public URL follows the OBS endpoint format (NOT the `obs-website` static-site endpoint):

```
https://<bucket>.obs.<region>.myhuaweicloud.com/<key>
```

## Key Gotchas

- **Bucket ACL does NOT cascade**: `chattri obs://<bucket> -acl=public-read` alone does not make objects public. Set the object-level ACL explicitly.
- **`-f` is mandatory**: `cp` without `-f` prompts "Please input (y/n)" on overwrite → agent hangs (TIMEOUT).
- **Public-read means anyone with the link can access**: use presigned URLs when the file is sensitive.
- **Presigned URL max expiry is 7 days** (`-e=604800`). For longer-lived public access, use the public-read option.
- **Credential setup**: OBS uses a separate config (`~/.obsutilconfig`). Call `huaweicloud_setup_obs_config` before first use.
