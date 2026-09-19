# OBS Static Website Deployment

## Workflow

```
1. Build project     → npm run build / yarn build
2. Create OBS bucket  → hcloud OBS mb obs://<bucket> -location=<region>
3. Bulk upload        → hcloud OBS cp <build-dir>/ obs://<bucket>/ -r -f -flat -acl=public-read
4. Set bucket ACL     → hcloud OBS chattri obs://<bucket> -acl=public-read
5. Set object ACLs    → hcloud OBS chattri obs://<bucket>/ -r -f -acl=public-read
6. Configure website  → REST API (KooCLI OBS lacks this feature)
7. Verify             → curl http://<bucket>.obs-website.<region>.myhuaweicloud.com
```

## Step-by-Step (Vue/Vite Example)

```bash
# 1. Build
npm run build                  # outputs to dist/

# 2. Create bucket in target region
hcloud OBS mb obs://my-static-site -location=cn-north-4

# 3. Upload entire build directory (recursive, force, flatten, public ACL)
hcloud OBS cp dist/ obs://my-static-site/ -r -f -flat -acl=public-read

# 4. Set bucket ACL (required for static website access)
hcloud OBS chattri obs://my-static-site -acl=public-read

# 5. Set object ACLs recursively (objects don't inherit bucket ACL automatically)
hcloud OBS chattri obs://my-static-site/ -r -f -acl=public-read

# 5. Configure static website hosting
# KooCLI OBS does NOT support this. Use one of:
# Option A — REST API:
#   PUT /?website HTTP/1.1
#   Host: my-static-site.obs.cn-north-4.myhuaweicloud.com
#   Body: {"IndexDocument": {"Suffix": "index.html"}, "ErrorDocument": {"Key": "error.html"}}
#   (requires AK/SK signature)
# Option B — Huawei Cloud Console:
#   Console → OBS → Bucket → Basic Settings → Static Website Hosting

# 6. Verify
curl http://my-static-site.obs-website.cn-north-4.myhuaweicloud.com
```

## Key Gotchas

- **Build output dir varies**: `dist/` (Vite), `build/` (CRA), `out/` (Next.js). Check `package.json` scripts.
- **Bucket name must be DNS-compliant**: lowercase, numbers, hyphens only. No underscores or uppercase.
- **Region matters**: Website endpoint depends on bucket region. `obs-website.<region>.myhuaweicloud.com`.
- **Index.html routing**: For SPA apps (Vue Router, React Router), set `ErrorDocument` to `index.html` as well.
- **Cache invalidation**: New uploads don't clear CDN cache. Add `?v=<timestamp>` to asset references or use OBS versioning.
- **Always use `-f`**: Without `-f`, obsutil prompts "Please input (y/n)" on large uploads — Agent hangs (TIMEOUT).
- **Use `-flat` for root files**: Without `-flat`, `dist/` becomes `bucket/dist/` prefix. Static sites need files at bucket root.
- **Objects don't inherit bucket ACL**: Setting bucket to public-read does NOT make individual objects public. You must also set object-level ACL with `chattri obs://<bucket>/ -r -f -acl=public-read` or use `-acl=public-read` during upload.
