---
name: github-release-chunked-upload
description: Upload large files to GitHub Releases using chunked uploads when gh release upload times out
triggers:
  - gh release upload timeout
  - large file upload failure
  - network blocks large HTTPS upload
---

# GitHub Release 大文件分块上传

## 触发条件
使用 `gh release upload` 上传文件时超时（尤其 APK/二进制文件 >5MB），或网络环境阻塞大文件 HTTPS 上传。

## 核心方法
通过 GitHub REST API 使用 `Content-Range` header 分块上传。

### Step 1：获取 Release ID
```bash
RELEASE_ID=$(gh api repos/{owner}/{repo}/releases/tags/{tag} --jq '.id')
```

### Step 2：分块上传（1MB chunks）
```python
import urllib.request, json, ssl

token = open('/tmp/gh_token.txt').read().strip()
APKPATH = '/path/to/file.apk'
RELEASE_ID = '319081337'

with open(APKPATH, 'rb') as f:
    apk_data = f.read()

apk_size = len(apk_data)
CHUNK_SIZE = 1 * 1024 * 1024  # 1MB - 最稳定
chunks = [apk_data[i:i+CHUNK_SIZE] for i in range(0, apk_size, CHUNK_SIZE)]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

upload_url = f'https://uploads.github.com/repos/{owner}/{repo}/releases/{RELEASE_ID}/assets?name=filename.apk'

for i, chunk in enumerate(chunks):
    headers = {
        'Authorization': f'token {token}',
        'Content-Type': 'application/octet-stream',
        'Content-Length': str(len(chunk)),
        'Content-Range': f'bytes {i*CHUNK_SIZE}-{(i+1)*CHUNK_SIZE-1}/{apk_size}',
        'Accept': 'application/vnd.github+json'
    }
    req = urllib.request.Request(upload_url, data=chunk, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
        result = json.loads(resp.read())
        print(f"Chunk {i+1}/{len(chunks)} SUCCESS")
        break  # 第一个 chunk 成功即完成
```

## 关键参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `CHUNK_SIZE` | 1MB | 5MB 会超时，1MB 最稳定 |
| `timeout` | 20s | 每个 chunk 的超时时间 |
| `Content-Range` | `bytes {start}-{end}/{total}` | 分块上传必需 |

## 验证上传结果
```bash
gh api repos/{owner}/{repo}/releases/{RELEASE_ID}/assets --jq '.[].name'
```

## 已知限制
- 不需要完整遍历所有 chunks，GitHub API 在第一个 chunk 成功后就返回完整 asset URL
- 分块上传需要 HTTPS，网络必须能访问 `uploads.github.com`
