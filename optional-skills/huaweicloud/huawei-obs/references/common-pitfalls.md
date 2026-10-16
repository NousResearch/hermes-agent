# OBS Common Pitfalls

基于真实测试暴露的陷阱和避坑方法。

## OBS 命令格式陷阱

OBS 使用 **obsutil-style** 子命令，不是 API-style 操作名：

| 错误写法（API-style） | 正确写法（obsutil-style）                     |
| --------------------- | --------------------------------------------- |
| `OBS --help`          | `hcloud OBS help`（无 `--`，`help` 是子命令） |
| `OBS CreateBucket`    | `hcloud OBS mb obs://<bucket>`                |
| `OBS PutObject`       | `hcloud OBS cp <file> obs://<bucket>/`        |
| `OBS DeleteBucket`    | `hcloud OBS rm obs://<bucket> -r`             |
| `OBS ListBuckets`     | `hcloud OBS ls`                               |

## 独立凭证配置

KooCLI OBS 使用独立的凭证文件 `~/.obsutilconfig`，**不是** `~/.hcloud/config.json`。

```bash
# OBS 命令报 "Please set ak, sk and endpoint" 时运行：
hcloud OBS config -i
# 交互式输入：AK、SK、endpoint（如 obs.cn-north-4.myhuaweicloud.com）
# 这必须由用户在 agent 对话外手动执行
```

## 目录上传语义

`hcloud OBS cp <dir>/ obs://bucket/ -r` 将本地**文件夹名作为对象前缀**：

```
本地结构:
  devkit/
    index.html
    assets/app.js

上传后 OBS 结构:
  obs://bucket/devkit/index.html     ← 多了 devkit/ 前缀
  obs://bucket/devkit/assets/app.js

预期结构:
  obs://bucket/index.html
  obs://bucket/assets/app.js
```

**规避方法**：

1. 先 `-dryRun` 预览实际键名，再执行真实上传
2. 或先将本地目录重命名为目标前缀名，如 `mv devkit/ root/` 后上传 `root/`
3. 或使用 `-flat` 参数（如支持）

## 静态网站托管

KooCLI OBS **不支持** `SetBucketWebsite` 命令。配置静态网站托管需要：

1. **REST API**: `PUT /?website` 到桶的 endpoint
2. **控制台**: OBS 控制台 → 桶 → 静态网站托管 → 开启

插件只能完成"上传文件 + 设置 ACL"，静态托管开关需用户手动完成。

## 权限模型（三层）

| 层级          | 作用域      | 优先级 |
| ------------- | ----------- | ------ |
| IAM Policy    | 用户/用户组 | 最高   |
| Bucket Policy | 桶级        | 中     |
| ACL           | 对象级      | 最低   |

**关键规则**：三层中**最严格**的生效。Bucket Policy 设为 public-read，但 IAM deny 了 → 结果是 deny。

**静态网站场景**：必须同时设置桶级 AND 对象级 `-acl=public-read`，因为 ACL 不级联。

```bash
# 桶级
hcloud OBS chattri obs://<bucket> -acl=public-read
# 对象级（必须单独执行）
hcloud OBS chattri obs://<bucket>/index.html -acl=public-read
```

## 桶命名约束

- 全局唯一，所有用户共享命名空间
- 小写字母、数字、连字符
- 3-63 字符
- 不能以连字符开头或结尾
- 不能是 IP 地址格式

## 版本控制陷阱

- 开启后**无法关闭**，只能暂停
- 删除桶前必须清空所有版本和删除标记
- 版本存储按对象大小计费

## 安全注意事项

- MUST 默认阻止公网访问
- MUST 启用 HTTPS-only
- SHOULD 开启访问日志
- 预签名 URL 最大 7 天有效期
- AK/SK 绝对不能写入 Bucket Policy
