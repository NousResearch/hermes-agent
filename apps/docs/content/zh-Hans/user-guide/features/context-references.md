---
sidebar_position: 9
sidebar_label: "Context References"
title: "Context References"
description: "用于将文件、文件夹、git diff 及 URL 直接附加到消息中的内联 @-语法"
---

# Context References

输入 `@` 后跟一个引用，即可将内容直接注入消息。Hermes 会将引用内联展开，并在 `--- Attached Context ---` 区块下追加相应内容。

## 支持的引用类型

| 语法 | 说明 |
|--------|-------------|
| `@file:path/to/file.py` | 注入文件内容 |
| `@file:path/to/file.py:10-25` | 注入指定行范围（从 1 开始，含首尾） |
| `@folder:path/to/dir` | 注入目录树列表及文件元数据 |
| `@diff` | 注入 `git diff`（未暂存的工作区变更） |
| `@staged` | 注入 `git diff --staged`（已暂存的变更） |
| `@git:5` | 注入最近 N 次提交及补丁（最多 10 次） |
| `@url:https://example.com` | 抓取并注入网页内容 |

## 使用示例

```text
Review @file:src/main.py and suggest improvements

What changed? @diff

Compare @file:old_config.yaml and @file:new_config.yaml

What's in @folder:src/components?

Summarize this article @url:https://arxiv.org/abs/2301.00001
```

单条消息中可使用多个引用：

```text
Check @file:main.py, and also @file:test.py.
```

引用值末尾的标点符号（`,`、`.`、`;`、`!`、`?`）会被自动去除。

## CLI Tab 补全

在交互式 CLI 中，输入 `@` 会触发自动补全：

- `@` 显示所有引用类型（`@diff`、`@staged`、`@file:`、`@folder:`、`@git:`、`@url:`）
- `@file:` 和 `@folder:` 触发文件系统路径补全，并显示文件大小元数据
- 裸 `@` 后跟部分文本时，显示当前目录中匹配的文件和文件夹

## 行范围

`@file:` 引用支持行范围，用于精确注入内容：

```text
@file:src/main.py:42        # 单行第 42 行
@file:src/main.py:10-25     # 第 10 至 25 行（含首尾）
```

行号从 1 开始。无效范围会被静默忽略（返回完整文件）。

## 大小限制

Context references 受大小限制，以防止超出模型的 context window（上下文窗口）：

| 阈值 | 值 | 行为 |
|-----------|-------|----------|
| 软限制 | 上下文长度的 25% | 追加警告，继续展开 |
| 硬限制 | 上下文长度的 50% | 拒绝展开，返回原始消息不变 |
| 文件夹条目 | 最多 200 个文件 | 超出部分替换为 `- ...` |
| Git 提交数 | 最多 10 次 | `@git:N` 限制在 [1, 10] 范围内 |

## 安全性

### 敏感路径拦截

以下路径始终被 `@file:` 引用拦截，以防止凭据泄露：

- SSH 密钥及配置：`~/.ssh/id_rsa`、`~/.ssh/id_ed25519`、`~/.ssh/authorized_keys`、`~/.ssh/config`
- Shell 配置文件：`~/.bashrc`、`~/.zshrc`、`~/.profile`、`~/.bash_profile`、`~/.zprofile`
- 凭据文件：`~/.netrc`、`~/.pgpass`、`~/.npmrc`、`~/.pypirc`
- Hermes 环境文件：`$HERMES_HOME/.env`

以下目录被完全拦截（目录内的任意文件均不可访问）：
- `~/.ssh/`、`~/.aws/`、`~/.gnupg/`、`~/.kube/`、`$HERMES_HOME/skills/.hub/`

### 路径遍历防护

所有路径均相对于工作目录解析。解析结果超出允许的工作区根目录的引用将被拒绝。

### 二进制文件检测

通过 MIME 类型和空字节扫描检测二进制文件。已知文本扩展名（`.py`、`.md`、`.json`、`.yaml`、`.toml`、`.js`、`.ts` 等）会跳过基于 MIME 的检测。二进制文件将被拒绝并附带警告。

## 平台可用性

Context references 主要是 **CLI 功能**。它们在交互式 CLI 中有效，`@` 触发 tab 补全，引用在消息发送给 agent 之前完成展开。

在**消息平台**（Telegram、Discord 等）中，`@` 语法不会被 gateway 展开——消息原样透传。agent 本身仍可通过 `read_file`、`search_files` 和 `web_extract` 工具引用文件。

## 与 Context 压缩的交互

当对话 context 被压缩时，展开后的引用内容会被纳入压缩摘要。这意味着：

- 通过 `@file:` 注入的大文件内容会占用 context 用量
- 若对话后续被压缩，文件内容将被摘要处理（而非原文保留）
- 对于非常大的文件，建议使用行范围（`@file:main.py:100-200`）仅注入相关片段

## 常用模式

```text
# 代码审查工作流
Review @diff and check for security issues

# 带上下文的调试
This test is failing. Here's the test @file:tests/test_auth.py
and the implementation @file:src/auth.py:50-80

# 项目探索
What does this project do? @folder:src @file:README.md

# 研究
Compare the approaches in @url:https://arxiv.org/abs/2301.00001
and @url:https://arxiv.org/abs/2301.00002
```

## 错误处理

无效引用会产生内联警告而非直接报错：

| 条件 | 行为 |
|-----------|----------|
| 文件未找到 | 警告："file not found" |
| 二进制文件 | 警告："binary files are not supported" |
| 文件夹未找到 | 警告："folder not found" |
| Git 命令失败 | 警告附带 git stderr 输出 |
| URL 无内容返回 | 警告："no content extracted" |
| 敏感路径 | 警告："path is a sensitive credential file" |
| 路径超出工作区 | 警告："path is outside the allowed workspace" |