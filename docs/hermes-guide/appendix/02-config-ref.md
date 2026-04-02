# 配置项参考

## B.1 默认配置

```python
# hermes_cli/config.py
DEFAULT_CONFIG = {
    # === 模型配置 ===
    "model": "claude-sonnet-4-6",       # 默认模型
    "base_url": "https://api.anthropic.com",  # API 端点
    "api_key": None,                     # API 密钥
    "max_tokens": 8192,                 # 最大输出 tokens
    "temperature": 0.7,                 # 温度参数
    "top_p": 1.0,                       # Top-p 采样

    # === 工具配置 ===
    "toolsets": [                        # 启用的工具集
        "files",
        "terminal",
        "web",
        "browser",
    ],
    "disabled_tools": [],               # 禁用的工具
    "tool_iteration_limit": 10,         # 单次对话最大工具调用

    # === CLI 配置 ===
    "theme": "default",                  # 皮肤主题
    "quick_sync": False,                 # 快速同步模式
    "stream": True,                     # 流式输出

    # === 代理配置 ===
    "http_proxy": None,
    "https_proxy": None,

    # === 会话配置 ===
    "session_db_path": "~/.hermes/sessions.db",
    "max_session_messages": 1000,       # 会话最大消息数

    # === 上下文压缩 ===
    "compression_threshold": 0.8,       # 压缩触发阈值
    "context_window": 200000,            # 上下文窗口大小

    # === 日志 ===
    "log_level": "INFO",
    "log_file": "~/.hermes/hermes.log",
}
```

---

## B.2 环境变量

| 变量 | 说明 | 优先级 |
|------|------|--------|
| `HERMES_MODEL` | 默认模型 | 高 |
| `HERMES_BASE_URL` | API 端点 | 高 |
| `HERMES_API_KEY` | API 密钥 | 高 |
| `HERMES_HOME` | 配置目录 | 高 |
| `ANTHROPIC_API_KEY` | Anthropic 密钥 | 中 |
| `OPENAI_API_KEY` | OpenAI 密钥 | 中 |
| `TAVILY_API_KEY` | Tavily 搜索密钥 | 低 |
| `BROWSER_PROVIDER` | 浏览器提供者 | 低 |

---

## B.3 配置文件

```
~/.hermes/
├── config.yaml          # 用户配置
├── .env                 # API 密钥
├── sessions.db          # SQLite 会话数据库
├── skills/              # 用户安装的技能
├── skins/               # 用户皮肤
└── hermes.log          # 日志文件
```

---

## B.4 配置优先级

```
高 ─► 环境变量 (HERMES_*)
    │
    │ CLI 参数 (--model)
    │
    │ 用户配置 (~/.hermes/config.yaml)
    │
低 ─► 默认配置 (hermes_cli/config.py::DEFAULT_CONFIG)
```

---

[← 返回总目录](../SUMMARY.md) | [← 文件索引](01-file-index.md) | [术语表 →](03-glossary.md)
