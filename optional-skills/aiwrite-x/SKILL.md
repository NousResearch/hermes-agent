# AIWriteX Skill

多平台内容创作技能，支持微信公众号、小红书、百家号等平台的文章生成。

## 功能特性

- **多平台支持**: 微信公众号、小红书、百家号、知乎、豆瓣等
- **智能内容生成**: 基于 CrewAI 多智能体协作
- **自动搜索**: 集成 AIForge 搜索引擎获取最新信息
- **创意变换**: 维度化创意引擎提升内容质量
- **模板支持**: 可选 HTML 模板或 AI 设计排版
- **自动发布**: 支持自动化发布到目标平台

## 触发条件

当用户请求以下内容时自动触发：
- "写一篇公众号文章"
- "生成小红书笔记"
- "创作百家号文章"
- "帮我写一篇文章关于..."
- 任何涉及内容创作、文章生成的请求

## 使用方法

### 1. 生成文章

```python
from hermes_agent.tools.aiwrite_tools import aiwrite_generate_article

result = aiwrite_generate_article(
    topic="人工智能的最新发展",
    platform="wechat",
    urls=["https://example.com/ref1"],
    reference_ratio=0.3,
    min_len=1000,
    max_len=2000
)
```

### 2. 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| topic | string | ✓ | 文章主题 |
| platform | string | ✗ | 目标平台（wechat/xiaohongshu/baijiahao/zhihu/douban） |
| urls | list | ✗ | 参考文章 URL 列表 |
| reference_ratio | float | ✗ | 参考比例（0.0-1.0） |
| min_len | int | ✗ | 最小字数（默认 1000） |
| max_len | int | ✗ | 最大字数（默认 2000） |
| use_template | bool | ✗ | 是否使用模板（默认 false） |

### 3. 返回结果

```json
{
  "success": true,
  "title": "文章标题",
  "content": "文章正文（Markdown 或 HTML）",
  "save_path": "/path/to/saved/article.md",
  "publish_result": {
    "success": true,
    "url": "https://..."
  }
}
```

## 工作流程

1. **内容生成**: 基于主题和参考资料生成基础内容
2. **创意变换**: 使用维度化创意引擎提升内容质量
3. **格式转换**: 根据平台要求转换为 HTML 或保持 Markdown
4. **保存发布**: 保存到本地并可选自动发布

## 配置说明

AIWriteX 使用配置文件 `config.yaml` 管理参数：

```yaml
# 发布平台
publish_platform: wechat

# 文章长度
min_article_len: 1000
max_article_len: 2000

# 格式设置
article_format: markdown  # 或 html
use_template: false

# AIForge 搜索
aiforge_api_key: your_api_key
```

## 依赖要求

- Python 3.12+
- CrewAI >= 0.102.0
- AIForge Engine
- 所有依赖已安装在 `/Users/xiaoxi/Downloads/workspace/AIWriteX/.venv`

## 注意事项

1. **API Key**: 使用 AIForge 搜索功能需要配置 API Key
2. **发布功能**: 自动发布需要配置目标平台的认证信息
3. **模板路径**: 自定义模板需放在 `templates/` 目录下
4. **日志查看**: 执行日志保存在 `logs/` 目录

## 故障排查

### 导入失败
```bash
# 验证依赖安装
cd /Users/xiaoxi/Downloads/workspace/AIWriteX
source .venv/bin/activate
pip list | grep -E "crewai|aiforge"
```

### 搜索失败
- 检查 AIForge API Key 是否配置
- 确认网络连接正常
- 查看 `logs/aiwrite.log` 获取详细错误

### 发布失败
- 验证平台认证信息
- 检查网络连接
- 确认目标平台 API 可用

## 示例场景

### 场景1: 生成微信公众号文章
```
用户: 帮我写一篇关于量子计算的公众号文章
→ 调用 aiwrite_generate_article(topic="量子计算", platform="wechat")
```

### 场景2: 基于参考资料创作
```
用户: 参考这几篇文章写一篇小红书笔记: [url1, url2]
→ 调用 aiwrite_generate_article(topic="...", platform="xiaohongshu", urls=[...], reference_ratio=0.5)
```

### 场景3: 使用模板排版
```
用户: 用模板生成一篇百家号文章
→ 调用 aiwrite_generate_article(topic="...", platform="baijiahao", use_template=true)
```
