# Volcengine Ark Provider (火山方舟)

This provider adds support for [Volcengine Ark](https://www.volcengine.com/product/ark), ByteDance's AI model platform.

## Features

- **Agent Plan Support**: Subscription-based access with dedicated endpoint (`/api/plan/v3`)
- **Multiple Models**: Doubao (豆包), DeepSeek-V3.2, GLM-5.1, Kimi-K2.6, MiniMax-M2.7, and more
- **OpenAI Compatible**: Full compatibility with OpenAI SDK and tool calling

## Setup

### 1. Get API Key

1. Sign up at [Volcengine](https://www.volcengine.com/)
2. Complete real-name authentication (实名认证)
3. Subscribe to [Agent Plan](https://console.volcengine.com/ark/agent-plan)
4. Create API Key at: 控制台 → API 密钥管理 → 创建 API Key

### 2. Configure Hermes

```bash
# Set API key
export VOLCENGINE_ARK_API_KEY="ark-your-key-here"

# Or use hermes config
hermes config set model.provider volcengine-ark
hermes config set model.api_key "ark-your-key-here"
hermes config set model.default "doubao-seed-1.6"  # or other model ID
```

### 3. Available Models

Common model IDs (check your console for exact IDs):
- `doubao-seed-1.6` - Doubao latest
- `deepseek-v4-pro` - DeepSeek V4 Pro
- `glm-5.2` - GLM 5.2 (latest)
- `kimi-k2.6` - Kimi K2.6
- `minimax-m2.7` - MiniMax M2.7

## Agent Plan vs API

- **Agent Plan**: Subscription-based, uses `/api/plan/v3` endpoint, AFP credit billing
- **API**: Pay-per-use, uses `/api/v3` endpoint, token-based billing

This provider defaults to Agent Plan endpoint. For API usage, override base_url:

```bash
hermes config set model.base_url "https://ark.cn-beijing.volces.com/api/v3"
```

## Troubleshooting

- **Model not found**: Use exact model ID from your console (e.g., `ark-code-latest`)
- **Authentication failed**: Verify API key starts with `ark-`
- **Endpoint mismatch**: Agent Plan uses `/api/plan/v3`, not `/api/v3`

## Links

- [Product Page](https://www.volcengine.com/product/ark)
- [Documentation](https://www.volcengine.com/docs/82379/1330310)
- [API Reference](https://www.volcengine.com/docs/82379/1511946)
