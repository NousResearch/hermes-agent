# Yandex Cloud LLM (YandexGPT) Provider

Native integration with [Yandex Cloud's LLM API](https://cloud.yandex.com/en/services/yandexgpt).

## Setup

### 1. Create a Yandex Cloud Account

- Go to [console.cloud.yandex.com](https://console.cloud.yandex.com)
- Create a new project (or use existing)
- Note your **Folder ID** (visible in the console sidebar)

### 2. Generate API Key

1. In the Yandex Cloud console, go to **Service Accounts**
2. Create a new service account or use existing
3. Add the role **ai.languageModels.user** to the account
4. Generate an **API Key** (not OAuth token)
5. Copy the key — you won't see it again

### 3. Configure Hermes

```bash
hermes setup

# Or manually set environment variables:
export YANDEX_GPT_API_KEY="your-api-key-here"
export YANDEX_GPT_FOLDER_ID="your-folder-id-here"

# Switch to Yandex GPT
hermes /model yandex-gpt:yandexgpt/pro
```

### 4. Verify Connection

```bash
hermes doctor
# Should show: ✓ yandex-gpt provider available
```

## Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `yandexgpt/latest` | Latest stable version | General purpose (recommended) |
| `yandexgpt/pro` | Advanced reasoning, complex tasks | Code, analysis, long documents |
| `yandexgpt/lite` | Fast, efficient inference | Quick responses, summaries |
| `yandexgpt/micro` | Smallest, fastest model | Simple tasks, high throughput |

## Environment Variables

| Variable | Required | Example |
|----------|----------|---------|
| `YANDEX_GPT_API_KEY` | ✅ | `AQVNy6F8h...` |
| `YANDEX_GPT_FOLDER_ID` | ✅ | `b1gxxxxxx...` |

## Configuration Example

In `~/.hermes/config.yaml`:

```yaml
model:
  provider: yandex-gpt
  name: yandexgpt/pro

yandex_gpt:
  api_key: ${YANDEX_GPT_API_KEY}
  folder_id: ${YANDEX_GPT_FOLDER_ID}
```

## Features

✅ **Native OpenAI-compatible API** — standard chat completions  
✅ **Automatic model discovery** — catalog fetching with fallbacks  
✅ **Graceful degradation** — works offline with cached models  
✅ **Full Hermes integration** — CLI, gateway, all platforms  
✅ **Cross-platform** — Linux, macOS, Windows, WSL2

## Troubleshooting

### "Invalid API key"

- Check the API key is correct (no spaces, proper format)
- Verify the service account has `ai.languageModels.user` role
- Try generating a new key in the console

### "Folder not found"

- Confirm `YANDEX_GPT_FOLDER_ID` matches your project
- Visible in console sidebar at the top-left

### "Model not available"

- Check the model name is correct (`yandexgpt/pro`, not `pro` alone)
- Available models: `latest`, `pro`, `lite`, `micro`

### Network timeout

- Verify internet connection
- Check Yandex Cloud API status
- Increase timeout in advanced config if needed

## References

- [Yandex Cloud LLM Documentation](https://cloud.yandex.com/en/docs/foundation-models/concepts/yandexgpt)
- [API Reference](https://cloud.yandex.com/en/docs/foundation-models/api-ref)
- [Auth Methods](https://cloud.yandex.com/en/docs/iam/concepts/authorization)

## License

MIT — same as Hermes Agent
