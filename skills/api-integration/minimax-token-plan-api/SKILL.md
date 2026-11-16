---
name: minimax-token-plan-api
description: MiniMax Token Plan API integration guide - endpoints, auth, response formats for image/music/TTS (not just M2.7)
---

# MiniMax Token Plan API Integration

## Overview
MiniMax Token Plan provides an Anthropic-compatible API for M2.7 text chat only. Other modalities (image, music, TTS) use the legacy `/v1/` endpoints.

## API Endpoints

| Modality | Endpoint | Base URL |
|----------|----------|----------|
| **Text Chat (M2.7) - OpenAI兼容** | **`/v1/chat/completions`** | **`https://api.minimaxi.com/v1`** |
| Text Chat (M2.7) - Anthropic兼容 | `/anthropic/v1/messages` | `https://api.minimaxi.com/anthropic/v1` |
| Image Generation | `/v1/image_generation` | `https://api.minimaxi.com/v1` |
| Music Generation | `/v1/music_generation` | `https://api.minimaxi.com/v1` |
| TTS Speech | `/v1/t2a_v2` | `https://api.minimaxi.com/v1` |
| Lyrics Generation | `/v1/lyrics_generation` | `https://api.minimaxi.com/v1` |

## OpenAI 兼容 API（推荐用于 Web 应用）

Token Plan 支持 OpenAI 兼容格式，比 Anthropic 格式更简单：

**端点**: `POST https://api.minimaxi.com/v1/chat/completions`

**请求示例**:
```javascript
const response = await fetch('https://api.minimaxi.com/v1/chat/completions', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
        model: 'MiniMax-M2.7',           // 或 'MiniMax-M2.7-highspeed'
        max_tokens: 500,                  // 不是 tokens_to_generate
        temperature: 0.8,
        messages: [{ role: 'user', content: 'prompt' }]
    })
});
const data = await response.json();
const content = data.choices[0].message.content;  // 不是 choices[0].text
```

**可用模型**: `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`

**注意**:
- `GroupId` 不需要在 URL 参数中（旧 API `api.minimax.chat` 才需要）
- `bot_setting` 和 `reply_constraints` 参数不是必需的
- 旧 API 格式 (`api.minimax.chat/v1/text/chatcompletion_pro`) 需要 `GroupId` 在 query 中，使用 `tokens_to_generate`，响应格式为 `choices[0].text`

**实测错误处理**:
- `insufficient balance`: API Key 余额不足或使用了错误的 API Key（Token Plan key 不能用于旧接口）
- `invalid params, binding: expr_path=...`: 缺少必需参数或参数格式错误

## Anthropic 兼容 API

适用于需要 Claude 兼容格式的场景：

```
POST https://api.minimaxi.com/anthropic/v1/messages
```

需设置 `ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic`

## Authentication
- Header: `Authorization: Bearer <API_KEY>`
- Token Plan API Key does NOT require `MM-Group-Id` header or GroupId query param
- Token Plan API Key is separate from usage-based billing API Key

## Response Formats

### Image Generation
```json
{
  "data": {
    "image_urls": ["https://..."]
  },
  "revised_prompt": "..."
}
```
Use `result.data.image_urls[0]` for the image URL.

### Music Generation
```json
{
  "data": {
    "audio": "<hex encoded audio>",
    "status": 2
  },
  "base_resp": { "status_code": 0, "status_msg": "success" }
}
```
**IMPORTANT**: `lyrics` parameter is **required** for music generation (even if empty string - API will auto-generate). Never omit it.

Check `result.data.status === 2` for successful generation. The `audio` field contains hex-encoded MP3 binary. Convert to base64 data URL for playback:
```javascript
function hexToBase64(hex) {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
  }
  const bin = String.fromCharCode(...bytes);
  return btoa(bin);
}
// Play: `data:audio/mp3;base64,${hexToBase64(audio)}`
```

### TTS Speech Synthesis
TTS endpoint returns **raw binary** (not JSON). Handle it with `response.arrayBuffer()`:
```javascript
const response = await fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`,
  },
  body: JSON.stringify(payload),
});
const buffer = await response.arrayBuffer();
const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
const audioUrl = `data:audio/mp3;base64,${base64}`;
```

TTS Payload example:
```json
{
  "model": "speech-2.8-hd",
  "text": "要合成的文本",
  "stream": false,
  "voice_setting": {
    "voice_id": "female-shaonv",
    "speed": 1,
    "vol": 1,
    "pitch": 0,
    "emotion": "happy"
  },
  "audio_setting": {
    "sample_rate": 32000,
    "bitrate": 128000,
    "format": "mp3",
    "channel": 1
  }
}
```

Available TTS voices (speech-2.8-hd):
- `male-qn-qingse` - 青年男声
- `female-shaonv` - 少女声音
- `male-qn-jingxing` - 激情男声
- `female-yujie` - 御姐声音
- `female-tianmei` - 甜妹声音
- `male-yunyang` - 云扬声音

### Image Generation Payload
```json
{
  "model": "image-01",
  "prompt": "图片描述",
  "n": 1,
  "aspect_ratio": "1:1",
  "response_format": "url",
  "prompt_optimizer": true
}
```
Aspect ratio mapping:
- `1024x1024` → `"1:1"`
- `1792x1024` → `"16:9"`
- `1024x1792` → `"9:16"`

## Common Issues

### 404 on /anthropic/v1 endpoints
**Cause**: Anthropic-compatible endpoint only works for M2.7 text. Use `/v1/` endpoints for other modalities.

### TTS returns non-JSON response
**Cause**: TTS `/v1/t2a_v2` returns raw binary, not JSON. Must use `response.arrayBuffer()` instead of `response.json()`.

### Music audio doesn't play
**Cause**: Music API returns hex-encoded binary, not base64. Must convert before creating data URL.

## Token Plan Quotas (Reference)
- image-01: Plus 50张/日, Max 120张/日
- Music-2.6: 100首/天 (每首≤5分钟)
- Speech 2.8: Plus 4000字符/日, Max 11000字符/日
- M2.7: 按请求数计，每5小时滚动重置

## Trigger Conditions
Use this skill when:
- Integrating with MiniMax Token Plan API
- Building H5/web apps that call MiniMax APIs (not just M2.7 chat)
- Getting 404 errors on MiniMax API endpoints
- Handling TTS or music audio binary responses
