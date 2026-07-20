---
name: narrato-ai
title: NarratoAI — 影视解说视频生成
description: "通过 NarratoAI 生成高质量影视解说视频，支持自动配音、字幕、背景音乐"
version: 1.0.0
author: integration-team
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Video, Narration, TTS, Media, Content-Creation]
    category: creative
    requires_toolsets: []
---

# NarratoAI 影视解说视频生成

通过深度集成 NarratoAI 项目，提供完整的影视解说视频生成能力。支持自动 TTS 配音、字幕生成、背景音乐混音、视频裁剪合并。

## When to use

- 用户要求生成影视解说视频
- 用户需要将长视频裁剪为短视频片段
- 用户需要自动配音、添加字幕
- 用户需要批量生成视频内容

## 核心功能

1. **视频脚本生成**: 基于原视频生成解说脚本（需要 LLM）
2. **TTS 配音**: 支持多种语音引擎（edge-tts, azure, doubao 等）
3. **字幕生成**: 自动生成 SRT 字幕并合并到视频
4. **背景音乐**: 支持随机或指定 BGM
5. **视频裁剪**: 智能裁剪原视频片段
6. **视频合并**: 自动合并所有素材生成最终视频

## 前置要求

- FFmpeg 已安装 (`brew install ffmpeg`)
- NarratoAI 依赖已安装 (`cd NarratoAI && uv sync`)
- Python 3.12+ 环境

## 使用方法

### 1. 生成影视解说视频

```python
from hermes_agent.tools.narrato_tools import generate_narration_video

result = generate_narration_video(
    video_path="/path/to/original.mp4",
    script_json="/path/to/script.json",  # 解说脚本（LLM 生成）
    voice_name="zh-CN-YunjianNeural",     # 语音名称
    video_aspect="9:16",                   # 视频比例
    output_dir="/path/to/output"
)
```

### 2. 脚本格式

解说脚本 JSON 格式：
```json
[
  {
    "narration": "这是第一段解说词",
    "OST": 0,           // 0=仅解说, 1=仅原声, 2=解说+原声
    "timestamp": "00:00:05,000 --> 00:00:15,000"
  },
  {
    "narration": "这是第二段解说词",
    "OST": 2,
    "timestamp": "00:00:15,000 --> 00:00:30,000"
  }
]
```

### 3. 可用语音

- `zh-CN-YunjianNeural` - 男声（云健）
- `zh-CN-XiaoxiaoNeural` - 女声（晓晓）
- `zh-CN-YunxiNeural` - 男声（云希）
- `en-US-JennyNeural` - 英文女声
- 更多语音见 NarratoAI 文档

## 工具列表

| 工具 | 功能 |
|------|------|
| `generate_narration_video` | 生成完整影视解说视频 |
| `generate_tts_audio` | 仅生成 TTS 配音 |
| `merge_audio_video` | 合并音频和视频 |
| `add_subtitles` | 添加字幕到视频 |

## 配置

环境变量（可选）：
- `NARRATO_TTS_ENGINE`: 默认 TTS 引擎（edge_tts/azure/doubao）
- `NARRATO_VOICE_NAME`: 默认语音名称
- `NARRATO_OUTPUT_DIR`: 默认输出目录

## 注意事项

1. **FFmpeg 必需**: 视频处理依赖 FFmpeg，请确保已安装
2. **内存占用**: 处理长视频时内存占用较高，建议分段处理
3. **TTS 引擎**: edge-tts 免费但质量一般，azure/doubao 质量更好但需要 API key
4. **输出格式**: 默认输出 MP4 格式，H.264 编码

## 故障排除

- **FFmpeg not found**: 运行 `brew install ffmpeg`
- **TTS 失败**: 检查网络连接（edge-tts 需要联网）
- **内存不足**: 减少 `n_threads` 参数或分段处理
