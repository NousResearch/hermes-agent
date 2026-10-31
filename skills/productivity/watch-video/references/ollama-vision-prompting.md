# Ollama Vision Prompting Guide

## Overview
This document describes the prompt engineering strategy for the Ollama vision model (qwen3-vl:4b) used in the watch-video skill.

## Model Characteristics
- **Model**: qwen3-vl:4b (4.4B parameters, Q4_K_M quantization)
- **Vision encoder**: qwen3vl.vision with 24 blocks, 16 attention heads
- **Context length**: 262,144 tokens
- **Image processing**: 16x16 patches, spatial merge size 2, temporal patch size 2

## Prompt Design Principles

### 1. Structured Output Enforcement
The prompt explicitly requests JSON array output matching the `VisionResult` schema. This avoids parsing ambiguity.

### 2. Batch Processing
Frames are sent in batches of 5-8 with a single prompt. This reduces API overhead and ensures consistent analysis style across frames.

### 3. Timestamp Injection
Each frame reference includes its timestamp in the prompt, enabling the model to produce timestamp-aware descriptions.

## Prompt Template

```
Analyze these {N} video frames with timestamps. For each frame, provide:
1. Visual description (2-3 sentences)
2. Any visible text (OCR) — UI text, code, slides, captions on screen
3. UI elements detected (buttons, menus, charts, code editor, terminal, etc.)
4. Scene type: talking_head | screen_demo | slides | b_roll | diagram | other
5. Speaker change detected? (yes/no)

Frames:
Frame 0 [00:30.0]: <base64_image>
Frame 1 [00:35.0]: <base64_image>
...
Frame N [MM:SS.mmm]: <base64_image>

Output as JSON array matching frame order:
[
  {"frame_index": 0, "description": "...", "ocr_text": "...", "ui_elements": ["..."], "scene_type": "...", "speaker_change": false},
  ...
]
```

## Scene Type Taxonomy

| Scene Type | Description | Visual Indicators |
|------------|-------------|-------------------|
| `talking_head` | Person speaking to camera | Face prominent, minimal background change |
| `screen_demo` | Software/code demonstration | Code editor, terminal, browser, IDE |
| `slides` | Presentation slides | Slide deck, bullet points, diagrams |
| `b_roll` | Supplementary footage | Cutaway shots, environment, hands typing |
| `diagram` | Technical diagram/chart | Flowcharts, architecture diagrams, graphs |
| `other` | Doesn't fit above | Miscellaneous |

## UI Element Detection

Common UI elements to detect:
- `button`, `menu`, `toolbar`, `sidebar`, `tab`
- `code_editor`, `terminal`, `browser`, `ide`
- `chart`, `graph`, `table`, `form`, `dialog`
- `subtitle`, `caption`, `watermark`, `logo`

## OCR Guidelines

- Extract **visible text only** — don't hallucinate
- Prioritize: code, UI labels, slide content, captions
- Ignore: watermarks, timestamps, decorative text
- Return empty string if no readable text

## Speaker Change Detection

Visual cues for speaker change:
- Different person appears on screen
- Camera angle changes to different person
- Picture-in-picture switches
- Name/role overlay changes

Return `true` only when confident.

## Temperature Setting

Use `temperature: 0.1` for deterministic, factual descriptions. Higher temperatures increase creativity but reduce consistency.

## Batch Size Optimization

- **Small batches (4-6)**: Better for detailed analysis, higher latency
- **Large batches (8-12)**: Faster throughput, risk of truncated responses
- **Default: 8** — balanced for qwen3-vl:4b context window

## Error Handling

If model returns invalid JSON:
1. Strip markdown code fences
2. Attempt JSON repair (add missing brackets)
3. If still invalid, return placeholder with error description

## Future Improvements

- Few-shot examples in prompt for better consistency
- Chain-of-thought for complex scenes
- Separate OCR-only pass for text-heavy frames
- Confidence scoring per field