# Tool Prioritization — Weighted Tool Selection

## Problem

When Hermes exposes multiple tools with overlapping functionality, the LLM's tool selection is non-deterministic. The model chooses based on name similarity and training bias, not operational needs like cost, latency, quality, or debugging.

**Example from production:**
- `mcp_ocr_ocr_image_from_base64` (local OCR MCP server, OpenRouter vision + Tesseract fallback)
- `mcp_zai_vision_extract_text_from_screenshot` (zai-vision MCP server, GLM-4V)

Both do OCR. The model consistently picks the first one because "ocr_image_from_base64" matches the task name better, even though:
- zai-vision produces better quality
- zai-vision is the intended production tool
- We want deterministic behavior for debugging

This happens anytime tools overlap:
- Multiple OCR implementations
- Multiple search tools (web_search vs x_search)
- Multiple vision tools
- Local vs cloud variants

## Solution

Add **tool prioritization** via configurable weights per profile. Tools can be assigned weights (0-100) that influence the LLM's selection.

### Three modes of operation

**1. Reorder mode**
Sort tools by weight descending. LLMs sometimes prefer tools listed earlier in the schema.

**2. Description hints mode** (recommended)
Inject `[PREFERRED]` or `[FALLBACK]` markers into tool descriptions:
```
[PREFERRED] Extract text from screenshots using z.ai GLM-4V...
[FALLBACK] Read a local image file and extract text using OpenRouter...
```

**3. System prompt mode**
Append explicit directives to system prompt:
```
When performing OCR, prefer mcp_zai_vision_extract_text_from_screenshot over mcp_ocr_ocr_image_from_base64.
```

### Configuration

```yaml
# ~/.hermes/config.yaml
profiles:
  documents:
    model: glm-4.7
    provider: zai
    toolsets:
    - mcp-ocr
    - mcp-zai-vision
    tool_weights:
      mode: description_hints  # reorder | description_hints | system_prompt
      weights:
        mcp_zai_vision_extract_text_from_screenshot: 90
        mcp_ocr_ocr_image_from_base64: 10
```

### Manual competition groups (v1)

To avoid applying weights globally, tools are grouped into explicit competition sets:

```yaml
tool_weights:
  competitions:
    - name: ocr
      tools:
        - mcp_zai_vision_extract_text_from_screenshot
        - mcp_ocr_ocr_image_from_base64
      preferred: mcp_zai_vision_extract_text_from_screenshot  # weight 100
      fallback: mcp_ocr_ocr_image_from_base64                # weight 10
```

This is explicit and debuggable. Auto-detection (v2) can use TF-IDF/embeddings on tool descriptions.

## Benefits

1. **Cost control**: Prefer free tools over paid ones
2. **Quality gates**: Ensure high-quality tools are used first
3. **Latency optimization**: Prefer faster tools
4. **Determinism**: Same prompt → same tool choice
5. **Debuggability**: Easier to troubleshoot when you know which tool will be called
6. **Gradual rollout**: Shift weights from old→new implementation (50/50 → 30/70 → 0/100)
7. **A/B testing**: Compare tool effectiveness by adjusting weights

## Implementation

### Changes

1. **`model_tools.py`** (~100 lines)
   - Hook into `get_tool_definitions()` after tool filtering
   - Read `tool_weights` from profile config
   - Apply mode logic (reorder/hints/system_prompt)

2. **`cli-config.yaml.example`** (~20 lines)
   - Add `tool_weights` schema documentation

3. **Tests** (~200 lines)
   - Test tool reordering
   - Test description hint injection
   - Test system prompt mode
   - Test profile isolation
   - Test competition groups

### Total scope: ~320 lines

## Future work (v2)

- **Auto-detection**: Use embeddings to detect overlapping tools automatically
- **Dynamic weights**: Adjust weights based on success/failure telemetry
- **Tool categories**: Tools register `category: "ocr"` for automatic grouping
- **Context-aware weights**: Different weights based on task type

## Alternatives considered

1. **Plugin only** — Works, but this is core agent behavior that benefits all users
2. **Hardcoded tool aliases** — Too rigid, can't adjust weights
3. **Toolset filtering only** — Can't express "prefer A, use B as fallback"

## Questions for maintainers

1. **Mode preference**: Is `description_hints` the right default, or should we start with `reorder`?
2. **Competition groups**: Should we support manual groups (v1), or try auto-detection from the start?
3. **Profile vs global**: Should weights be profile-scoped only, or also support global defaults?
4. **Testing**: What test coverage is expected? Unit tests only, or integration tests with actual LLMs?

## Checklist

- [ ] Config schema in `cli-config.yaml.example`
- [ ] Core logic in `model_tools.get_tool_definitions()`
- [ ] Three modes implemented
- [ ] Competition groups supported
- [ ] Profile isolation tested
- [ ] Documentation updated
- [ ] Migration guide (if any breaking changes)

---

**Draft PR description — NO CODE CHANGES**