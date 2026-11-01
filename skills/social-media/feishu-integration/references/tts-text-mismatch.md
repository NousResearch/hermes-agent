# TTS Text Mismatch: Image Content Leaks into Voice Output

## Symptom
TTS voice output doesn't match the AI's written reply. Voice contains fragments like emoji descriptions ("smiling eyes") that don't appear in the text, or is shorter than expected.

## Root Cause Chain

### Layer 1: `extract_images()` strips image markdown
`gateway/platforms/base.py` extracts image URLs from markdown (`![alt](url)`) and HTML tags. The extracted URLs are removed from content before TTS processing.

### Layer 2: TTS cleanup regex is incomplete
In base.py line ~2788, text passed to TTS goes through:
```python
speech_text = re.sub(r'[*_`#\\\\[\\\\]()]', '', text_content)[:4000].strip()
```
This strips `*`, `_`, `` ` ``, `#`, `\`, `[`, `]`, `(`, `)` — but does NOT strip words.

### The Bug (wrong fix)
If you replace image markdown with `[image: alt_text]` to preserve alt-text for TTS, the cleanup regex removes brackets but leaves `image smiling eyes` as plain text → TTS speaks it.

### Correct Fix
Completely discard all image-related content from text passed to TTS. In `extract_images()`:
```python
# Remove image markdown entirely — alt-text must NOT enter TTS
if images:
    extracted_urls = {url for url, _ in images}
    def _remove_extracted(match):
        url = match.group(2) if match.lastindex >= 2 else match.group(1)
        return '' if url in extracted_urls else match.group(0)
    cleaned = re.sub(md_pattern, _remove_extracted, cleaned)
    cleaned = re.sub(html_pattern, _remove_extracted, cleaned)
```

## Debugging Steps
1. Check if AI reply contains image markdown (`![...](url)`) or HTML `<img>` tags
2. Compare TTS text vs written reply — look for missing segments OR unexpected emoji descriptions
3. Verify `extract_images` in `base.py` around line 1748-1780
4. Ensure no alt-text preservation logic exists (that's the bug, not the fix)

## Related Code
- `gateway/platforms/base.py:1733-1781` — extract_images()
- `gateway/platforms/base.py:2788` — TTS text cleanup regex
- `tools/tts_tool.py` — want_opus platform setting
