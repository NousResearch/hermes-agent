

## Summary

This PR attempts four optimizations but contains multiple critical architectural bugs that will cause runtime failures, security issues, and silent data corruption. The benchmarks in the PR description are marketing fiction — the core "One-Shot Hydration" feature is architecturally broken, the schema minifier deletes semantically meaningful fields, and the output reducer has a Python slice bug that bypasses truncation entirely.

## Critical Issues (Blocking)

### 1. One-Shot Hydration is Architecturally Broken — Repaired Names Still Fail Validation

`repair_tool_call` can now return a deferred tool name (e.g., `"process_manage"`). But the caller in `turn_tool_validation.py`:

```python
valid_names = agent.valid_tool_names
for tc in tool_calls:
    if tc.function.name not in valid_names:
        repaired = agent._repair_tool_call(tc.function.name)
        if repaired:
            tc.function.name = repaired
invalid_tool_calls = [tc.function.name for tc in tool_calls if tc.function.name not in valid_names]
```

`valid_names` does NOT contain deferred names. The repaired name is written to `tc.function.name`, then **immediately flagged as invalid on the very next line**. This increments `_invalid_tool_retries`, sends an error message back to the model, and after 3 strikes aborts the turn.

The test `TestFastToolResolver` only tests `repair_tool_call` in isolation — it never tests the actual validation loop. This is a textbook false-confidence test masking a fatal integration bug.

**The entire One-Shot Hydration feature does the opposite of what it claims.** It makes things worse: previously an unknown deferred name would fail once and the model would retry via `tool_search`. Now it "repairs" to the deferred name, which still fails validation, burning a retry attempt on a name that can never pass.

### 2. `output[-tail_chars:]` When `tail_chars == 0` Returns the Entire String

In `reduce_tool_output`, the no-SNR-blocks fallback path:

```python
head_chars = int(available_chars * 0.4)
tail_chars = available_chars - head_chars
```

If `available_chars` is small enough that `tail_chars` computes to 0, or in the SNR path where `tail_chars = int(available_chars * 0.25)` could round to 0:

```python
return output[:head_chars] + notice + output[-tail_chars:]
```

`output[-0:]` in Python is `output[0:]` — the **entire original string**. This completely bypasses truncation and can inject megabytes into the context window, blowing past token limits and causing downstream failures.

The guard `if tail_chars > 0` exists only in the no-SNR fallback. The SNR path at the bottom of the function has no such guard:

```python
return output[:head_chars] + notice + output[-tail_chars:]  # no guard
```

### 3. `additionalProperties: false` Deletion Breaks Strict Structured Outputs

```python
if k == "additionalProperties" and v is False:
    continue
```

OpenAI's structured outputs API **requires** `additionalProperties: false` on every object schema. Anthropic and other providers also use it for strict validation. This is not "redundant metadata" — it is a semantic constraint. Deleting it causes:
- OpenAI API rejections with `"additionalProperties is required"` errors
- Schema validation failures where extra properties are now silently accepted
- Behavioral changes in model output structure

The PR description calls this "lossless" — it is not.

### 4. Bytes vs Chars Mismatch in `code_execution_tool.py`

```python
text = reduce_tool_output(stdout_text, MAX_STDOUT_BYTES)
```

`MAX_STDOUT_BYTES` is 50,000 (bytes). `reduce_tool_output` compares `len(output)` which counts **unicode code points** (chars). For multi-byte UTF-8 content (CJK, emoji, etc.), a 50,000-char string can be 150,000+ bytes. The truncation target is wrong in both directions: it either truncates too aggressively (ASCII) or not enough (multi-byte), and the "bytes" metadata reported to the user is incorrect.

### 5. No Cycle Detection in Schema Minifier — Stack Overflow on Recursive Schemas

`minify_and_sort_schema_node` recurses into every dict value and list element with no visited-node tracking. JSON Schemas with `$ref` cycles or self-referential structures (common in OpenAPI specs) will cause `RecursionError` and crash the tool loading path.

The `try/except` in `model_tools.py` catches this, but then **all tools load without any minification**, silently degrading the claimed optimization with no alert beyond a warning log.

### 6. ReDoS Risk in `_ERROR_BLOCK_PATTERNS`

```python
re.compile(r"(Traceback \(most recent call last\):[\s\S]*?(?:^\w*(?:Error|Exception|Exit|Interrupt):.*$))", re.M)
```

On a multi-megabyte log output that contains `"Traceback (most recent call last):"` but no matching `Error:` line, `[\s\S]*?` scans the entire remaining text character by character with backtracking. With 5 patterns run sequentially on potentially 5MB+ outputs, this is a performance bomb. `_truncate_stdout_text` is called on every code execution — this is a hot path.

## Required Changes

1. **Remove the entire One-Shot Hydration feature** from `repair_tool_call`. It cannot work without also modifying `turn_tool_validation.py` to add repaired deferred names to `valid_names` AND registering the tool schema with the executor so `_unwrap_tool_search_call` can dispatch it. This is not a one-line fix — it requires rearchitecting the tool registration pipeline. Ship it separately after proper design.

2. **Do not delete `additionalProperties: false`**. Remove that condition entirely from the minifier. If you want to strip truly redundant fields, limit to `$schema` and `title` on nested properties only — and document exactly which providers this is safe for.

3. **Fix the `output[-0:]` slice bug**:
   ```python
   if tail_chars > 0:
       result = output[:head_chars] + notice + output[-tail_chars:]
   else:
       result = output[:head_chars] + notice
   ```
   Apply this in BOTH code paths (SNR and non-SNR).

4. **Fix bytes/chars mismatch**: Either convert `MAX_STDOUT_BYTES` to an approximate char count before passing to `reduce_tool_output`, or make `reduce_tool_output` byte-aware. The simplest fix:
   ```python
   # ponytail: approximate; exact byte accounting needs encode/measure loop
   max_chars = MAX_STDOUT_BYTES  # conservative for ASCII; for multi-byte, pre-truncate by bytes first
   ```
   But the current code already does byte-level truncation before this point — the real fix is to not mix the two. Keep the existing byte-level head/tail split for `code_execution_tool.py` and only use `reduce_tool_output` for the char-based `terminal_tool_result.py` path.

5. **Add cycle detection** to `minify_and_sort_schema_node` or cap recursion depth:
   ```python
   def minify_and_sort_schema_node(node, *, prune_meta=True, _depth=0):
       if _depth > 50:
           return node
       # ... recurse with _depth=_depth+1
   ```

6. **Add timeout/size guard** before running regex patterns:
   ```python
   if len(text) > 500_000:
       return []  # skip SNR extraction on huge outputs; fall back to head/tail
   ```

7. **Delete `TestFastToolResolver`** — it tests a feature that must be removed. Replace with an integration test that verifies `turn_tool_validation.py` end-to-end behavior when `repair_tool_call` returns various names.

8. **Do not sort tools alphabetically**. Tool ordering in the schema affects model behavior (tools listed first get preferential attention in some models). Sorting by name is a behavioral change disguised as an optimization. Remove `sorted(processed, key=_tool_sort_key)` — key sorting within schemas is fine, but reordering the tool list is not.

## Suggestions

- The `extract_high_snr_blocks` line-splitting path creates `O(n)` string objects. For the common case (no multi-line error blocks found, output is just noise), consider checking `len(extracted) < max_blocks` before doing `text.splitlines()`.
- The `--- [HIGH-SNR DIAGNOSTIC ...` banner injected into tool output will corrupt any tool that returns structured JSON. Consider checking if the output starts with `{` or `[` and skipping SNR extraction for JSON payloads.
- `copy.deepcopy` in `minify_tool_definition` is expensive for large schemas. Since you're already rebuilding the dict via comprehension, a recursive rebuild without deepcopy would be faster.

## Verdict

**Reject.** The One-Shot Hydration feature is architecturally broken at the integration level — it makes tool resolution strictly worse. The schema minifier deletes semantically required fields. The output reducer has a Python slice edge case that can bypass truncation entirely. The test suite provides false confidence by testing components in isolation without verifying the actual runtime behavior. Strip the hydration feature, fix the three data-corruption bugs, and re-submit.