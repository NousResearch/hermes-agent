# Offline dataset token budgets

`scripts/sample_and_compress.py` uses the `tokenizer` section of its compression
YAML for both sampling and compression. Select the tokenizer for the **target
training model**, regardless of which models generated the source datasets:

```yaml
tokenizer:
  name: Qwen/Qwen2.5-0.5B-Instruct
  revision: 7ae557604adf67be50417f59c2c2f167def9a775
  trust_remote_code: false
compression:
  target_max_tokens: 15250
  summary_target_tokens: 750
```

Run with `python scripts/sample_and_compress.py --config=/path/to/compression.yaml`.
The standalone compressor reads the same YAML with `--config`. A local tokenizer
directory can also be used as `name`; it must include a chat template. The existing
default remains Kimi, whose custom tokenizer requires `trust_remote_code: true`.

Hub branches and tags are resolved to a commit before sampling workers start. The
same resolved revision is passed into compression. Tokenizer name, resolved
revision, chat-template SHA-256, and BOS/EOS/UNK token IDs are printed when loaded
and included in `compression_metrics.json`. Retain that report with the dataset.
For a local tokenizer directory, retain the tokenizer files as well; there is no
Hub revision identifying those files.

Counts include the whole conversation serialized by the selected tokenizer's
chat template (`tokenize=True`, `add_generation_prompt=False`). ShareGPT
`from`/`value` turns are mapped to `role`/`content` messages (`human` → `user`,
`gpt` → `assistant`); existing message metadata is preserved. Empty messages still
contribute formatting tokens. No extra BOS/EOS tokens are added after templating.
Debug logging reports BOS/EOS/UNK occurrences, and UNK occurrences emit a warning;
the expected BOS/EOS counts depend on the template, so no universal count is imposed.

Sampling includes trajectories at or above `min_tokens`. Compression skips ones
at or below `target_max_tokens`, and re-counts the complete result after inserting
the summary and notice. Content-only estimates still choose the initial summary
region; they do not decide whether a trajectory fits. Protected turns and an
oversized summary can still leave a result over budget, as reported by
`still_over_limit` and handled by the existing `save_over_limit` setting.

A missing/ambiguous chat template or a tokenization error is not replaced by a
character-count estimate. Re-sample old cached inputs when changing tokenizers:
`--skip_download` reuses the previous selection even though compression re-counts
its input with the current tokenizer.
