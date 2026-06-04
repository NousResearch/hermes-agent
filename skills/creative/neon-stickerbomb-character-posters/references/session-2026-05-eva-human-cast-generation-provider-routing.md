# Session note: EVA human cast neon batch provider/routing lessons

Context: Nick requested a seven-image Evangelion human/mascot batch in the established neon sticker-bomb style: Misato, Kaworu, Gendo, Ritsuko, Pen Pen, Mari, and eyepatch Asuka.

Durable lessons for future neon/IP character batches:

- Preserve the numbered manifest before any generation attempt. This keeps later `发布 1 3 7` mappings stable even if the provider hangs or a process is killed.
- For CLIProxyAPI `/v1/responses` image generation, do **not** add `tool_choice` for `image_generation`. In this session it returned a hard 400: `Tool choice 'image_generation' not found in 'tools' parameter.` The safe route is to provide the `image_generation` tool and strongly instruct the dialogue model to use it.
- Long, dense IP prompts can hang with no stdout and no manifest update. If a first item runs longer than the normal image-generation window without any output, kill it and retry with a shorter prompt rather than waiting through the whole batch.
- Do a one-image smoke test with a compact prompt before launching a long multi-image batch. A compact Misato prompt produced a valid result after ~114 seconds, while longer batch prompts hung.
- Avoid concurrent retries when the proxy is unstable. Parallel seven-image or three-image retries produced 502s in this session; use serial retries and update the manifest after each completed item.
- If the generated image call completes but returns only reasoning/message output and no `image_generation_call.result`, treat it as a routing failure, not a usable generation. Retry with shorter wording and clearer `Use image_generation tool. Generate exactly one image.` instructions.
- For large IP casts, keep the style block compact. Put only the essential identity cues, one composition archetype, dense neon/stickerbomb requirements, and safety negatives in the active prompt. Store the richer class-level guidance in the skill/reference, not in every single request payload.

Recommended recovery pattern:

1. Write or preserve the manifest with all intended indices and prompts.
2. Run a single compact prompt smoke test.
3. If it succeeds, generate serially, one image per request, saving the manifest after every item.
4. If an item hangs past the expected window, kill only that process and retry only that index with a shorter prompt.
5. If parallel attempts return 502, stop parallelizing; do not mark the provider as permanently broken.
