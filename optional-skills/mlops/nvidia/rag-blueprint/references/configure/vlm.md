# VLM, VLM Embeddings & Image Captioning

## When to Use
User wants image understanding, visual content analysis, VLM inference, multimodal embeddings, or image captioning during ingestion.

## Restrictions
- Not available on B200 GPUs — use H100, A100 SXM 80GB, or RTX PRO 6000
- Requires extra GPU (GPU 1+ for 2-GPU systems, GPU 2+ for 3+ GPUs with fallback)
- VLM embeddings: experimental, PDF-only, no summarization, no citations with page-as-image
- Image captioning on Helm: on-prem only (modify `values.yaml` to enable)

## Process
1. Detect the deployment mode (Docker / Helm / Library). Docker: edit the active env file. Helm: edit `values.yaml`. Library: edit `notebooks/config.yaml`
2. Read the relevant source doc for detailed steps:
   - VLM generation: `docs/vlm.md`
   - VLM embeddings: `docs/vlm-embed.md`
   - Image captioning: `docs/image_captioning.md`
3. Start VLM NIM (self-hosted) or configure cloud endpoint (NVIDIA-hosted)
4. Set the required variables in the active config:
   - Enabling: `ENABLE_VLM_INFERENCE=true` and `APP_NVINGEST_EXTRACTIMAGES=True`
   - Disabling: re-comment those variables in the env file
5. Restart affected services and verify with a health check + image-containing document query

## Decision Table

| Goal | Source Doc | Docker Profile | Notes |
|------|-----------|---------------|-------|
| VLM replaces LLM | `docs/vlm.md` | `--profile vlm-generation` | LLM not started; set `VLM_TO_LLM_FALLBACK=false` |
| VLM + LLM fallback | `docs/vlm.md` | `--profile vlm-only` | Needs 3+ GPUs; both VLM and LLM running |
| VLM embeddings | `docs/vlm-embed.md` | `--profile vlm-embed` | Experimental; requires re-ingestion |
| Image captioning | `docs/image_captioning.md` | `--profile vlm-only` | Requires VLM NIM; Helm: on-prem only |
| Multimodal query | `docs/multimodal-query.md` | (depends on VLM mode) | Image + text querying |

## Agent-Specific Notes

- `--profile vlm-generation` skips the LLM entirely — use `--profile vlm-only` for fallback mode
- `VLM_TO_LLM_FALLBACK` defaults to `true`, but `vlm-generation` profile does not start LLM
- Helm VLM: disable `nim-llm` and enable `nim-vlm` (VLM uses LLM's GPU allocation)
- Helm fallback: keep both `nim-vlm` and `nim-llm` enabled, set `VLM_TO_LLM_FALLBACK: "true"`
- VLM context window is limited — keep queries self-contained
- Image captioning known issue: files without graphs/charts/tables/plots fail to ingest when captioning is enabled

### Key Env Vars (always needed)
- `ENABLE_VLM_INFERENCE=true` — master toggle
- `APP_NVINGEST_EXTRACTIMAGES=True` — extract images during ingestion
- `VLM_MS_GPU_ID=<gpu-id>` — self-hosted GPU assignment

## Notebooks
- `notebooks/image_input.ipynb` — Multimodal queries with VLM (text + image)

## Source Documentation
- `docs/vlm.md` — VLM generation (self-hosted, NVIDIA-hosted, Helm, Library)
- `docs/vlm-embed.md` — VLM embeddings (experimental)
- `docs/image_captioning.md` — Image captioning during ingestion
- `docs/multimodal-query.md` — Image + text querying
- `docs/service-port-gpu-reference.md` — default GPU assignments for VLM and other NIMs
