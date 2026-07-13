# Ingestion: Text-Only, Audio, Nemotron Parse, OCR & Batch

## When to Use
User wants to configure ingestion mode (text-only, audio, Nemotron Parse), switch OCR engines, save extraction results to disk, use standalone NV-Ingest, tune ingestion performance, or run batch ingestion.

## Restrictions
- Nemotron Parse: not available on B200 or RTX PRO 6000 GPUs (requires H100 or A100 SXM 80GB)
- Audio on Helm: not supported on RTX PRO 6000
- Nemotron Parse GPU conflict: read `docs/service-port-gpu-reference.md` for default GPU assignments. Nemotron Parse defaults to the same GPU as LLM — reassign on limited-GPU systems

## Process

1. Detect the deployment mode (Docker self-hosted / NVIDIA-hosted / Helm / Library). Docker: edit the active env file. Helm: edit `values.yaml`. Library: edit `notebooks/config.yaml`
2. Read the relevant source doc for detailed configuration
3. Apply the required env vars to the active config, restart ingestor (and NIM services if enabling new profiles)
4. Verify: upload a test document and check ingestion status

## Decision Table

| Goal | Source Doc | Key Action |
|------|-----------|------------|
| Text-only ingestion | `docs/text_only_ingest.md` | Set extract vars to False, set `COMPONENTS_TO_READY_CHECK=""` |
| Audio ingestion | `docs/audio_ingestion.md` | Start audio NIM (`--profile audio`), set `AUDIO_MS_GPU_ID` |
| Nemotron Parse | `docs/nemotron-parse-extraction.md` | `APP_NVINGEST_PDFEXTRACTMETHOD=nemotron_parse`, start NIM |
| OCR config/switch | `docs/nemoretriever-ocr.md` | Switch between NeMo Retriever OCR and Paddle OCR |
| Save to disk | `docs/mount-ingestor-volume.md` | `APP_NVINGEST_SAVETODISK=True`, mount volume |
| Standalone NV-Ingest | `docs/nv-ingest-standalone.md` | Direct Python client, no full ingestor server |
| Batch ingestion | See `scripts/batch_ingestion.py` | `python scripts/batch_ingestion.py --folder ... --collection-name ...` |
| Tune performance | `docs/accuracy_perf.md` | Adjust chunk size, overlap, batch settings |
| Summarization at ingest | `references/configure/summarization.md` | `generate_summary: true` in upload payload |

## Agent-Specific Notes

- Text-only mode: set `COMPONENTS_TO_READY_CHECK=""` in the active env file so NV-Ingest does not wait for disabled extraction services. If the compose file hardcodes `COMPONENTS_TO_READY_CHECK=ALL`, update it to `${COMPONENTS_TO_READY_CHECK:-ALL}` so the env var takes effect
- Use `--profile rag` with nims.yaml to skip OCR/detection NIMs in text-only mode
- Audio formats supported: `.mp3`, `.wav`, `.mp4`, `.avi`, `.mov`, `.mkv`
- Riva ASR requires ~8GB VRAM
- NeMo Retriever OCR is 2x+ faster than Paddle OCR but needs 8GB vs 3GB VRAM
- Batch CLI: `pip install -r scripts/requirements.txt` first; idempotent (skips already-ingested files)
- MIG deployments: reduce batch sizes for large bulk ingestion jobs

## Notebooks
- `notebooks/ingestion_api_usage.ipynb` — Ingestor API: collections, uploads, document management

## Source Documentation
- `docs/text_only_ingest.md` — Text-only ingestion (skip OCR/detection)
- `docs/audio_ingestion.md` — Audio/video ingestion via ASR
- `docs/nemotron-parse-extraction.md` — Nemotron Parse PDF extraction
- `docs/nemoretriever-ocr.md` — OCR configuration and switching
- `docs/mount-ingestor-volume.md` — Volume mount for extraction results
- `docs/nv-ingest-standalone.md` — Standalone NV-Ingest without ingestor server
- `docs/accuracy_perf.md` — Ingestion tuning settings (chunk size, overlap, batch params)
- `docs/service-port-gpu-reference.md` — OCR port mappings and GPU assignments
