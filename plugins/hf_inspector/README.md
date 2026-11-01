# Hugging Face Model & Quant Explorer Plugin (`hf-inspector`)

Zero-dependency toolset allowing Hermes Agent to inspect model architecture, parameter count, context window size, licenses, and discover available GGUF quantization weights directly from Hugging Face repositories.

## Tools

- `hf_inspect_model(model_id)`: Fetches metadata including architecture, parameters (from safetensors), max sequence length / context tokens, license, download metrics, and gated status.
- `hf_list_quants(model_id)`: Scans repository files for GGUF, AWQ, and GPTQ files, reporting exact filenames, file sizes, and download URLs.

## Example Usage

```bash
# Inspect a popular Nous model
hermes chat -q "Inspect the Hugging Face model NousResearch/Hermes-3-Llama-3.1-8B and tell me its parameters and context length"

# Discover GGUF quantizations
hermes chat -q "List all available GGUF quants for bartowski/Hermes-3-Llama-3.1-8B-GGUF"
```
