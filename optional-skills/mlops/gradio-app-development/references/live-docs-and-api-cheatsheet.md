# Gradio 6, Hugging Face Spaces & `daggr` — Live Docs & API Cheatsheet

## 1. Direct Documentation URLs (Fetch with `web_extract` When Unsure)

### Gradio 6 Core & Components
- **Docs Root:** https://www.gradio.app/docs
- **Guides Root:** https://www.gradio.app/guides
- **Gradio 6 Migration Guide:** https://www.gradio.app/guides/gradio-6-migration-guide
- **Blocks & Event Listeners:** https://www.gradio.app/guides/blocks-and-event-listeners
- **Controlling Layout:** https://www.gradio.app/guides/controlling-layout
- **State in Blocks:** https://www.gradio.app/guides/state-in-blocks
- **Dynamic Apps (`@gr.render`):** https://www.gradio.app/guides/dynamic-apps-with-render-decorator
- **Custom HTML (`gr.HTML` templates):** https://www.gradio.app/guides/custom-HTML-components
- **Custom CSS & JS:** https://www.gradio.app/guides/custom-CSS-and-JS
- **Streaming Outputs:** https://www.gradio.app/guides/streaming-outputs
- **Progress Bars (`gr.Progress`):** https://www.gradio.app/guides/progress-bars
- **File Access & `allowed_paths`:** https://www.gradio.app/guides/file-access
- **Themes & Styling:** https://www.gradio.app/guides/themes
- **3D Model Component (`gr.Model3D`):** https://www.gradio.app/guides/how-to-use-3D-model-component

### Gradio 6 Advanced Modes (Server Mode, Workflows, MCP, Clients)
- **Server Mode (`gradio.Server`):** https://www.gradio.app/guides/server-mode
- **Server Class API Reference:** https://www.gradio.app/docs/gradio/server
- **Workflows (`gr.Workflow`):** https://www.gradio.app/guides/workflows
- **MCP Server (`mcp_server=True`):** https://www.gradio.app/guides/building-mcp-server-with-gradio
- **MCP Client in Gradio:** https://www.gradio.app/guides/building-an-mcp-client-with-gradio
- **Python `gradio_client`:** https://www.gradio.app/guides/getting-started-with-the-python-client
- **JS `@gradio/client`:** https://www.gradio.app/guides/getting-started-with-the-js-client
- **Querying Gradio Apps with cURL:** https://www.gradio.app/guides/querying-gradio-apps-with-curl
- **Gradio Skills for Coding Assistants:** https://www.gradio.app/guides/Gradio-skills-for-ai-coding-assistants

### Hugging Face Spaces & Hub API
- **Gradio Spaces Overview:** https://huggingface.co/docs/hub/spaces-sdks-gradio
- **Docker Spaces Overview:** https://huggingface.co/docs/hub/spaces-run-with-docker
- **Spaces `README.md` YAML Config Reference:** https://huggingface.co/docs/hub/spaces-config-reference
- **ZeroGPU Guide (`@spaces.GPU`):** https://huggingface.co/docs/hub/spaces-zerogpu
- **ZeroGPU Ahead-of-Time Compilation:** https://huggingface.co/blog/zerogpu-aoti
- **Spaces GPU Hardware Tiers:** https://huggingface.co/docs/hub/spaces-gpus
- **Spaces OAuth (`hf_oauth: true`):** https://huggingface.co/docs/hub/spaces-oauth
- **Manage Spaces with `huggingface_hub` (`HfApi`):** https://huggingface.co/docs/huggingface_hub/guides/manage-spaces
- **Spaces as API Endpoints:** https://huggingface.co/docs/hub/spaces-api-endpoints

### `daggr`
- **`daggr` GitHub Repository:** https://github.com/gradio-app/daggr

---

## 2. Gradio 6 Quick Component & API Cheatsheet

### Key Breaking Changes in Gradio 6 (vs Gradio 4/5)
1. **App-level styling moved to `launch()`:**
   ```python
   # WRONG in Gradio 6:
   with gr.Blocks(theme=gr.themes.Soft(), css="...") as demo: ...

   # RIGHT in Gradio 6:
   with gr.Blocks(title="My App") as demo: ...
   demo.launch(theme=gr.themes.Soft(), css="...", js="...", head="...")
   ```
2. **`show_api` removed:**
   - In `demo.launch()`: use `footer_links=["api", "gradio", "settings"]` (default) or `footer_links=["gradio", "settings"]` to hide API link.
   - In event listeners (`.click()`, `.submit()`): use `api_visibility="public"` (default), `"undocumented"`, or `"private"` (replaces `api_name=False`).
3. **Default API endpoint names:**
   - `gr.Interface(fn=my_func, ...)` now exposes `/my_func` by default instead of `/predict`. Pass `api_name="predict"` if backward compatibility is needed.
4. **`gr.Chatbot` & `gr.ChatInterface` message format:**
   - Tuple format `[[user, bot]]` is removed in Gradio 6. Always use OpenAI-style structured messages: `[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]`.
5. **`gr.Dataframe` counts:**
   - Use `row_count=(3, "dynamic")` / `column_count=(3, "fixed")` (check Gradio 6 migration guide if migrating legacy tables).
6. **`gradio_client.Client` authentication:**
   - Use `Client("owner/space", token=os.getenv("HF_TOKEN"))` (`hf_token=` was removed).

### High-Value Components for JIT Tools

| Task | Component & Recommended Args | Input/Output Shape |
|------|------------------------------|-------------------|
| Text / Prompt | `gr.Textbox(label="Prompt", lines=3, value="...")` | `str` |
| Code Editor | `gr.Code(label="Script", language="python", lines=15)` | `str` |
| Image (Path) | `gr.Image(label="Image", type="filepath")` | `str` path on disk (best for `daggr` & `gradio_client`) |
| Image (PIL) | `gr.Image(label="Image", type="pil", image_mode="RGBA")` | `PIL.Image.Image` |
| Image Inpaint/Mask | `gr.ImageEditor(label="Edit / Mask", type="filepath")` | `{"background": path, "layers": [path, ...], "composite": path}` |
| Image Batch | `gr.Gallery(label="Results", columns=4, height="auto")` | `list[str \| tuple[str, str]]` (path or `(path, caption)`) |
| Video | `gr.Video(label="Video")` | `str` filepath (`.mp4`, `.webm`) |
| Audio | `gr.Audio(label="Audio", type="filepath")` | `str` filepath (`.wav`, `.mp3`) or `(sample_rate, np.ndarray)` |
| 3D Mesh | `gr.Model3D(label="3D Mesh (.glb/.obj)")` | `str` filepath (`.glb`, `.gltf`, `.obj`, `.stl`) |
| Structured Data | `gr.JSON(label="Manifest / Metadata")` | `dict` or `list` |
| Table | `gr.Dataframe(headers=["name", "status"], datatype=["str", "str"])` | `list[list]` or `pandas.DataFrame` |
| Plot | `gr.Plot(label="Chart")` | `matplotlib.figure.Figure` or `plotly.graph_objects.Figure` |
| File Download | `gr.File(label="Download Artifact")` | `str` filepath or `list[str]` |
| Session State | `gr.State(value={})` | Arbitrary Python object (persists per browser tab session) |
| Custom Widget | `gr.HTML(value=..., html_template="...", css_template="...")` | Supports `{{handlebars}}` and `${js_expr}` |

### Calling Any Gradio Space from Python (`gradio_client`)

```python
import os
from pathlib import Path
import tempfile
from gradio_client import Client, handle_file

client = Client("owner/space-name", token=os.getenv("HF_TOKEN"))

# 1. Inspect live endpoints first (never guess parameter names!)
api_info = client.view_api(return_format="dict")
print(list(api_info.get("named_endpoints", {}).keys()))

# 2. Call synchronous predict (blocks until queued job completes)
sample_img = str(Path(tempfile.gettempdir()) / "input.png")
result = client.predict(
    prompt="a futuristic rover",
    image=handle_file(sample_img),  # Always wrap local files with handle_file()
    api_name="/generate",
)

# 3. Or submit non-blocking job and iterate streaming updates
job = client.submit(prompt="a futuristic rover", api_name="/generate")
for update in job:
    print("Stream update:", update)
print("Final result:", job.result())
```

### Valid Hardware Flavors for HF Spaces (`request_space_hardware` / `suggested_hardware`)
- **CPU:** `"cpu-basic"` (free with HF PRO), `"cpu-upgrade"`
- **ZeroGPU:** `"zero-gpu"` (in `daggr deploy` / UI; requires `spaces` package + `@spaces.GPU` + `sdk: gradio`; up to 10 Spaces on PRO)
- **Dedicated GPUs:** `"t4-small"`, `"t4-medium"`, `"l4x1"`, `"l4x4"`, `"l40sx1"`, `"a10g-small"`, `"a10g-large"`, `"a100-large"`, `"rtx-pro-6000"`
