# Hermes MCP Configuration For Future Video Studio

Future Video Studio exposes a hosted HTTP MCP server:

```text
https://mcp.future.video/mcp
```

Hermes reads MCP configuration from `~/.hermes/config.yaml` under `mcp_servers`.

## Pay-As-You-Go Mode

No FVS account key is required. Configure only the remote endpoint:

```yaml
mcp_servers:
  future_video_studio:
    url: "https://mcp.future.video/mcp"
```

Use `fvs_create_paid_render_quote` to create a Link payment quote. The response includes a `payment_url`, `quote_id`, `claim_token`, and `status_url`. After payment, poll with `fvs_get_paid_render_status`.

## Account/API-Key Mode

Set the API key outside chat:

```bash
export FVS_AGENT_API_KEY="<set through secret manager>"
```

On Windows PowerShell:

```powershell
$env:FVS_AGENT_API_KEY = "<set through secret manager>"
```

Then configure the MCP header:

```yaml
mcp_servers:
  future_video_studio:
    url: "https://mcp.future.video/mcp"
    headers:
      X-FVS-Agent-Key: "${FVS_AGENT_API_KEY}"
```

Account mode spends the owning FVS account's wallet credits. The agent must get explicit user approval before calling `fvs_submit_render`.

## Optional Explicit Billing Mode

Pay-as-you-go can also be requested explicitly:

```yaml
mcp_servers:
  future_video_studio:
    url: "https://mcp.future.video/mcp"
    headers:
      X-FVS-Billing-Mode: "pay-per-render"
```

Do not send an empty `X-FVS-Agent-Key` header if the user wants pay-as-you-go mode.

## Tool Names

Hermes prefixes MCP tools with the server name. With `future_video_studio`, tools are usually registered as:

- `mcp_future_video_studio_fvs_submit_render`
- `mcp_future_video_studio_fvs_create_paid_render_quote`
- `mcp_future_video_studio_fvs_get_render_status`
- `mcp_future_video_studio_fvs_get_paid_render_status`
- `mcp_future_video_studio_fvs_cancel_render`
- `mcp_future_video_studio_fvs_download_final_video`
- `mcp_future_video_studio_fvs_example_render_request`

If these tools are not visible, restart Hermes after editing `~/.hermes/config.yaml`.
