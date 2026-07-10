param(
    [int]$WebUiPort = 8787,
    [int]$LinePort = 8646,
    [int]$LlamaPort = 8080,
    [int]$MemoryGraphPort = 8765
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$tailscale = Get-Command tailscale.exe -ErrorAction Stop | Select-Object -First 1 -ExpandProperty Source

& $tailscale serve --bg --yes --set-path / "http://127.0.0.1:$WebUiPort"
& $tailscale serve --bg --yes --set-path /line "http://127.0.0.1:$LinePort/line"
& $tailscale serve --bg --yes --set-path /v1 "http://127.0.0.1:${LlamaPort}/v1"
& $tailscale serve --bg --yes --set-path /memory-graph "http://127.0.0.1:$MemoryGraphPort"

$dns = $null
try {
    $json = & $tailscale status --json | ConvertFrom-Json
    $dns = [string]$json.Self.DNSName
    if ($dns) { $dns = $dns.TrimEnd('.') }
} catch {}

if ($dns) {
    Write-Host "Memory graph (Tailscale): https://$dns/memory-graph/obsidian-memory-graph.html"
}

& $tailscale serve status
