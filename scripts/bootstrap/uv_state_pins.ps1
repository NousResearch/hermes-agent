# --- BEGIN canonical uv state pins (scripts/gen-bootstrap-resolver.py) ---
function Set-UvStatePins {
    # uv's default state belongs to the user's uv (#101269), so pin both to the
    # machine root unless the caller already set them. The cache keeps its own
    # slot: pm seeds <root>\cache\uv once and skips existing entries, so
    # bootstrap bytes there first would mark a partial seed done.
    $hermesRoot = Get-HermesRoot
    $cache = Join-Path $hermesRoot 'cache'
    if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $cache 'uv-bootstrap' }
    if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $cache 'uv-python' }
}
# --- END canonical uv state pins ---
