# --- BEGIN canonical uv state pins (scripts/gen-bootstrap-resolver.py) ---
function Set-UvStatePins {
    # uv's default state belongs to the USER's uv (#101269), so pin both to the
    # MACHINE root unless the caller already set them: an explicitly configured
    # value stays authoritative, exactly as the shell twins leave it. The cache
    # keeps its own slot — pm seeds <root>\cache\uv once, skipping entries that
    # already exist, so bootstrap bytes there first would mark a partial seed
    # done — while nothing seeds the python dir.
    $hermesRoot = Get-HermesRoot
    $cache = Join-Path $hermesRoot 'cache'
    if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $cache 'uv-bootstrap' }
    if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $cache 'uv-python' }
}
# --- END canonical uv state pins ---
