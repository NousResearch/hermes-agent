from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cloudflare_worker_scans_unsloth_gguf_and_exposes_http_get_fallback():
    worker = (REPO_ROOT / "cloudflare/hf-model-index-worker/src/index.ts").read_text(encoding="utf-8")
    wrangler = (REPO_ROOT / "cloudflare/hf-model-index-worker/wrangler.toml").read_text(encoding="utf-8")
    migration = (REPO_ROOT / "cloudflare/hf-model-index-worker/migrations/0001_init.sql").read_text(encoding="utf-8")

    assert 'url.pathname === "/admin/scan" && (request.method === "POST" || request.method === "GET")' in worker
    assert '"http-get-fallback"' in worker
    assert "searchUnslothGguf" in worker
    assert 'url.searchParams.set("author", "unsloth")' in worker
    assert 'url.searchParams.set("search", `${term} GGUF`)' in worker
    assert "MAX_MODEL_BYTES" in worker
    assert "isShardedModelFile" in worker
    assert "model-index:latest" in worker
    assert "payload_canonical" in worker
    assert 'crons = ["17 */6 * * *"]' in wrangler
    assert "UNSLOTH_SEARCH_TERMS" in wrangler
    assert "5368709120" in wrangler
    assert "CREATE TABLE IF NOT EXISTS models" in migration
    assert "CREATE TABLE IF NOT EXISTS worker_state" in migration


def test_android_verifies_signed_catalog_and_queues_dropdown_selection():
    client = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/HuggingFaceModelIndexClient.kt").read_text(encoding="utf-8")
    downloads_view_model = (
        REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt"
    ).read_text(encoding="utf-8")
    downloads_section = (
        REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsSection.kt"
    ).read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")

    assert "https://hf-model-index-worker.adybag14.workers.dev/models.json" in client
    assert "PUBLIC_JWK_JSON" in client
    assert "payload_canonical" in client
    assert "verifySignature(canonicalPayload, root.getJSONObject(\"signature\"))" in client
    assert "rawEcdsaToDer" in client
    assert "stableStringify(root.getJSONObject(\"payload\"))" in client
    assert "DetectedHfModel" in client
    assert "refreshDetectedModels()" in downloads_view_model
    assert "startDetectedModelDownload(" in downloads_view_model
    assert "HuggingFaceModelIndexClient.fetchDetectedModels()" in downloads_view_model
    assert "DropdownMenu(" in downloads_section
    assert "DropdownMenuItem(" in downloads_section
    assert "selectedDetectedModel" in downloads_section
    assert "strings.detectedModelCatalogTitle()" in downloads_section
    assert "detectedModelCatalogTitle" in strings
    assert "detectedModelCatalogDescription" in strings
    assert "detectedModelDropdownPlaceholder" in strings
    assert "refreshCatalog" in strings
